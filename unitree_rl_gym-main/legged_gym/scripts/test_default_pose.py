#!/usr/bin/env python3
"""Measure the G1 default-pose PD equilibrium in Isaac Gym without a policy."""

import isaacgym
import torch
from isaacgym import gymtorch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


TEST_DURATION_S = 1.5
MEASURE_AFTER_S = 0.0
PRINT_INTERVAL_S = 0.5


def configure_pose_test(env_cfg):
    """Remove training disturbances so this test measures PD equilibrium only."""
    env_cfg.env.num_envs = 1
    env_cfg.env.test = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False


def set_exact_default_state(env):
    """Override the environment's randomized reset with the configured default pose."""
    env.dof_pos[:] = env.default_dof_pos
    env.dof_vel[:] = 0.0
    env.root_states[:] = env.base_init_state
    env.root_states[:, :3] += env.env_origins
    env.root_states[:, 7:13] = 0.0
    env.reset_buf.zero_()
    env.episode_length_buf.zero_()
    env.actions.zero_()
    env.last_actions.zero_()

    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_actor_root_state_tensor(env.sim)


def test_default_pose(args):
    if TEST_DURATION_S <= 0.0:
        raise ValueError("TEST_DURATION_S must be greater than zero")

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    configure_pose_test(env_cfg)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    set_exact_default_state(env)

    default_positions = env.default_dof_pos[0].detach().cpu().tolist()
    print("Default-pose PD test", flush=True)
    print("  physics dt: {:.4f} s".format(env.sim_params.dt), flush=True)
    print("  policy dt:  {:.4f} s".format(env.dt), flush=True)
    print("  duration:   {:.1f} s".format(TEST_DURATION_S), flush=True)
    print("  dof order:  {}".format(env.dof_names), flush=True)
    print("  default q:  {}".format(default_positions), flush=True)

    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    control_steps = int(TEST_DURATION_S / env.dt)
    # For short diagnostic runs, always reserve the final half for measuring.
    measure_after_s = min(MEASURE_AFTER_S, TEST_DURATION_S * 0.5)
    measure_start = int(measure_after_s / env.dt)
    print_every = max(1, int(PRINT_INTERVAL_S / env.dt))
    heights = []

    for step in range(control_steps):
        _, _, _, dones, _ = env.step(actions)
        sim_time = (step + 1) * env.dt
        height = float(env.root_states[0, 2].item())

        if bool(dones[0].item()):
            print("FAILED: robot fell or pelvis contacted the ground at {:.3f} s".format(sim_time))
            return

        if step >= measure_start:
            heights.append(height)

        if step % print_every == 0 or step == control_steps - 1:
            print("t={:5.2f} s  pelvis_z={:.5f} m".format(sim_time, height), flush=True)

    height_tensor = torch.tensor(heights)
    print("PASS: default pose remained upright for {:.1f} s".format(TEST_DURATION_S))
    print(
        "Suggested base_height_target: {:.5f} m (mean over final {:.1f} s)".format(
            float(height_tensor.mean().item()), TEST_DURATION_S - measure_after_s
        )
    )
    print(
        "Height variation: {:.5f} m (standard deviation)".format(
            float(height_tensor.std(unbiased=False).item())
        )
    )


if __name__ == "__main__":
    args = get_args()
    test_default_pose(args)
