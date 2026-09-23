import time

import mujoco.viewer
import mujoco
import numpy as np
import threading
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


class KeyboardCommand:
    """Convert keyboard press/release events into normalized velocity commands."""

    def __init__(self):
        try:
            from pynput import keyboard
        except ImportError as exc:
            raise RuntimeError(
                "Keyboard control requires pynput. Install it with: pip install pynput"
            ) from exc

        self._keyboard = keyboard
        self._lock = threading.Lock()
        self._pressed = set()
        self.exit_requested = False
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )

    def _key_name(self, key):
        try:
            return key.char.lower()
        except AttributeError:
            special_keys = {
                self._keyboard.Key.up: "up",
                self._keyboard.Key.down: "down",
                self._keyboard.Key.left: "left",
                self._keyboard.Key.right: "right",
                self._keyboard.Key.esc: "esc",
            }
            if key in special_keys:
                return special_keys[key]
            return None

    def _on_press(self, key):
        name = self._key_name(key)
        if name == "esc":
            self.exit_requested = True
            return False
        if name in {"up", "down", "left", "right", "q", "e"}:
            with self._lock:
                self._pressed.add(name)

    def _on_release(self, key):
        name = self._key_name(key)
        if name is not None:
            with self._lock:
                self._pressed.discard(name)

    def command(self):
        with self._lock:
            pressed = set(self._pressed)

        # [forward/backward, left/right, yaw left/right]
        cmd = np.zeros(3, dtype=np.float32)
        cmd[0] = float("up" in pressed) - float("down" in pressed)
        cmd[1] = float("left" in pressed) - float("right" in pressed)
        cmd[2] = float("q" in pressed) - float("e" in pressed)
        return cmd

    def start(self):
        print("Keyboard: arrows forward/back/left/right, Q/E turn, ESC exit")
        self._listener.start()

    def stop(self):
        self._listener.stop()


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]

        # Keyboard input replaces the fixed cmd_init command.
        cmd = np.zeros(3, dtype=np.float32)
        keyboard_cmd_scale = np.array([0.3, 0.3, 0.5], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)

    keyboard_control = KeyboardCommand()
    keyboard_control.start()
    try:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            # Close the viewer automatically after simulation_duration wall-seconds.
            start = time.time()
            while (viewer.is_running() and
                   not keyboard_control.exit_requested and
                   time.time() - start < simulation_duration):
                step_start = time.time()
                tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)

                counter += 1
                if counter % control_decimation == 0:
                    # Read the current keyboard command at the policy rate.
                    cmd[:] = keyboard_control.command() * keyboard_cmd_scale

                    qj = d.qpos[7:]
                    dqj = d.qvel[6:]
                    quat = d.qpos[3:7]
                    omega = d.qvel[3:6]

                    qj = (qj - default_angles) * dof_pos_scale
                    dqj = dqj * dof_vel_scale
                    gravity_orientation = get_gravity_orientation(quat)
                    omega = omega * ang_vel_scale

                    period = 0.8
                    count = counter * simulation_dt
                    phase = count % period / period
                    sin_phase = np.sin(2 * np.pi * phase)
                    cos_phase = np.cos(2 * np.pi * phase)

                    obs[:3] = omega
                    obs[3:6] = gravity_orientation
                    obs[6:9] = cmd * cmd_scale
                    obs[9 : 9 + num_actions] = qj
                    obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                    obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                    obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                    action = policy(obs_tensor).detach().numpy().squeeze()
                    target_dof_pos = action * action_scale + default_angles

                viewer.sync()

                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    finally:
        keyboard_control.stop()
