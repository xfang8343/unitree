#!/usr/bin/env python3
"""TCP policy inference service for G1 Unity Sim2Sim validation."""

import argparse
import json
import socket
import socketserver
import threading
import time
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "g1.yaml"
MAX_MESSAGE_BYTES = 1024 * 1024


def _vector(value, size, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,):
        raise ValueError("{} must contain {} numbers, got shape {}".format(name, size, array.shape))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} contains NaN or infinity".format(name))
    return array


def _normalized_quaternion(value):
    quaternion = _vector(value, 4, "base_quaternion_wxyz")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-6:
        raise ValueError("base_quaternion_wxyz must not be a zero quaternion")
    return quaternion / norm


def rotate_world_to_body(quaternion_wxyz, vector_world):
    """Rotate a world-frame vector into the robot body frame."""
    quaternion = _normalized_quaternion(quaternion_wxyz)
    vector = _vector(vector_world, 3, "world vector")
    w = quaternion[0]
    xyz = quaternion[1:]
    return (
        vector * (2.0 * w * w - 1.0)
        - 2.0 * w * np.cross(xyz, vector)
        + 2.0 * xyz * np.dot(xyz, vector)
    ).astype(np.float32)


def load_config(path):
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", str(PROJECT_ROOT))
    config["policy_path"] = str(Path(policy_path).expanduser().resolve())
    config["config_path"] = str(config_path)

    num_actions = int(config["num_actions"])
    for key in ("joint_names", "default_joint_positions", "stiffness", "damping"):
        if len(config[key]) != num_actions:
            raise ValueError("{} must contain {} entries".format(key, num_actions))
    if int(config["num_observations"]) != 47 or num_actions != 12:
        raise ValueError("This server currently implements the G1 47-observation, 12-action contract")
    return config


class ObservationBuilder:
    """Build the exact 47-value G1 policy observation from Unity state."""

    def __init__(self, config):
        self.control_dt = float(config["control_dt"])
        self.gait_period = float(config["gait_period"])
        self.num_actions = int(config["num_actions"])
        self.default_positions = _vector(
            config["default_joint_positions"], self.num_actions, "default_joint_positions"
        )
        scales = config["observation_scales"]
        self.angular_velocity_scale = float(scales["angular_velocity"])
        self.command_scale = _vector(scales["command"], 3, "observation_scales.command")
        self.joint_position_scale = float(scales["joint_position"])
        self.joint_velocity_scale = float(scales["joint_velocity"])
        limits = config["command_limits"]
        self.command_minimum = _vector(limits["minimum"], 3, "command_limits.minimum")
        self.command_maximum = _vector(limits["maximum"], 3, "command_limits.maximum")
        self.previous_action = np.zeros(self.num_actions, dtype=np.float32)
        self.elapsed_time = 0.0

    def reset(self):
        self.previous_action.fill(0.0)
        self.elapsed_time = 0.0

    def build(self, state):
        quaternion = _normalized_quaternion(state["base_quaternion_wxyz"])
        if "base_angular_velocity" in state:
            # The default Unity protocol defines this value in the pelvis/body frame.
            angular_velocity_body = _vector(
                state["base_angular_velocity"], 3, "base_angular_velocity"
            )
        elif "base_angular_velocity_body" in state:
            angular_velocity_body = _vector(
                state["base_angular_velocity_body"], 3, "base_angular_velocity_body"
            )
        else:
            angular_velocity_world = _vector(
                state["base_angular_velocity_world"], 3, "base_angular_velocity_world"
            )
            angular_velocity_body = rotate_world_to_body(quaternion, angular_velocity_world)

        projected_gravity = rotate_world_to_body(quaternion, [0.0, 0.0, -1.0])
        joint_position = _vector(state["joint_position"], self.num_actions, "joint_position")
        joint_velocity = _vector(state["joint_velocity"], self.num_actions, "joint_velocity")
        command = np.clip(
            _vector(state["command"], 3, "command"),
            self.command_minimum,
            self.command_maximum,
        ).astype(np.float32)

        # sim_time is the public Unity field. episode_time remains supported for
        # compatibility with earlier clients.
        episode_time = float(state.get("sim_time", state.get("episode_time", self.elapsed_time)))
        if not np.isfinite(episode_time) or episode_time < 0.0:
            raise ValueError("sim_time must be a finite, non-negative number")
        phase = (episode_time % self.gait_period) / self.gait_period
        phase_features = np.array(
            [np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)],
            dtype=np.float32,
        )

        observation = np.concatenate(
            (
                angular_velocity_body * self.angular_velocity_scale,
                projected_gravity,
                command * self.command_scale,
                (joint_position - self.default_positions) * self.joint_position_scale,
                joint_velocity * self.joint_velocity_scale,
                self.previous_action,
                phase_features,
            )
        ).astype(np.float32)
        if observation.shape != (47,):
            raise RuntimeError("Observation contract error: expected 47 values")
        return observation, command, episode_time

    def commit(self, action, episode_time):
        self.previous_action[:] = action
        self.elapsed_time = episode_time + self.control_dt


class PolicyService:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        policy_path = Path(config["policy_path"])
        if not policy_path.is_file():
            raise FileNotFoundError(
                "TorchScript policy not found: {}. Run play.py to export it first.".format(policy_path)
            )
        self.policy = torch.jit.load(str(policy_path), map_location=self.device)
        self.policy.eval()
        self.observation_builder = ObservationBuilder(config)
        self.default_positions = _vector(config["default_joint_positions"], 12, "default positions")
        self.action_scale = float(config["action_scale"])
        self.startup_delay_s = float(config.get("startup_delay_s", 0.0))
        if not np.isfinite(self.startup_delay_s) or self.startup_delay_s < 0.0:
            raise ValueError("startup_delay_s must be a finite value greater than or equal to zero")
        self.lock = threading.Lock()
        self.warmup_start_sim_time = None
        self.warmup_complete = False
        self.reset()
        with torch.no_grad():
            self.policy(torch.zeros((1, 47), dtype=torch.float32, device=self.device))
        self.reset()

    def reset(self):
        with self.lock:
            self._reset_policy_memory_locked()
            self._restart_warmup_locked()

    def _reset_policy_memory_locked(self):
        self.observation_builder.reset()
        if hasattr(self.policy, "reset_memory"):
            self.policy.reset_memory()

    def _restart_warmup_locked(self):
        self.warmup_start_sim_time = None
        self.warmup_complete = False

    def _warmup_response(self, request, remaining_s):
        return {
            "type": "hold",
            "seq": request.get("seq"),
            "reason": "startup_delay",
            "remaining_s": max(0.0, float(remaining_s)),
        }

    def infer(self, request):
        with self.lock:
            if bool(request.get("reset", False)):
                self._reset_policy_memory_locked()
                self._restart_warmup_locked()

            # Validate Unity state but do not update the LSTM during warmup.
            observation, command, episode_time = self.observation_builder.build(request)
            if self.startup_delay_s > 0.0 and not self.warmup_complete:
                if self.warmup_start_sim_time is None:
                    self.warmup_start_sim_time = episode_time

                elapsed = max(0.0, episode_time - self.warmup_start_sim_time)
                remaining = self.startup_delay_s - elapsed
                if remaining > 0.0:
                    return self._warmup_response(request, remaining)

                # Start from a clean recurrent state once Unity has held the
                # default standing pose for the requested duration.
                self._reset_policy_memory_locked()
                self.warmup_start_sim_time = None
                self.warmup_complete = True
                observation, command, episode_time = self.observation_builder.build(request)

            tensor = torch.from_numpy(observation).unsqueeze(0).to(self.device)
            start = time.perf_counter()
            with torch.no_grad():
                output = self.policy(tensor)
            inference_ms = (time.perf_counter() - start) * 1000.0
            action = output.detach().to("cpu").numpy().reshape(-1).astype(np.float32)
            if action.shape != (12,) or not np.all(np.isfinite(action)):
                raise RuntimeError("Policy must return 12 finite actions, got {}".format(action.shape))

            target_position = self.default_positions + action * self.action_scale
            self.observation_builder.commit(action, episode_time)
            response = {
                "type": "action",
                "seq": request.get("seq"),
                "action": action.tolist(),
                "target_joint_position": target_position.tolist(),
                "command_used": command.tolist(),
                "inference_ms": inference_ms,
            }
            if bool(request.get("debug", False)):
                response["observation"] = observation.tolist()
            return response

    def metadata(self):
        return {
            "type": "config",
            "protocol_version": 1,
            "robot": "unitree_g1_12dof",
            "control_dt": float(self.config["control_dt"]),
            "startup_delay_s": self.startup_delay_s,
            "num_observations": 47,
            "num_actions": 12,
            "joint_names": self.config["joint_names"],
            "default_joint_positions": self.config["default_joint_positions"],
            "stiffness": self.config["stiffness"],
            "damping": self.config["damping"],
            "coordinate_frame": "right-handed: x forward, y left, z up",
            "quaternion_order": "wxyz",
            "angle_unit": "radian",
        }

    def handle(self, request):
        message_type = request.get("type", "infer")
        if message_type == "infer":
            return self.infer(request)
        if message_type == "reset":
            self.reset()
            return {"type": "reset_ok", "seq": request.get("seq")}
        if message_type == "get_config":
            return self.metadata()
        if message_type == "ping":
            return {"type": "pong", "seq": request.get("seq")}
        raise ValueError("Unknown message type: {}".format(message_type))


class PolicyRequestHandler(socketserver.StreamRequestHandler):
    def setup(self):
        super().setup()
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def handle(self):
        peer = "{}:{}".format(*self.client_address)
        print("Unity client connected: {}".format(peer), flush=True)
        message_count = 0
        try:
            for raw_line in self.rfile:
                request = {}
                if len(raw_line) > MAX_MESSAGE_BYTES:
                    response = {"type": "error", "error": "message exceeds 1 MiB"}
                else:
                    try:
                        request = json.loads(raw_line.decode("utf-8"))
                        if not isinstance(request, dict):
                            raise ValueError("message must be a JSON object")
                        if (self.server.print_messages and
                                message_count % self.server.print_every == 0):
                            print(
                                "RX {}: {}".format(
                                    peer, json.dumps(request, ensure_ascii=False)
                                ),
                                flush=True,
                            )
                        response = self.server.policy_service.handle(request)
                    except Exception as exc:
                        response = {
                            "type": "error",
                            "seq": request.get("seq"),
                            "error": str(exc),
                        }
                if (self.server.print_messages and
                        message_count % self.server.print_every == 0):
                    print(
                        "TX {}: {}".format(
                            peer, json.dumps(response, ensure_ascii=False)
                        ),
                        flush=True,
                    )
                payload = json.dumps(response, separators=(",", ":"), allow_nan=False) + "\n"
                self.wfile.write(payload.encode("utf-8"))
                self.wfile.flush()
                message_count += 1
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.server.policy_service.reset()
            print("Unity client disconnected: {}".format(peer), flush=True)


class PolicyTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

    def __init__(self, address, policy_service, print_messages=False, print_every=1):
        self.policy_service = policy_service
        self.print_messages = print_messages
        self.print_every = print_every
        super().__init__(address, PolicyRequestHandler)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to service YAML")
    parser.add_argument("--host", help="override listen address")
    parser.add_argument("--port", type=int, help="override listen port")
    parser.add_argument("--policy", help="override TorchScript policy path")
    parser.add_argument("--device", help="override torch device, for example cpu or cuda:0")
    parser.add_argument(
        "--startup-delay",
        type=float,
        help="override Unity standing warmup time in seconds; 0 disables it",
    )
    parser.add_argument(
        "--print-messages",
        action="store_true",
        help="print received Unity JSON and returned policy JSON",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="with --print-messages, print one message every N requests",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.host is not None:
        config["host"] = args.host
    if args.port is not None:
        config["port"] = args.port
    if args.policy is not None:
        config["policy_path"] = str(Path(args.policy).expanduser().resolve())
    if args.device is not None:
        config["device"] = args.device
    if args.startup_delay is not None:
        config["startup_delay_s"] = args.startup_delay
    if args.print_every < 1:
        raise ValueError("--print-every must be at least 1")

    service = PolicyService(config)
    address = (config["host"], int(config["port"]))
    with PolicyTCPServer(
        address,
        service,
        print_messages=args.print_messages,
        print_every=args.print_every,
    ) as server:
        print("G1 Unity policy server", flush=True)
        print("  listen: {}:{}".format(*address), flush=True)
        print("  policy: {}".format(config["policy_path"]), flush=True)
        print("  device: {}".format(config["device"]), flush=True)
        print("  startup delay: {:.3f} s".format(service.startup_delay_s), flush=True)
        print("  protocol: one JSON object per line, 50 Hz", flush=True)
        if args.print_messages:
            print("  logging: print one RX/TX pair every {} requests".format(args.print_every), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping policy server.", flush=True)


if __name__ == "__main__":
    main()
