#!/usr/bin/env python3
"""Small client that exercises the Unity policy protocol without Unity."""

import argparse
import json
import socket
import time

import numpy as np


def exchange(reader, writer, message):
    writer.write(json.dumps(message, separators=(",", ":")) + "\n")
    writer.flush()
    line = reader.readline()
    if not line:
        raise ConnectionError("policy server closed the connection")
    response = json.loads(line)
    if response.get("type") == "error":
        raise RuntimeError(response.get("error", "unknown server error"))
    return response


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9002)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--realtime", action="store_true", help="sleep 20 ms between requests")
    parser.add_argument("--command", nargs=3, type=float, default=[0.3, 0.0, 0.0])
    return parser.parse_args()


def main():
    args = parse_args()

    with socket.create_connection((args.host, args.port), timeout=5.0) as connection:
        reader = connection.makefile("r", encoding="utf-8", newline="\n")
        writer = connection.makefile("w", encoding="utf-8", newline="\n")
        metadata = exchange(reader, writer, {"type": "get_config"})
        control_dt = float(metadata["control_dt"])
        joint_position = np.asarray(
            metadata["default_joint_positions"], dtype=np.float32
        ).copy()
        joint_velocity = np.zeros(int(metadata["num_actions"]), dtype=np.float32)
        print("Connected to {} at {} Hz".format(metadata["robot"], 1.0 / control_dt))

        for step in range(args.steps):
            request = {
                "seq": step,
                "reset": step == 0,
                "debug": step == 0,
                "sim_time": step * control_dt,
                "base_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "base_angular_velocity": [0.0, 0.0, 0.0],
                "joint_position": joint_position.tolist(),
                "joint_velocity": joint_velocity.tolist(),
                "command": args.command,
            }
            response = exchange(reader, writer, request)
            if response["type"] == "hold":
                if step == 0 or (step + 1) % 10 == 0:
                    print(
                        "step={:4d} holding, remaining={:6.3f} s".format(
                            step, response["remaining_s"]
                        )
                    )
                if args.realtime:
                    time.sleep(control_dt)
                continue
            if response["type"] != "action":
                raise RuntimeError("unexpected response type: {}".format(response["type"]))

            target = np.asarray(response["target_joint_position"], dtype=np.float32)

            # This is only a fake first-order joint response, not a physics simulation.
            previous_position = joint_position.copy()
            joint_position += 0.15 * (target - joint_position)
            joint_velocity = (joint_position - previous_position) / control_dt

            if step == 0 or (step + 1) % 10 == 0:
                print(
                    "step={:4d} inference={:7.3f} ms action_norm={:7.3f}".format(
                        step, response["inference_ms"], np.linalg.norm(response["action"])
                    )
                )
            if args.realtime:
                time.sleep(control_dt)

        exchange(reader, writer, {"type": "reset", "seq": args.steps})
        writer.close()
        reader.close()


if __name__ == "__main__":
    main()
