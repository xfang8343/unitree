#!/usr/bin/env python3

import unittest
from pathlib import Path

import numpy as np

from policy_server import ObservationBuilder, load_config, rotate_world_to_body


CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "g1.yaml"


class ObservationBuilderTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG_PATH)
        self.builder = ObservationBuilder(self.config)
        self.default_positions = np.asarray(
            self.config["default_joint_positions"], dtype=np.float32
        )

    def state(self):
        return {
            "base_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "base_angular_velocity": [1.0, 2.0, 3.0],
            "joint_position": self.default_positions.tolist(),
            "joint_velocity": [0.0] * 12,
            "command": [0.5, -0.25, 0.4],
            "sim_time": 0.0,
        }

    def test_identity_state_observation_layout(self):
        observation, command, episode_time = self.builder.build(self.state())
        self.assertEqual(observation.shape, (47,))
        np.testing.assert_allclose(observation[0:3], [0.25, 0.5, 0.75])
        np.testing.assert_allclose(observation[3:6], [0.0, 0.0, -1.0])
        np.testing.assert_allclose(observation[6:9], [1.0, -0.5, 0.1])
        np.testing.assert_allclose(observation[9:45], np.zeros(36))
        np.testing.assert_allclose(observation[45:47], [0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(command, [0.5, -0.25, 0.4])
        self.assertEqual(episode_time, 0.0)

    def test_command_is_clipped(self):
        state = self.state()
        state["command"] = [2.0, -3.0, 4.0]
        observation, command, _ = self.builder.build(state)
        np.testing.assert_allclose(command, [1.0, -1.0, 1.0])
        np.testing.assert_allclose(observation[6:9], [2.0, -2.0, 0.25])

    def test_previous_action_and_reset(self):
        action = np.arange(12, dtype=np.float32)
        self.builder.commit(action, 0.0)
        observation, _, _ = self.builder.build(self.state())
        np.testing.assert_allclose(observation[33:45], action)
        self.builder.reset()
        observation, _, _ = self.builder.build(self.state())
        np.testing.assert_allclose(observation[33:45], np.zeros(12))

    def test_world_to_body_rotation(self):
        half = np.sqrt(0.5)
        result = rotate_world_to_body([half, 0.0, 0.0, half], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-6)

    def test_configured_startup_delay_is_loaded(self):
        self.assertEqual(self.config["startup_delay_s"], 5.0)


if __name__ == "__main__":
    unittest.main()
