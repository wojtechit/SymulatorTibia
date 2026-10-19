#!/usr/bin/env python3
"""
Reference inference loader for an exported policy (numpy only, no Qt, no torch).

Integration in brainstem.py:

    from brainstem_example import BrainstemPolicy
    from rl_twin.observation import GameSnapshot, MonsterView, encode_observation

    policy = BrainstemPolicy("druid_policy.npz")
    snapshot = GameSnapshot(...)          # filled from screen reading / memory / API
    action_name = policy.act_name(encode_observation(snapshot))   # e.g. "attack"

Demo (runs the policy in the simulator and prints its decisions):

    python3 brainstem_example.py druid_policy.npz --episodes 3
"""

import argparse
import json

import numpy as np


class BrainstemPolicy:
    def __init__(self, path):
        data = np.load(path, allow_pickle=False)
        self.metadata = json.loads(str(data["metadata"]))
        if self.metadata.get("format") != "rl_twin.dqn_policy":
            raise ValueError(f"Unsupported model format: {self.metadata.get('format')}")
        layer_count = len(self.metadata["layer_sizes"]) - 1
        self.weights = [data[f"W{i}"] for i in range(layer_count)]
        self.biases = [data[f"b{i}"] for i in range(layer_count)]
        self.action_names = self.metadata["action_names"]
        self.obs_size = self.metadata["obs_size"]

    def q_values(self, obs):
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != self.obs_size:
            raise ValueError(f"Expected {self.obs_size} features, got {x.shape[1]}")
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = np.maximum(x, 0.0)
        return x[0]

    def act(self, obs):
        return int(np.argmax(self.q_values(obs)))

    def act_name(self, obs):
        return self.action_names[self.act(obs)]


def demo(path, episodes):
    from rl_twin.env import DruidHuntEnv, EnvConfig

    policy = BrainstemPolicy(path)
    config = EnvConfig.from_dict(policy.metadata["env_config"]) if "env_config" in policy.metadata else EnvConfig()
    env = DruidHuntEnv(config)
    print(f"Model: {policy.metadata['algorithm']}, warstwy {policy.metadata['layer_sizes']}, "
          f"trenowany {policy.metadata['training']['episodes']} epizodów")
    for episode in range(1, episodes + 1):
        obs, _ = env.reset(seed=episode)
        counts = {name: 0 for name in policy.action_names}
        done = False
        while not done:
            action = policy.act(obs)
            counts[policy.action_names[action]] += 1
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        result = "ZGON" if info["hp"] <= 0 else ("WYGRANA" if info["monsters_alive"] == 0 else "LIMIT KROKÓW")
        print(f"Epizod {episode}: {result}, nagroda {info['episode_reward']:.1f}, "
              f"zabite {info['kills']}, kroki {info['steps']}, akcje {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test wyeksportowanej polityki")
    parser.add_argument("model", help="plik .npz z eksportu")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    demo(args.model, args.episodes)
