"""
Double DQN agent implemented in pure numpy (no framework required).

The policy is a ReLU MLP. export_policy() writes a self-describing .npz file
(weights + JSON metadata) and, when PyTorch is installed, an equivalent .pt
file whose state_dict loads into nn.Sequential(Linear, ReLU, ..., Linear).
"""

import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from .observation import (
    ACTION_NAMES,
    ATTACK_RANGE,
    FEATURE_NAMES,
    OBS_SIZE,
    SIGHT_RANGE,
    VIEW_RADIUS,
)

EXPORT_FORMAT = "rl_twin.dqn_policy"
EXPORT_VERSION = 1


@dataclass
class DQNConfig:
    hidden_sizes: tuple = (128, 128)
    gamma: float = 0.99
    learning_rate: float = 5e-4
    batch_size: int = 64
    buffer_size: int = 100_000
    warmup_steps: int = 1_000
    train_every: int = 4
    target_update_steps: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 150_000
    reward_scale: float = 0.01     # keeps -1000 / +100 rewards in a range MLPs like
    grad_clip: float = 10.0


class MLP:
    """Fully connected ReLU network with manual backprop."""

    def __init__(self, sizes, rng):
        self.sizes = list(sizes)
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(self.sizes[:-1], self.sizes[1:]):
            self.weights.append((rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in))
                                .astype(np.float32))
            self.biases.append(np.zeros(fan_out, dtype=np.float32))

    @property
    def params(self):
        return self.weights + self.biases

    def forward(self, x):
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = np.maximum(x, 0.0)
        return x

    def forward_with_cache(self, x):
        activations = [x]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = np.maximum(x, 0.0)
            activations.append(x)
        return x, activations

    def backward(self, activations, grad_out):
        """Return gradients in the same order as self.params (weights then biases)."""
        grad_w = [None] * len(self.weights)
        grad_b = [None] * len(self.biases)
        grad = grad_out
        for i in reversed(range(len(self.weights))):
            grad_w[i] = activations[i].T @ grad
            grad_b[i] = grad.sum(axis=0)
            if i > 0:
                grad = (grad @ self.weights[i].T) * (activations[i] > 0)
        return grad_w + grad_b

    def copy_from(self, other):
        for dst, src in zip(self.params, other.params):
            dst[...] = src


class Adam:
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        correction1 = 1 - self.beta1 ** self.t
        correction2 = 1 - self.beta2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.beta1
            m += (1 - self.beta1) * g
            v *= self.beta2
            v += (1 - self.beta2) * g * g
            p -= self.lr * (m / correction1) / (np.sqrt(v / correction2) + self.eps)


class ReplayBuffer:
    def __init__(self, capacity, obs_size):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.index = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        i = self.index
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = done
        self.index = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)
        return self.obs[idx], self.actions[idx], self.rewards[idx], self.next_obs[idx], self.dones[idx]


class DQNAgent:
    def __init__(self, n_actions=len(ACTION_NAMES), obs_size=OBS_SIZE, config=None, seed=None):
        self.config = config or DQNConfig()
        self.n_actions = n_actions
        self.obs_size = obs_size
        self.rng = np.random.default_rng(seed)
        sizes = [obs_size, *self.config.hidden_sizes, n_actions]
        self.online = MLP(sizes, self.rng)
        self.target = MLP(sizes, self.rng)
        self.target.copy_from(self.online)
        self.optimizer = Adam(self.online.params, self.config.learning_rate)
        self.buffer = ReplayBuffer(self.config.buffer_size, obs_size)
        self.total_steps = 0
        self.episodes_trained = 0
        self.last_loss = 0.0

    @property
    def epsilon(self):
        cfg = self.config
        fraction = min(1.0, self.total_steps / max(1, cfg.epsilon_decay_steps))
        return cfg.epsilon_start + fraction * (cfg.epsilon_end - cfg.epsilon_start)

    def q_values(self, obs):
        return self.online.forward(np.asarray(obs, dtype=np.float32)[None, :])[0]

    def act(self, obs, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(obs)))

    def observe(self, obs, action, reward, next_obs, terminated):
        """Store a transition and train when due. Truncation is not a terminal state."""
        self.buffer.add(obs, action, reward * self.config.reward_scale, next_obs, float(terminated))
        self.total_steps += 1
        cfg = self.config
        if self.buffer.size >= cfg.warmup_steps and self.total_steps % cfg.train_every == 0:
            self.last_loss = self._learn()
        if self.total_steps % cfg.target_update_steps == 0:
            self.target.copy_from(self.online)

    def _learn(self):
        cfg = self.config
        obs, actions, rewards, next_obs, dones = self.buffer.sample(cfg.batch_size, self.rng)

        # Double DQN: online net picks the action, target net evaluates it
        next_actions = np.argmax(self.online.forward(next_obs), axis=1)
        next_q = self.target.forward(next_obs)[np.arange(cfg.batch_size), next_actions]
        targets = rewards + cfg.gamma * (1.0 - dones) * next_q

        q, activations = self.online.forward_with_cache(obs)
        rows = np.arange(cfg.batch_size)
        diff = q[rows, actions] - targets
        loss = np.where(np.abs(diff) <= 1.0, 0.5 * diff ** 2, np.abs(diff) - 0.5).mean()

        grad_out = np.zeros_like(q)
        grad_out[rows, actions] = np.clip(diff, -1.0, 1.0) / cfg.batch_size
        grads = self.online.backward(activations, grad_out)
        norm = np.sqrt(sum(float((g * g).sum()) for g in grads))
        if norm > cfg.grad_clip:
            grads = [g * (cfg.grad_clip / norm) for g in grads]
        self.optimizer.step(grads)
        return float(loss)

    # ----- export / import ------------------------------------------------- #

    def export_policy(self, path, extra_metadata=None):
        """Write <path>.npz (always) and <path>.pt (if torch is available). Returns written paths."""
        base = path[:-4] if path.endswith((".npz", ".pkl")) else path
        base = base[:-3] if base.endswith(".pt") else base
        metadata = {
            "format": EXPORT_FORMAT,
            "version": EXPORT_VERSION,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "algorithm": "Double DQN",
            "layer_sizes": self.online.sizes,
            "activation": "relu",
            "obs_size": self.obs_size,
            "feature_names": FEATURE_NAMES,
            "action_names": ACTION_NAMES,
            "observation_constants": {
                "view_radius": VIEW_RADIUS,
                "sight_range": SIGHT_RANGE,
                "attack_range": ATTACK_RANGE,
            },
            "training": {
                "total_steps": self.total_steps,
                "episodes": self.episodes_trained,
                "epsilon": self.epsilon,
                "dqn_config": asdict(self.config),
            },
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        arrays = {}
        for i, (w, b) in enumerate(zip(self.online.weights, self.online.biases)):
            arrays[f"W{i}"] = w
            arrays[f"b{i}"] = b
        arrays["metadata"] = np.array(json.dumps(metadata, ensure_ascii=False))
        written = [base + ".npz"]
        np.savez(written[0], **arrays)

        try:
            import torch
        except ImportError:
            return written
        state_dict = {}
        for i, (w, b) in enumerate(zip(self.online.weights, self.online.biases)):
            # nn.Sequential index: Linear layers sit at 0, 2, 4 ... (ReLU in between)
            state_dict[f"{2 * i}.weight"] = torch.from_numpy(w.T.copy())
            state_dict[f"{2 * i}.bias"] = torch.from_numpy(b.copy())
        torch.save({"state_dict": state_dict, "metadata": metadata}, base + ".pt")
        written.append(base + ".pt")
        return written

    def load_policy(self, path):
        """Load weights from an exported .npz (training counters are kept as metadata only)."""
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data["metadata"]))
        if metadata.get("format") != EXPORT_FORMAT:
            raise ValueError("Nieznany format pliku modelu")
        if metadata["obs_size"] != self.obs_size or metadata["layer_sizes"][-1] != self.n_actions:
            raise ValueError("Model nie pasuje do przestrzeni obserwacji/akcji")
        sizes = metadata["layer_sizes"]
        self.online = MLP(sizes, self.rng)
        self.target = MLP(sizes, self.rng)
        for i in range(len(sizes) - 1):
            self.online.weights[i][...] = data[f"W{i}"]
            self.online.biases[i][...] = data[f"b{i}"]
        self.target.copy_from(self.online)
        self.optimizer = Adam(self.online.params, self.config.learning_rate)
        self.config.hidden_sizes = tuple(sizes[1:-1])
        training = metadata.get("training", {})
        self.total_steps = int(training.get("total_steps", 0))
        self.episodes_trained = int(training.get("episodes", 0))
        return metadata
