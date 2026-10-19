"""
Headless training loop, shared by the GUI QThread and the command line.

    python -m rl_twin.trainer scenario.json --episodes 3000 --out druid_policy
"""

import argparse
import time
from collections import deque

from .agent import DQNAgent
from .env import DruidHuntEnv, EnvConfig


def run_training(env, agent, episodes, on_progress=None, should_stop=None, progress_every=1.0):
    """
    Train `agent` on `env` for `episodes` episodes.

    on_progress(stats: dict) is called at most every `progress_every` seconds and
    after the last episode; should_stop() is polled every episode.
    """
    recent_rewards = deque(maxlen=100)
    recent_kills = deque(maxlen=100)
    recent_deaths = deque(maxlen=100)
    started = time.perf_counter()
    last_report = 0.0
    start_steps = agent.total_steps
    episode_rewards = []
    stats = {}

    for episode in range(1, episodes + 1):
        if should_stop is not None and should_stop():
            break
        obs, _ = env.reset()
        done = False
        info = {}
        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.observe(obs, action, reward, next_obs, terminated)
            obs = next_obs
            done = terminated or truncated
        agent.episodes_trained += 1

        recent_rewards.append(info["episode_reward"])
        recent_kills.append(info["kills"])
        recent_deaths.append(1.0 if info["hp"] <= 0 else 0.0)
        episode_rewards.append(info["episode_reward"])

        now = time.perf_counter()
        elapsed = now - started
        stats = {
            "episode": episode,
            "episodes_total": episodes,
            "episode_reward": info["episode_reward"],
            "avg_reward": sum(recent_rewards) / len(recent_rewards),
            "avg_kills": sum(recent_kills) / len(recent_kills),
            "death_rate": sum(recent_deaths) / len(recent_deaths),
            "epsilon": agent.epsilon,
            "loss": agent.last_loss,
            "total_steps": agent.total_steps,
            "steps_per_second": (agent.total_steps - start_steps) / max(elapsed, 1e-6),
            "elapsed": elapsed,
            "new_rewards": episode_rewards,
        }
        if on_progress is not None and (now - last_report >= progress_every or episode == episodes):
            on_progress(stats)
            episode_rewards = []
            last_report = now
    if on_progress is not None and episode_rewards:
        stats["new_rewards"] = episode_rewards
        on_progress(stats)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Trening agenta DQN (bez GUI)")
    parser.add_argument("scenario", nargs="?", help="plik scenariusza JSON zapisany z GUI")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--out", default="druid_policy", help="ścieżka eksportu (bez rozszerzenia)")
    parser.add_argument("--resume", help="kontynuuj trening z pliku .npz")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.scenario:
        with open(args.scenario, encoding="utf-8") as handle:
            config = EnvConfig.from_json(handle.read())
    else:
        config = EnvConfig()
    config.seed = args.seed
    env = DruidHuntEnv(config)
    agent = DQNAgent(seed=args.seed)
    if args.resume:
        agent.load_policy(args.resume)
        print(f"Wczytano model: {args.resume}")

    def report(stats):
        print(f"Epizod {stats['episode']}/{stats['episodes_total']} | "
              f"śr. nagroda {stats['avg_reward']:8.1f} | śr. zabicia {stats['avg_kills']:.2f} | "
              f"zgony {stats['death_rate'] * 100:5.1f}% | eps {stats['epsilon']:.3f} | "
              f"{stats['steps_per_second']:.0f} kroków/s")

    run_training(env, agent, args.episodes, on_progress=report, progress_every=5.0)
    paths = agent.export_policy(args.out, {"env_config": config.to_dict()})
    print("Zapisano model: " + ", ".join(paths))


if __name__ == "__main__":
    main()
