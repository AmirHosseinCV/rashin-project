#!/usr/bin/env python3
"""
Quick script to load and run the best DQN model saved by train.py.
"""

import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

import train

def _build_demo_env(env_kwargs: dict[str, Any]) -> VecFrameStack:
    env_id = "CarRacingObstacles-v0"
    demo_env_fn = lambda: train.apply_visual_wrappers(
        train.make_discrete_action_env(gym.make(env_id, **env_kwargs)),
        monitor=False,
    )
    env = DummyVecEnv([demo_env_fn])
    return VecFrameStack(env, n_stack=4)


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the best DQN model trained in train.py.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("./dqn_carracing_logs"),
        help="Directory that contains the saved models (default: ./dqn_carracing_logs).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional explicit path to a saved SB3 model (.zip). Overrides --log-dir.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of demo episodes to run (default: 1).",
    )
    parser.add_argument(
        "--render-mode",
        choices=["human", "rgb_array"],
        default="human",
        help="Render mode for the environment.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy instead of deterministic actions.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    candidate_paths = []
    if args.model_path is not None:
        candidate_paths.append(args.model_path)
    else:
        log_dir = args.log_dir.expanduser().resolve()
        candidate_paths.append(log_dir / "best_model" / "best_model.zip")
        candidate_paths.append(log_dir / "final_model.zip")

    model_path = next((path for path in candidate_paths if path is not None and path.exists()), None)
    if model_path is None:
        searched = "\n  ".join(str(path) for path in candidate_paths if path is not None)
        raise FileNotFoundError(f"Could not find a saved model. Checked:\n  {searched}")

    print(f"Loading model from: {model_path}")
    model = DQN.load(str(model_path), device=device)

    env_kwargs = {
        "continuous": True,
        "domain_randomize": False,
        "lap_complete_percent": 0.95,
        "max_episode_steps": 1500,
        "end_on_obstacle": True,
        "obstacle_penalty": -20.0,
        "obstacle_probability": 0.08,
        "obstacle_min_gap": 12,
        "render_mode": args.render_mode,
    }

    demo_env = _build_demo_env(env_kwargs)

    try:
        for episode_idx in range(args.episodes):
            obs = demo_env.reset()
            total_reward = 0.0
            num_collisions = 0
            step_count = 0

            while True:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, rewards, dones, infos = demo_env.step(action)

                total_reward += float(rewards[0])
                step_count += 1

                info = infos[0]
                if "num_collisions" in info:
                    num_collisions = info["num_collisions"]

                if bool(dones[0]):
                    terminated = bool(info.get("terminated", False))
                    truncated = bool(info.get("TimeLimit.truncated", False) or info.get("truncated", False))
                    termination_reason = info.get("terminated_reason")
                    if truncated and not termination_reason:
                        termination_reason = "time limit"
                    elif terminated and termination_reason is None:
                        termination_reason = "episode ended"

                    print(
                        f"Episode {episode_idx + 1}: reward={total_reward:.2f}, "
                        f"steps={step_count}, collisions={num_collisions}"
                    )
                    if termination_reason:
                        print(f"  reason={termination_reason}")
                    break
    finally:
        demo_env.close()


if __name__ == "__main__":
    main()
