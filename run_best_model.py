#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gymnasium.wrappers import ResizeObservation

# ایمپورت فایل موانع برای رجیستر شدن محیط
import carracing_obstacles 

# --- کپی Wrappers از فایل train برای اطمینان از یکسان بودن محیط تست و آموزش ---
class DiscreteCarRacingAction(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._actions = np.array(
            [
                [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                [-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], # Soft turns
                [0.0, 0.8, 0.0], [0.0, 0.0, 0.8],
                [-1.0, 0.6, 0.0], [1.0, 0.6, 0.0],
                [-0.5, 0.6, 0.0], [0.5, 0.6, 0.0],
            ],
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(self._actions.shape[0])

    def action(self, action):
        return self._actions[int(action)]

class TransposeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(obs_shape[2], obs_shape[0], obs_shape[1]), 
            dtype=self.observation_space.dtype
        )

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

# --- تابع ساخت محیط دمو ---
def _build_demo_env(env_kwargs: dict[str, Any]) -> VecFrameStack:
    env_id = "CarRacingObstacles-v0"
    
    def demo_env_fn():
        env = gym.make(env_id, **env_kwargs)
        env = DiscreteCarRacingAction(env) 
        env = ResizeObservation(env, (64, 64))
        env = TransposeObservation(env)
        return env

    env = DummyVecEnv([demo_env_fn])
    # بسیار مهم: channels_order='first'
    return VecFrameStack(env, n_stack=4, channels_order='first')

def _parse_args():
    parser = argparse.ArgumentParser(description="Run the best DQN model.")
    parser.add_argument("--log-dir", type=Path, default=Path("./dqn_carracing_logs"))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--render-mode", choices=["human", "rgb_array"], default="human")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = _parse_args()
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # جستجو برای پیدا کردن مدل
    candidate_paths = []
    if args.model_path:
        candidate_paths.append(args.model_path)
    else:
        log_dir = args.log_dir.expanduser().resolve()
        candidate_paths.append(log_dir / "final_model_optimized.zip") 
        candidate_paths.append(log_dir / "best_model" / "best_model.zip")
        candidate_paths.append(log_dir / "final_model.zip")

    model_path = next((path for path in candidate_paths if path is not None and path.exists()), None)
    if not model_path:
        raise FileNotFoundError(f"No model found. Checked: {candidate_paths}")

    print(f"Loading model from: {model_path}")
    model = DQN.load(str(model_path), device=device)

    # تنظیمات محیط تست
    env_kwargs = {
        "continuous": True,
        "domain_randomize": False,
        "lap_complete_percent": 0.95,
        "max_episode_steps": 1500,
        "end_on_obstacle": True,
        "obstacle_penalty": -10.0, 
        "obstacle_probability": 0.08,
        "obstacle_min_gap": 12,
        "render_mode": args.render_mode,
    }

    demo_env = _build_demo_env(env_kwargs)

    try:
        for episode_idx in range(args.episodes):
            obs = demo_env.reset()
            total_reward = 0.0
            step_count = 0
            done = False
            
            print(f"--- Episode {episode_idx + 1} Start ---")
            while not done:
                # پیش‌بینی اکشن توسط مدل
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, rewards, dones, infos = demo_env.step(action)
                
                done = dones[0]
                total_reward += float(rewards[0])
                step_count += 1
                
                if done:
                    info = infos[0]
                    col = info.get("num_collisions", 0)
                    reason = info.get("terminated_reason", "End")
                    print(f"Finished: Reward={total_reward:.2f}, Steps={step_count}, Collisions={col}, Reason={reason}")
    finally:
        demo_env.close()

if __name__ == "__main__":
    main()
