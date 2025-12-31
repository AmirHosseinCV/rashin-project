import os
import numpy as np
import gymnasium as gym
import torch
import random

from gymnasium.envs.registration import register
from gymnasium.wrappers import ResizeObservation
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import carracing_obstacles

# --- ۱. تغییر در Action Wrapper برای پذیرش ضریب سرعت ---
class DiscreteCarRacingAction(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # ضریب سرعت پیش‌فرض
        self.current_speed_factor = 1.0
        
        # مقادیر پایه اکشن‌ها [steer, gas, brake]
        self._base_actions = np.array(
            [
                [0.0, 0.0, 0.0],    # 0: Do nothing
                [-1.0, 0.0, 0.0],   # 1: Left (Hard)
                [1.0, 0.0, 0.0],    # 2: Right (Hard)
                [-0.5, 0.0, 0.0],   # 3: Left (Soft)
                [0.5, 0.0, 0.0],    # 4: Right (Soft)
                [0.0, 0.8, 0.0],    # 5: Gas
                [0.0, 0.0, 0.8],    # 6: Brake
                [-1.0, 0.6, 0.0],   # 7: Hard Left + Gas
                [1.0, 0.6, 0.0],    # 8: Hard Right + Gas
                [-0.5, 0.6, 0.0],   # 9: Soft Left + Gas
                [0.5, 0.6, 0.0],    # 10: Soft Right + Gas
            ],
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(len(self._base_actions))

    def action(self, action_idx):
        act = self._base_actions[int(action_idx)].copy()
        # اعمال ضریب سرعت فقط روی مقدار Gas (ایندکس 1)
        act[1] *= self.current_speed_factor
        return act

# --- ۲. Wrapper جدید برای تغییر سرعت در هر Reset ---
class VariableSpeedWrapper(gym.Wrapper):
    def __init__(self, env, speed_options):
        super().__init__(env)
        self.speed_options = speed_options

    def reset(self, **kwargs):
        # انتخاب یکی از سرعت‌هایی که شما در لیست پایین تعیین کردید
        chosen_speed = random.choice(self.speed_options)
        
        # پیدا کردن اکشن ورپر و تزریق سرعت به آن
        curr_env = self.env
        while curr_env is not None:
            if isinstance(curr_env, DiscreteCarRacingAction):
                curr_env.current_speed_factor = chosen_speed
                break
            curr_env = getattr(curr_env, 'env', None)
            
        return self.env.reset(**kwargs)

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

# --- آماده‌سازی محیط با اعمال تنظیمات سرعت ---
def make_env_instance(env_id, speed_options, **kwargs):
    env = gym.make(env_id, **kwargs)
    env = Monitor(env)
    env = DiscreteCarRacingAction(env) # اول اکشن‌ها را تعریف می‌کنیم
    env = VariableSpeedWrapper(env, speed_options) # سپس مدیریت سرعت در ریست
    env = ResizeObservation(env, (64, 64))
    env = TransposeObservation(env)
    return env

if __name__ == "__main__":
    # --- ۳. تنظیمات سرعت رانندگی ---
    # شما می‌توانید هر چند تا سرعت که می‌خواهید اینجا اضافه کنید
    # مثلا [0.4, 0.7, 1.2] باعث می‌شود ماشین در هر دور یکی از این سرعت‌ها را تجربه کند
    #88888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888
    SPEED_OPTIONS = [0.5] 

    ENV_ID = "CarRacingObstacles-v0"
    LOG_DIR = "./dqn_carracing_logs_multispeed/"
    MODEL_SAVE_PATH = os.path.join(LOG_DIR, "best_model")
    #8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888
    TOTAL_TIMESTEPS = 100_000  
    NUM_ENVS = 1 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} | Training with speeds: {SPEED_OPTIONS}")

    env_kwargs = {
        "continuous": True, 
        "domain_randomize": False,
        "lap_complete_percent": 0.95,
        "max_episode_steps": 1200, # زمان کمی بیشتر برای سرعت‌های پایین
        "end_on_obstacle": True,
        "obstacle_penalty": -20.0, 
        "obstacle_probability": 0.08,
        "obstacle_min_gap": 12,
        "render_mode": "rgb_array"
    }

    os.makedirs(LOG_DIR, exist_ok=True)

    vec_env = make_vec_env(
        lambda: make_env_instance(ENV_ID, SPEED_OPTIONS, **env_kwargs), 
        n_envs=NUM_ENVS, 
        seed=42
    )
    vec_env = VecFrameStack(vec_env, n_stack=4, channels_order='first')

    # محیط ارزیابی (تست روی سرعت متوسط)
    eval_env = make_vec_env(
        lambda: make_env_instance(ENV_ID, [1.0], **env_kwargs), 
        n_envs=1, 
        seed=123
    )
    eval_env = VecFrameStack(eval_env, n_stack=4, channels_order='first')

    model = DQN(
        "CnnPolicy",
        vec_env,
        verbose=1,
        device=device,
        buffer_size=100_000,
        learning_rate=1e-4,
        batch_size=64,
        learning_starts=50_000,
        target_update_interval=5000,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        gamma=0.99,
        tensorboard_log=LOG_DIR,
        policy_kwargs={"normalize_images": True}
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_SAVE_PATH,
        log_path=LOG_DIR,
        eval_freq=10_000,
        deterministic=True,
        render=False
    )

    print(f"Starting training...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, progress_bar=True)

    model.save(os.path.join(LOG_DIR, "final_model_multispeed"))
    print("Training finished.")