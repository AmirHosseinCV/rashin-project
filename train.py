import os
import numpy as np
import gymnasium as gym
import torch

from gymnasium.envs.registration import register
from gymnasium.wrappers import ResizeObservation
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# این خط باعث می‌شود محیط شما در Gym ثبت شود
# مطمئن شوید فایل carracing_obstacles.py کنار همین فایل است
import carracing_obstacles

# --- Wrapper Action اصلاح شده (فرمان نرم) ---
class DiscreteCarRacingAction(gym.ActionWrapper):
    """
    اکشن‌های گسسته بهبود یافته.
    شامل فرمان نرم (0.5) برای جلوگیری از حرکات مارپیچ شدید.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # فرمت: [steer, gas, brake]
        self._actions = np.array(
            [
                [0.0, 0.0, 0.0],    # 0: Do nothing
                [-1.0, 0.0, 0.0],   # 1: Left (Hard)
                [1.0, 0.0, 0.0],    # 2: Right (Hard)
                [-0.5, 0.0, 0.0],   # 3: Left (Soft) - جدید برای دقت بیشتر
                [0.5, 0.0, 0.0],    # 4: Right (Soft) - جدید برای دقت بیشتر
                [0.0, 0.8, 0.0],    # 5: Gas
                [0.0, 0.0, 0.8],    # 6: Brake
                [-1.0, 0.6, 0.0],   # 7: Hard Left + Gas
                [1.0, 0.6, 0.0],    # 8: Hard Right + Gas
                [-0.5, 0.6, 0.0],   # 9: Soft Left + Gas - جدید
                [0.5, 0.6, 0.0],    # 10: Soft Right + Gas - جدید
            ],
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(self._actions.shape[0])

    def action(self, action):
        return self._actions[int(action)]

# --- Wrapper برای اصلاح ترتیب کانال‌ها (PyTorch Standard) ---
class TransposeObservation(gym.ObservationWrapper):
    """
    تغییر ابعاد تصویر از (Height, Width, Channel) به (Channel, Height, Width).
    این کار برای شبکه‌های PyTorch (CnnPolicy) ضروری است.
    """
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(obs_shape[2], obs_shape[0], obs_shape[1]), 
            dtype=self.observation_space.dtype
        )

    def observation(self, observation):
        # جابجایی محورها: HWC -> CHW
        return np.moveaxis(observation, 2, 0)

# --- تابع آماده‌سازی محیط ---
def make_env_instance(env_id, **kwargs):
    env = gym.make(env_id, **kwargs)
    env = Monitor(env)  # برای لاگ برداری
    env = DiscreteCarRacingAction(env) # تبدیل اکشن‌ها
    env = ResizeObservation(env, (64, 64)) # کاهش سایز برای سرعت (64x64)
    # نکته: ما Grayscale را حذف کردیم تا رنگ‌ها (موانع vs جاده) دیده شوند
    env = TransposeObservation(env) # اصلاح کانال‌ها
    return env

# --- بدنه اصلی آموزش ---
if __name__ == "__main__":

    # تنظیمات
    ENV_ID = "CarRacingObstacles-v0"
    LOG_DIR = "./dqn_carracing_logs/"
    MODEL_SAVE_PATH = os.path.join(LOG_DIR, "best_model")
    
    # تعداد قدم‌های آموزش (پیشنهاد: حداقل 1 میلیون برای نتیجه خوب)
    TOTAL_TIMESTEPS = 1_000_000  
    NUM_ENVS = 1 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # پارامترهای محیط (Environment Kwargs)
    env_kwargs = {
        "continuous": True, 
        "domain_randomize": False,
        "lap_complete_percent": 0.95,
        "max_episode_steps": 1000, 
        "end_on_obstacle": True,
        "obstacle_penalty": -10.0, # جریمه برخورد
        "obstacle_probability": 0.08,
        "obstacle_min_gap": 12,
        "render_mode": "rgb_array"
    }

    print("Creating vectorized environment...")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # ساخت محیط وکتورایز شده
    vec_env = make_vec_env(
        lambda: make_env_instance(ENV_ID, **env_kwargs), 
        n_envs=NUM_ENVS, 
        seed=42
    )
    
    # Frame Stacking
    # channels_order='first' بسیار مهم است چون ما با TransposeObservation کانال را اول آوردیم
    # ورودی نهایی: (Batch, 12, 64, 64) -> 3 رنگ * 4 فریم
    vec_env = VecFrameStack(vec_env, n_stack=4, channels_order='first')


    # محیط ارزیابی (Evaluation Environment)
    eval_env = make_vec_env(
        lambda: make_env_instance(ENV_ID, **env_kwargs), 
        n_envs=1, 
        seed=123
    )
    eval_env = VecFrameStack(eval_env, n_stack=4, channels_order='first')

    # تعریف مدل DQN
    model = DQN(
        "CnnPolicy",
        vec_env,
        verbose=1,
        device=device,
        buffer_size=100_000,       # بافر بزرگتر برای یادگیری پایدارتر
        learning_rate=1e-4,        
        batch_size=64,             
        learning_starts=50_000,    # 50 هزار قدم اول فقط دیتا جمع می‌کند (آموزش نمی‌بیند)
        target_update_interval=5000, 
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.20, # 20% کل زمان آموزش را صرف جستجو می‌کند
        exploration_final_eps=0.05,
        gamma=0.99,
        optimize_memory_usage=False, 
        tensorboard_log=LOG_DIR,
        policy_kwargs={"normalize_images": True} # نرمال‌سازی پیکسل‌ها (0-1)
    )

    # تنظیم ذخیره بهترین مدل
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_SAVE_PATH,
        log_path=LOG_DIR,
        eval_freq=10_000,
        deterministic=True,
        render=False
    )

    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True
    )

    # ذخیره مدل نهایی
    final_model_path = os.path.join(LOG_DIR, "final_model_optimized")
    model.save(final_model_path)
    print(f"Training finished. Final model saved to {final_model_path}")

    eval_env.close()
    vec_env.close()