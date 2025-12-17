import math
import os
from typing import Tuple, List, Optional, Callable

import numpy as np
import gymnasium as gym
import torch

from gymnasium.envs.box2d.car_racing import CarRacing
from gymnasium.envs.registration import register
from gymnasium.utils.play import play
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation

from Box2D.b2 import polygonShape, fixtureDef, contactListener

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# --- Your Environment Code (Copied Verbatim) ---
# I've included this so the script is self-contained.

try:
    from gymnasium.envs.box2d import car_racing as car_mod
    _TRACK_WIDTH_DEFAULT = float(getattr(car_mod, "TRACK_WIDTH", 6.0))
except Exception:
    _TRACK_WIDTH_DEFAULT = 6.0


class _ObstacleCollisionListener(contactListener):
    def __init__(self, env, base_listener=None, penalty: float = 0.0, end_on_obstacle: bool = True):
        super().__init__()
        self.env = env
        self.base = base_listener
        self.penalty = penalty
        self.end_on_obstacle = end_on_obstacle

    def BeginContact(self, contact):
        if self.base is not None and hasattr(self.base, "BeginContact"):
            self.base.BeginContact(contact)

        u1 = getattr(contact.fixtureA.body, "userData", None)
        u2 = getattr(contact.fixtureB.body, "userData", None)

        if (u1 is not None and getattr(u1, "is_obstacle", False)) or (
            u2 is not None and getattr(u2, "is_obstacle", False)
        ):
            self.env.num_collisions += 1
            if self.penalty:
                self.env.reward += self.penalty
            if self.end_on_obstacle:
                # Flag for termination; done in step() to keep Gymnasium API order
                self.env._obstacle_collision_happened = True

    def EndContact(self, contact):
        if self.base is not None and hasattr(self.base, "EndContact"):
            self.base.EndContact(contact)


class CarRacingObstacles(CarRacing):
    metadata = CarRacing.metadata

    def __init__(
        self,
        *,
        obstacle_probability: float = 0.05,
        obstacle_min_gap: int = 15,
        obstacle_width_ratio: float = 0.30,
        obstacle_length_ratio: float = 0.15,
        obstacle_penalty: float = 0.0,
        end_on_obstacle: bool = True,   # <-- new: lose on touch
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.obstacle_probability = float(obstacle_probability)
        self.obstacle_min_gap = int(obstacle_min_gap)
        self.obstacle_width_ratio = float(obstacle_width_ratio)
        self.obstacle_length_ratio = float(obstacle_length_ratio)
        self.obstacle_penalty = float(obstacle_penalty)
        self.end_on_obstacle = bool(end_on_obstacle)

        self.obstacle_bodies = []
        self.obstacle_polys: List[Tuple[List[Tuple[float, float]], Tuple[float, float, float]]] = []
        self.num_collisions = 0
        self._obstacle_collision_happened = False

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._clear_obstacles()

        base_listener = getattr(self.world, "contactListener", None)
        self._collision_listener = _ObstacleCollisionListener(
            env=self,
            base_listener=base_listener,
            penalty=self.obstacle_penalty,
            end_on_obstacle=self.end_on_obstacle,
        )
        self.world.contactListener = self._collision_listener

        self._spawn_obstacles()
        info = dict(info)
        info["num_obstacles"] = len(self.obstacle_bodies)
        info["num_collisions"] = 0
        self._obstacle_collision_happened = False
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # If we touched an obstacle since the last step, end the episode now.
        if self._obstacle_collision_happened and not terminated and not truncated:
            terminated = True
            info = dict(info)
            info["terminated_reason"] = "obstacle_collision"

        info = dict(info)
        info["num_obstacles"] = len(self.obstacle_bodies)
        info["num_collisions"] = self.num_collisions
        return obs, reward, terminated, truncated, info

    def close(self):
        self._clear_obstacles()
        return super().close()

    def _clear_obstacles(self):
        if getattr(self, "world", None) is not None:
            for b in self.obstacle_bodies:
                try:
                    self.world.DestroyBody(b)
                except Exception:
                    pass
        self.obstacle_bodies.clear()
        if hasattr(self, "obstacle_polys") and hasattr(self, "road_poly"):
            for poly in self.obstacle_polys:
                try:
                    self.road_poly.remove(poly)
                except ValueError:
                    pass
        self.obstacle_polys.clear()
        self.num_collisions = 0
        self._obstacle_collision_happened = False

    def _spawn_obstacles(self):
        if not hasattr(self, "track") or not hasattr(self, "road_poly") or not self.track:
            return
        rng = getattr(self, "np_random", np.random.default_rng())
        gap = max(1, int(self.obstacle_min_gap))
        last_idx = -gap
        for idx in range(5, len(self.track) - 5):
            if idx - last_idx < gap:
                continue
            if rng.random() >= self.obstacle_probability:
                continue
            self._create_obstacle_for_tile(idx, rng)
            last_idx = idx

    def _create_obstacle_for_tile(self, idx: int, rng):
        _, beta, cx, cy = self.track[idx]
        try:
            p0, p1, p2, p3 = self.road_poly[idx][0]
            width_vec = np.array(p1) - np.array(p0)
            width = float(np.linalg.norm(width_vec))
            if width <= 1e-6:
                return
            width_dir = width_vec / width
            half_width = 0.5 * width
            along_vec = np.array(p3) - np.array(p0)
            along = float(np.linalg.norm(along_vec))
            if along <= 1e-6:
                along_dir = np.array([-width_dir[1], width_dir[0]])
            else:
                along_dir = along_vec / along
        except Exception:
            width_dir = np.array([math.cos(beta), math.sin(beta)], dtype=np.float32)
            half_width = getattr(self, "TRACK_WIDTH", _TRACK_WIDTH_DEFAULT)
            along_dir = np.array([-width_dir[1], width_dir[0]], dtype=np.float32)

        side = rng.choice([-1.0, 1.0])
        center_offset = side * (0.60 * half_width)
        center = np.array([cx, cy]) + width_dir * center_offset

        hw = max(0.5, self.obstacle_width_ratio * half_width)
        hh = max(0.5, self.obstacle_length_ratio * half_width)

        local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
        R = np.stack([width_dir, along_dir], axis=1)
        verts = (local @ R.T) + center

        shape = polygonShape(vertices=[tuple(v) for v in verts])
        fix = fixtureDef(shape=shape, density=0.0, friction=0.5, restitution=0.0)
        body = self.world.CreateStaticBody(fixtures=fix)
        body.userData = body
        body.is_obstacle = True
        self.obstacle_bodies.append(body)

        color = (0.55, 0.27, 0.07)
        poly = ([(float(v[0]), float(v[1])) for v in verts], color)
        self.obstacle_polys.append(poly)
        self.road_poly.append(poly)


class EnsureChannelLast(gym.ObservationWrapper):
    """
    Gymnasium's ResizeObservation drops the singleton channel axis for grayscale inputs.
    This wrapper restores the channel dimension so downstream wrappers (e.g. VecFrameStack)
    always receive observations with shape (H, W, C).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(self.observation_space, gym.spaces.Box):
            raise TypeError("EnsureChannelLast expects a Box observation space.")

        obs_space = self.observation_space
        obs_shape = obs_space.shape

        if len(obs_shape) == 2:
            low = np.expand_dims(obs_space.low, axis=-1)
            high = np.expand_dims(obs_space.high, axis=-1)
            self.observation_space = gym.spaces.Box(
                low=low,
                high=high,
                dtype=obs_space.dtype,
            )
            self._target_ndim = 3
        elif len(obs_shape) >= 3:
            self.observation_space = obs_space
            self._target_ndim = len(obs_shape)
        else:
            raise ValueError(f"Unexpected observation shape {obs_shape}; cannot ensure channel-last format.")

    def observation(self, observation):
        observation = np.asarray(observation)
        if observation.ndim < self._target_ndim:
            observation = np.expand_dims(observation, axis=-1)
        return observation


class DiscreteCarRacingAction(gym.ActionWrapper):
    """
    Map a small discrete action space to CarRacing's continuous controls.

    This makes the environment compatible with DQN while still allowing
    combined steer+gas/brake actions (unlike CarRacing's built-in 5-action mode).
    """

    def __init__(self, env: gym.Env, actions: Optional[np.ndarray] = None):
        super().__init__(env)
        if actions is None:
            actions = np.array(
                [
                    [0.0, 0.0, 0.0],   # noop
                    [-1.0, 0.0, 0.0],  # left
                    [1.0, 0.0, 0.0],   # right
                    [0.0, 0.7, 0.0],   # gas
                    [0.0, 0.0, 1.0],   # brake
                    [-1.0, 0.7, 0.0],  # left + gas
                    [1.0, 0.7, 0.0],   # right + gas
                    [-1.0, 0.0, 1.0],  # left + brake
                    [1.0, 0.0, 1.0],   # right + brake
                ],
                dtype=np.float32,
            )
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 3:
            raise ValueError(f"Expected actions with shape (n, 3), got {actions.shape}")
        self._actions = actions
        self.action_space = gym.spaces.Discrete(self._actions.shape[0])

    def action(self, action):
        return np.array(self._actions[int(action)], dtype=np.float32)


# --- Register the Environment ---
# This makes `gym.make("CarRacingObstacles-v0")` work.
register(
    id="CarRacingObstacles-v0",
    entry_point=f"{__name__}:CarRacingObstacles",
)


# --- Helper Function to Create and Wrap the Environment ---

def apply_visual_wrappers(env: gym.Env, *, monitor: bool = True) -> gym.Env:
    if monitor:
        env = Monitor(env)
    env = GrayscaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, (84, 84))
    env = EnsureChannelLast(env)
    return env


def make_discrete_action_env(env: gym.Env) -> gym.Env:
    """
    Wrap a continuous CarRacing-like env with a discrete action interface for DQN.
    """
    return DiscreteCarRacingAction(env)


def create_env(env_id: str, rank: int, seed: int = 0, **kwargs) -> Callable:
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    :param kwargs: (dict) keyword arguments for the environment
    """
    def _init() -> gym.Env:
        env = gym.make(env_id, **kwargs)
        env = apply_visual_wrappers(env, monitor=True)
        env.reset(seed=seed + rank)
        return env
    return _init

# --- Main Training and Demo Script ---

if __name__ == "__main__":

    # --- Configuration ---
    ENV_ID = "CarRacingObstacles-v0"
    LOG_DIR = "./dqn_carracing_logs/"
    MODEL_SAVE_PATH = os.path.join(LOG_DIR, "best_model")
    TOTAL_TIMESTEPS = 250_000  # Total steps to train (250k is a quick start)
    NUM_ENVS = 1              # DQN + replay buffer is memory-heavy; keep this small
    
    # Set device to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Environment keyword arguments
    # This is where you set your custom parameters
    env_kwargs = {
        "continuous": True,  # keep continuous controls; we discretize via wrapper for DQN
        "domain_randomize": False,
        "lap_complete_percent": 0.95,
        "max_episode_steps": 1500,     # Max steps per run (as requested)
        
        # --- Obstacle Parameters ---
        "end_on_obstacle": True,
        "obstacle_penalty": -20.0,    # **Punishment for hitting (as requested)**
        "obstacle_probability": 0.08,
        "obstacle_min_gap": 12,
    }

    # --- Create Vectorized Training Environment ---
    print("Creating vectorized environment...")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    def _make_env():
        env = gym.make(ENV_ID, **env_kwargs)
        env = make_discrete_action_env(env)
        env = apply_visual_wrappers(env, monitor=False)  # make_vec_env already adds Monitor
        return env

    vec_env = make_vec_env(_make_env, n_envs=NUM_ENVS, seed=0)

    # Stack 4 frames together (84, 84, 1) -> (84, 84, 4)
    # This is crucial for the agent to learn velocity
    vec_env = VecFrameStack(vec_env, n_stack=4)

    eval_env = make_vec_env(_make_env, n_envs=1, seed=10)
    eval_env = VecFrameStack(eval_env, n_stack=4)

    # --- Define DQN Model ---
    model = DQN(
        "CnnPolicy",
        vec_env,
        verbose=1,
        device=device,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=10_000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10_000,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        # SB3 ReplayBuffer can't combine optimize_memory_usage with timeout handling (default True).
        optimize_memory_usage=False,
        tensorboard_log=LOG_DIR,
    )

    # --- Setup Callback to Save Best Model ---
    # It will evaluate the agent every 5000 steps using a separate eval env
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_SAVE_PATH,
        log_path=LOG_DIR,
        eval_freq=max(5000 // NUM_ENVS, 1),
        deterministic=True,
        render=False
    )

    # --- Train the Agent ---
    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    print("You can monitor progress in TensorBoard: tensorboard --logdir ./dqn_carracing_logs/")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback
    )

    # Save the final model
    final_model_path = os.path.join(LOG_DIR, "final_model")
    model.save(final_model_path)
    print(f"Training finished. Final model saved to {final_model_path}")
    print(f"Best model saved to {os.path.join(MODEL_SAVE_PATH, 'best_model.zip')}")

    eval_env.close()
    vec_env.close()
