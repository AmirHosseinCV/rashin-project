"""
CarRacing with static obstacles, lose-on-touch, and terminal score logging.

Usage:
    pip install swig "gymnasium[box2d]" pygame numpy
    python car_racing_obstacles.py

Controls: W=gas, S=brake, A/D=steer (combos like WA or WD). Close the window to quit.
"""

import math
from typing import Tuple, List

import numpy as np
import gymnasium as gym

from gymnasium.envs.box2d.car_racing import CarRacing
from gymnasium.envs.registration import register
from gymnasium.utils.play import play

# Box2D
from Box2D.b2 import polygonShape, fixtureDef, contactListener


# --- best-effort fallback for a track width constant used in placement math
try:
    from gymnasium.envs.box2d import car_racing as car_mod
    _TRACK_WIDTH_DEFAULT = float(getattr(car_mod, "TRACK_WIDTH", 6.0))
except Exception:
    _TRACK_WIDTH_DEFAULT = 6.0


# ======================
# Collision listener
# ======================
class _ObstacleCollisionListener(contactListener):
    """
    Wrap the base CarRacing contact listener so we keep its normal behavior
    (tile friction / completion logic) and also detect obstacle collisions,
    optionally applying a score penalty and flagging for termination.
    """
    def __init__(self, env, base_listener=None, penalty: float = 0.0, end_on_obstacle: bool = True):
        super().__init__()
        self.env = env
        self.base = base_listener
        self.penalty = float(penalty)
        self.end_on_obstacle = bool(end_on_obstacle)

    def BeginContact(self, contact):
        # Preserve original logic
        if self.base is not None and hasattr(self.base, "BeginContact"):
            self.base.BeginContact(contact)

        u1 = getattr(contact.fixtureA.body, "userData", None)
        u2 = getattr(contact.fixtureB.body, "userData", None)

        if (u1 is not None and getattr(u1, "is_obstacle", False)) or (
            u2 is not None and getattr(u2, "is_obstacle", False)
        ):
            self.env.num_collisions += 1
            if self.penalty:
                # CarRacing internally keeps a running total in self.reward
                self.env.reward += self.penalty
            if self.end_on_obstacle:
                # Mark for termination; actual termination done in step() to respect API
                self.env._obstacle_collision_happened = True

    def EndContact(self, contact):
        if self.base is not None and hasattr(self.base, "EndContact"):
            self.base.EndContact(contact)


# ======================
# Environment subclass
# ======================
class CarRacingObstacles(CarRacing):
    """
    CarRacing with static Box2D obstacles placed on some road tiles.
    Touching an obstacle can immediately terminate the episode (configurable).
    """
    metadata = CarRacing.metadata  # keep parent render metadata

    def __init__(
        self,
        *,
        # Obstacle generation controls
        obstacle_probability: float = 0.06,   # chance to place an obstacle on an eligible tile
        obstacle_min_gap: int = 14,           # min tile gap between consecutive obstacles
        obstacle_width_ratio: float = 0.32,   # half-width as fraction of half road width
        obstacle_length_ratio: float = 0.18,  # half-length as fraction of half road width
        # Gameplay effects
        obstacle_penalty: float = 0.0,        # score penalty on first contact (e.g., -5.0)
        end_on_obstacle: bool = True,         # lose immediately on contact
        **kwargs,
    ):
        """
        kwargs pass through to CarRacing:
          - continuous=True/False
          - domain_randomize=True/False
          - lap_complete_percent=0.95
          - render_mode in {"rgb_array","human"} (use "rgb_array" for play())
        """
        super().__init__(**kwargs)

        # Config
        self.obstacle_probability = float(obstacle_probability)
        self.obstacle_min_gap = int(obstacle_min_gap)
        self.obstacle_width_ratio = float(obstacle_width_ratio)
        self.obstacle_length_ratio = float(obstacle_length_ratio)
        self.obstacle_penalty = float(obstacle_penalty)
        self.end_on_obstacle = bool(end_on_obstacle)

        # State
        self.obstacle_bodies: List = []
        self.obstacle_polys: List[Tuple[List[Tuple[float, float]], Tuple[float, float, float]]] = []
        self.num_collisions = 0
        self._obstacle_collision_happened = False

    # ------------- Gymnasium API -------------
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # Clean up from previous episodes
        self._clear_obstacles()

        # Wrap contact listener
        base_listener = getattr(self.world, "contactListener", None)
        self._collision_listener = _ObstacleCollisionListener(
            env=self,
            base_listener=base_listener,
            penalty=self.obstacle_penalty,
            end_on_obstacle=self.end_on_obstacle,
        )
        self.world.contactListener = self._collision_listener

        # Create obstacles for this newly generated track
        self._spawn_obstacles()

        info = dict(info)
        info["num_obstacles"] = len(self.obstacle_bodies)
        info["num_collisions"] = 0
        self._obstacle_collision_happened = False
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # If we hit an obstacle since last step, end now
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

    # ------------- Obstacle internals -------------
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
        """
        Choose tiles along the track and place oriented static rectangles offset
        left/right from the centerline. Also append their quads to self.road_poly
        so they're visible in rgb_array renders.
        """
        if not hasattr(self, "track") or not hasattr(self, "road_poly") or not self.track:
            return

        rng = getattr(self, "np_random", np.random.default_rng())
        gap = max(1, int(self.obstacle_min_gap))
        last_idx = -gap

        # skip a few tiles near start/finish to avoid unfair spawns
        for idx in range(5, len(self.track) - 5):
            if idx - last_idx < gap:
                continue
            if rng.random() >= self.obstacle_probability:
                continue

            self._create_obstacle_for_tile(idx, rng)
            last_idx = idx

    def _create_obstacle_for_tile(self, idx: int, rng):
        """
        Compute obstacle quad for tile `idx` using the road polygon to infer
        width and forward directions, then create a static body at that location.
        """
        # Track tuple looks like: (alpha, beta, x, y)
        _, beta, cx, cy = self.track[idx]

        # Try extracting from the precomputed road polygon to align & size nicely
        try:
            p0, p1, p2, p3 = self.road_poly[idx][0]  # [left1, right1, right2, left2]
            width_vec = np.array(p1) - np.array(p0)
            width = float(np.linalg.norm(width_vec))
            if width <= 1e-6:
                return
            width_dir = width_vec / width
            half_width = 0.5 * width

            along_vec = np.array(p3) - np.array(p0)
            along = float(np.linalg.norm(along_vec))
            if along <= 1e-6:
                along_dir = np.array([-width_dir[1], width_dir[0]])  # perpendicular
            else:
                along_dir = along_vec / along
        except Exception:
            # Fallback using steering angle when road_poly indexing fails
            width_dir = np.array([math.cos(beta), math.sin(beta)], dtype=np.float32)
            half_width = getattr(self, "TRACK_WIDTH", _TRACK_WIDTH_DEFAULT)
            along_dir = np.array([-width_dir[1], width_dir[0]], dtype=np.float32)

        # Choose left/right side of lane
        side = rng.choice([-1.0, 1.0])
        center_offset = side * (0.60 * half_width)  # keep inside lane
        center = np.array([cx, cy]) + width_dir * center_offset

        # Rectangle half-extents (relative to road width)
        hw = max(0.5, self.obstacle_width_ratio * half_width)
        hh = max(0.5, self.obstacle_length_ratio * half_width)

        # Local rectangle in [width_dir, along_dir] basis
        local = np.array([
            [-hw, -hh],
            [ hw, -hh],
            [ hw,  hh],
            [-hw,  hh],
        ])

        # Rotate + translate into world frame
        R = np.stack([width_dir, along_dir], axis=1)  # columns are basis vectors
        verts = (local @ R.T) + center                # (4,2)

        # Create static Box2D body
        shape = polygonShape(vertices=[tuple(v) for v in verts])
        fix = fixtureDef(shape=shape, density=0.0, friction=0.5, restitution=0.0)
        body = self.world.CreateStaticBody(fixtures=fix)
        body.userData = body
        body.is_obstacle = True  # tag for collision detection
        self.obstacle_bodies.append(body)

        # Add to drawing list so obstacles are visible in renders
        color = (0.55, 0.27, 0.07)  # brown-ish
        poly = ([(float(v[0]), float(v[1])) for v in verts], color)
        self.obstacle_polys.append(poly)
        self.road_poly.append(poly)


# ======================
# Register environment
# ======================
register(
    id="CarRacingObstacles-v0",
    entry_point=f"{__name__}:CarRacingObstacles",
)


# ======================
# Run with keyboard + logger
# ======================
if __name__ == "__main__":
    import numpy as np
    from time import time

    # Make env; render_mode='rgb_array' is required for gymnasium.utils.play
    env = gym.make(
        "CarRacingObstacles-v0",
        render_mode="rgb_array",
        continuous=True,
        domain_randomize=False,
        lap_complete_percent=0.95,
        # obstacle/gameplay controls:
        end_on_obstacle=True,          # lose immediately on contact
        obstacle_penalty=0.0,          # also apply a score hit if you want (e.g., -10.0)
        obstacle_probability=0.08,
        obstacle_min_gap=12,
        obstacle_width_ratio=0.34,
        obstacle_length_ratio=0.18,
    )

    # WASD mapping for continuous actions [steer, gas, brake]
    keys_to_action = {
        "w":   np.array([0.0, 0.7, 0.0], dtype=np.float32),  # gas
        "a":   np.array([-1.0, 0.0, 0.0], dtype=np.float32), # steer left
        "d":   np.array([1.0,  0.0, 0.0], dtype=np.float32), # steer right
        "s":   np.array([0.0, 0.0, 1.0], dtype=np.float32),  # brake
        "wa":  np.array([-1.0, 0.7, 0.0], dtype=np.float32),
        "wd":  np.array([1.0,  0.7, 0.0], dtype=np.float32),
        "sa":  np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        "sd":  np.array([1.0,  0.0, 1.0], dtype=np.float32),
    }
    noop = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Episode score logger for play()
    class ScoreLogger:
        """
        Callback compatible with gymnasium.utils.play.
        Receives: (obs_t, obs_tp1, action, rew, terminated, truncated, info)
        """
        def __init__(self):
            self.reset()

        def reset(self):
            self.return_, self.steps = 0.0, 0
            self.t0 = time()

        def __call__(self, obs_t, obs_tp1, action, rew, terminated, truncated, info):
            # accumulate reward
            self.return_ += float(rew)
            self.steps += 1

            # episode end -> print score line
            if terminated or truncated:
                reason = info.get("terminated_reason")
                if reason is None:
                    # Could be lap completed, off-track time limit, etc.
                    reason = "terminated" if terminated else "truncated"

                num_collisions = info.get("num_collisions", 0)
                duration = time() - self.t0
                print(
                    f"[GAME OVER] {reason} — Final score: {self.return_:.1f}  | "
                    f"steps: {self.steps}  | collisions: {num_collisions}  | "
                    f"duration: {duration:.1f}s"
                )
                # Reset for next automatic episode
                self.reset()

            # return any value to enable live plotting (not used here)
            return [self.return_]

    logger = ScoreLogger()

    try:
        print(
            "\nLose condition: touching any obstacle ends the run immediately."
            "\nControls: W=gas, S=brake, A/D=steer. Close the window to quit.\n"
        )
        play(env, keys_to_action=keys_to_action, noop=noop, callback=logger)
    finally:
        env.close()
