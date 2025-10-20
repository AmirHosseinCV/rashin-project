import math
from typing import Tuple, List

import numpy as np
import gymnasium as gym

from gymnasium.envs.box2d.car_racing import CarRacing
from gymnasium.envs.registration import register
from gymnasium.utils.play import play

# Box2D
from Box2D.b2 import polygonShape, fixtureDef, contactListener


# --- a small helper to get module-level constants if present across versions
try:
    # In Gymnasium's car_racing module these names are typically present
    from gymnasium.envs.box2d import car_racing as car_mod
    _TRACK_WIDTH_DEFAULT = float(getattr(car_mod, "TRACK_WIDTH", 6.0))
except Exception:
    _TRACK_WIDTH_DEFAULT = 6.0


class _ObstacleCollisionListener(contactListener):
    """
    A minimal contact listener that forwards to the original listener (which
    handles tile friction / progress logic) and also detects collisions with
    our obstacle bodies to optionally apply a penalty and keep a collision count.
    """
    def __init__(self, env, base_listener=None, penalty: float = 0.0):
        super().__init__()
        self.env = env
        self.base = base_listener
        self.penalty = penalty

    # Forward compatibility with Box2D listener interface used by CarRacing
    def BeginContact(self, contact):
        # Keep original tile/friction logic
        if self.base is not None and hasattr(self.base, "BeginContact"):
            self.base.BeginContact(contact)

        u1 = getattr(contact.fixtureA.body, "userData", None)
        u2 = getattr(contact.fixtureB.body, "userData", None)

        # We flag obstacles by setting body.is_obstacle = True
        if (u1 is not None and getattr(u1, "is_obstacle", False)) or (
            u2 is not None and getattr(u2, "is_obstacle", False)
        ):
            self.env.num_collisions += 1
            # Apply flat penalty once per contact start if configured
            if self.penalty:
                # CarRacing internally accumulates reward in env.reward
                # so we follow the same convention.
                self.env.reward += self.penalty

    def EndContact(self, contact):
        if self.base is not None and hasattr(self.base, "EndContact"):
            self.base.EndContact(contact)


class CarRacingObstacles(CarRacing):
    """
    CarRacing with static Box2D obstacles added on selected road tiles.
    Obstacles are visible (drawn onto the background polys) and physically collide
    with the car. Optional per-collision penalty can be applied.
    """
    # keep parent metadata (render_fps, supported render_modes, etc.)
    metadata = CarRacing.metadata

    def __init__(
        self,
        *,
        obstacle_probability: float = 0.05,     # chance to place an obstacle on an eligible tile
        obstacle_min_gap: int = 15,             # min tile gap between successive obstacles
        obstacle_width_ratio: float = 0.30,     # obstacle half-width as fraction of half road width
        obstacle_length_ratio: float = 0.15,    # obstacle half-length as fraction of half road width
        obstacle_penalty: float = 0.0,          # add to env.reward on obstacle contact (e.g., -5.0)
        **kwargs,
    ):
        """
        kwargs are forwarded to CarRacing, e.g.
        - continuous=True/False
        - domain_randomize=True/False
        - lap_complete_percent=0.95
        - render_mode=("rgb_array" for play, or "human" for live window)
        """
        super().__init__(**kwargs)

        # Config
        self.obstacle_probability = float(obstacle_probability)
        self.obstacle_min_gap = int(obstacle_min_gap)
        self.obstacle_width_ratio = float(obstacle_width_ratio)
        self.obstacle_length_ratio = float(obstacle_length_ratio)
        self.obstacle_penalty = float(obstacle_penalty)

        # State
        self.obstacle_bodies = []
        self.obstacle_polys: List[Tuple[List[Tuple[float, float]], Tuple[float, float, float]]] = []
        self.num_collisions = 0

    # --------- lifecycle ----------
    def reset(self, *, seed=None, options=None):
        # Reset base env to generate track & car, world, etc. (Gymnasium API)
        obs, info = super().reset(seed=seed, options=options)

        # Remove any prior obstacles from a previous episode
        self._clear_obstacles()

        # Wrap original contact listener so we can detect obstacle contacts
        base_listener = getattr(self.world, "contactListener", None)
        self._collision_listener = _ObstacleCollisionListener(
            env=self, base_listener=base_listener, penalty=self.obstacle_penalty
        )
        # Install our composite listener
        self.world.contactListener = self._collision_listener

        # Spawn new obstacles for this track
        self._spawn_obstacles()

        # Add a little telemetry to info
        info = dict(info)
        info["num_obstacles"] = len(self.obstacle_bodies)
        info["num_collisions"] = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # expose obstacle stats in info
        info = dict(info)
        info["num_obstacles"] = len(self.obstacle_bodies)
        info["num_collisions"] = self.num_collisions
        return obs, reward, terminated, truncated, info

    def close(self):
        self._clear_obstacles()
        return super().close()

    # --------- obstacle logic ----------
    def _clear_obstacles(self):
        if getattr(self, "world", None) is not None:
            for b in self.obstacle_bodies:
                try:
                    self.world.DestroyBody(b)
                except Exception:
                    pass
        self.obstacle_bodies.clear()
        # We append obstacle polys to road_poly for drawing; remove only our additions
        if hasattr(self, "obstacle_polys") and hasattr(self, "road_poly"):
            for poly in self.obstacle_polys:
                try:
                    self.road_poly.remove(poly)
                except ValueError:
                    pass
        self.obstacle_polys.clear()
        self.num_collisions = 0

    def _spawn_obstacles(self):
        """
        Choose tiles along the track and place oriented static rectangles offset to
        the left/right of the track centerline, with sizes relative to road width.
        Also append their quads to self.road_poly so they are visible in rgb_array renders.
        """
        if not hasattr(self, "track") or not hasattr(self, "road_poly") or not self.track:
            return

        rng = getattr(self, "np_random", np.random.default_rng())
        gap = max(1, int(self.obstacle_min_gap))
        last_idx = -gap

        # Avoid spawn very near start/finish to keep the spawn safe to drive into
        for idx in range(5, len(self.track) - 5):
            if idx - last_idx < gap:
                continue
            if rng.random() >= self.obstacle_probability:
                continue

            self._create_obstacle_for_tile(idx, rng)
            last_idx = idx

    def _create_obstacle_for_tile(self, idx: int, rng):
        """
        Compute obstacle quad for tile `idx` using the stored road polygon to infer
        the width direction, then create a Box2D static body at that location.
        """
        # Track tuple: (alpha, beta, x, y) in the reference implementation
        _, beta, cx, cy = self.track[idx]

        # Road polygon vertices for this tile: [road1_l, road1_r, road2_r, road2_l]
        # From these we infer the cross-road direction and width.
        try:
            p0, p1, p2, p3 = self.road_poly[idx][0]
            width_vec = np.array(p1) - np.array(p0)                # across the road
            width = float(np.linalg.norm(width_vec))
            if width <= 1e-6:
                return
            width_dir = width_vec / width                          # unit vector across width
            half_width = 0.5 * width
            along_vec = np.array(p3) - np.array(p0)                # along the road
            along = float(np.linalg.norm(along_vec))
            if along <= 1e-6:
                along_dir = np.array([-width_dir[1], width_dir[0]])  # fallback perp
            else:
                along_dir = along_vec / along
        except Exception:
            # Fallback based on steering angle
            width_dir = np.array([math.cos(beta), math.sin(beta)], dtype=np.float32)
            half_width = getattr(self, "TRACK_WIDTH", _TRACK_WIDTH_DEFAULT)
            along_dir = np.array([-width_dir[1], width_dir[0]], dtype=np.float32)

        # Shift to left/right lane position
        side = rng.choice([-1.0, 1.0])      # left or right
        center_offset = side * (0.60 * half_width)  # keep inside the lane
        center = np.array([cx, cy]) + width_dir * center_offset

        # Rectangle half-extents (relative to half road width)
        hw = max(0.5, self.obstacle_width_ratio * half_width)
        hh = max(0.5, self.obstacle_length_ratio * half_width)

        # Local rectangle (in [width_dir, along_dir] frame)
        local = np.array([
            [-hw, -hh],
            [ hw, -hh],
            [ hw,  hh],
            [-hw,  hh],
        ])

        # Rotate into world frame
        R = np.stack([width_dir, along_dir], axis=1)  # 2x2: columns are basis vectors
        verts = (local @ R.T) + center               # 4x2

        # Create static Box2D body with polygon fixture for collision
        shape = polygonShape(vertices=[tuple(v) for v in verts])
        fix = fixtureDef(shape=shape, density=0.0, friction=0.5, restitution=0.0)
        body = self.world.CreateStaticBody(fixtures=fix)
        body.userData = body
        body.is_obstacle = True           # flag checked by our contact listener

        self.obstacle_bodies.append(body)

        # Add to drawing list so obstacles are visible in rgb_array renders
        color = (0.55, 0.27, 0.07)        # brown-ish
        poly = ([(float(v[0]), float(v[1])) for v in verts], color)
        self.obstacle_polys.append(poly)
        self.road_poly.append(poly)


# --- Register the env so you can gym.make("CarRacingObstacles-v0")
register(
    id="CarRacingObstacles-v0",
    entry_point=f"{__name__}:CarRacingObstacles",
)


if __name__ == "__main__":
    """
    Launch a human-play session using WASD.
    IMPORTANT: For keyboard play, pass render_mode='rgb_array' so gymnasium.utils.play
    can display the frames (per Gymnasium utils docs).
    """
    import numpy as np

    # Configure your env args here; continuous=True for analog steering/gas/brake
    env = gym.make(
        "CarRacingObstacles-v0",
        render_mode="rgb_array",    # required for gymnasium.utils.play
        continuous=True,
        domain_randomize=False,
        lap_complete_percent=0.95,
        # obstacle-specific defaults can be overridden via make_kwargs too
        # e.g., obstacle_probability=0.08, obstacle_penalty=-5.0
    )

    # WASD mapping for continuous actions [steer, gas, brake] per CarRacing docs.
    # (You can hold keys together; e.g., 'W' + 'A' to gas + steer left.)
    keys_to_action = {
        "w":   np.array([0.0, 0.7, 0.0], dtype=np.float32),  # gas
        "a":   np.array([-1.0, 0.0, 0.0], dtype=np.float32), # steer left
        "d":   np.array([1.0, 0.0, 0.0], dtype=np.float32),  # steer right
        "s":   np.array([0.0, 0.0, 1.0], dtype=np.float32),  # brake
        "wa":  np.array([-1.0, 0.7, 0.0], dtype=np.float32),
        "wd":  np.array([1.0,  0.7, 0.0], dtype=np.float32),
        "sa":  np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        "sd":  np.array([1.0,  0.0, 1.0], dtype=np.float32),
    }
    # When no known combo is pressed, keep a neutral action
    noop = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    try:
        print("\nControls: W=gas, S=brake, A/D=steer, combos like WA or WD.\n"
              "Close the window to quit.")
        play(env, keys_to_action=keys_to_action, noop=noop)
    finally:
        env.close()
