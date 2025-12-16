import math
from typing import Tuple, List

import numpy as np
import gymnasium as gym

from gymnasium.envs.box2d.car_racing import CarRacing
from gymnasium.envs.registration import register
from gymnasium.utils.play import play

from Box2D.b2 import polygonShape, fixtureDef, contactListener

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
        info["terminated_reason"] = None
        self._obstacle_collision_happened = False
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        info = dict(info)
        info.setdefault("terminated_reason", None)

        # If we touched an obstacle since the last step, end the episode now.
        if self._obstacle_collision_happened and not terminated and not truncated:
            terminated = True
            info["terminated_reason"] = "obstacle_collision"

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


register(
    id="CarRacingObstacles-v0",
    entry_point=f"{__name__}:CarRacingObstacles",
)

if __name__ == "__main__":
    import numpy as np
    import pygame
    
    env = gym.make(
        "CarRacingObstacles-v0",
        render_mode="rgb_array",
        continuous=True,
        domain_randomize=False,
        lap_complete_percent=0.95,
        # obstacle controls:
        end_on_obstacle=True,       # <-- lose on touch
        obstacle_penalty=0.0,       # set negative if you also want score penalty
        obstacle_probability=0.08,
        obstacle_min_gap=12,
    )

    keys_to_action = {
        pygame.K_UP:   np.array([0.0, 0.7, 0.0], dtype=np.float32),
        pygame.K_LEFT:   np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        pygame.K_RIGHT:   np.array([1.0, 0.0, 0.0], dtype=np.float32),
        pygame.K_DOWN:   np.array([0.0, 0.0, 1.0], dtype=np.float32),
        (pygame.K_UP, pygame.K_LEFT):  np.array([-1.0, 0.7, 0.0], dtype=np.float32),
        (pygame.K_UP, pygame.K_RIGHT):  np.array([1.0,  0.7, 0.0], dtype=np.float32),
        (pygame.K_DOWN, pygame.K_LEFT):  np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        (pygame.K_DOWN, pygame.K_RIGHT):  np.array([1.0,  0.0, 1.0], dtype=np.float32),
    }
    noop = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    try:
        print("\nLose condition: touching any obstacle ends the run immediately."
              "\nControls: UP=gas, DOWN=brake, LEFT/RIGHT=steer. Close the window to quit.")
        play(env, keys_to_action=keys_to_action, noop=noop)
    finally:
        env.close()
