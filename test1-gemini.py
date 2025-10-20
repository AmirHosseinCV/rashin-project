import numpy as np
import gymnasium as gym
from gymnasium.envs.box2d.car_racing import CarRacing, FrictionDetector
from gymnasium.utils import EzPickle
import Box2D
from Box2D.b2 import fixtureDef, polygonShape, contactListener
import pygame

# --- Combined Contact Listener (Replaces ObstacleContactListener) ---
class CustomContactListener(FrictionDetector):
    """
    A contact listener that combines the original friction detection
    with our custom obstacle collision detection.
    """
    def __init__(self, env):
        # Initialize the parent FrictionDetector.
        FrictionDetector.__init__(self, env, env.lap_complete_percent)

    def BeginContact(self, contact):
        # Get userData first
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData

        # Check if either userData is a string (our custom obstacles/car)
        if isinstance(u1, str) or isinstance(u2, str):
            # Handle our obstacle detection logic
            if isinstance(u1, str) and isinstance(u2, str):
                if ("car" in u1 and "obstacle" in u2) or \
                   ("obstacle" in u1 and "car" in u2):
                    self.env.collided_with_obstacle = True
            # Don't call parent's method for string userData
            return
        
        # Only call parent's method for non-string userData (original game objects)
        super().BeginContact(contact)
    
    def EndContact(self, contact):
        # Get userData first
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData

        # If either userData is a string, skip parent's method
        if isinstance(u1, str) or isinstance(u2, str):
            return
        
        # Only call parent's method for non-string userData
        super().EndContact(contact)

# --- Custom Environment ---
class CarRacingWithObstacles(CarRacing, EzPickle):
    def __init__(self, num_obstacles=15, **kwargs):
        super().__init__(**kwargs)
        EzPickle.__init__(self, num_obstacles=num_obstacles, **kwargs)

        self.num_obstacles = num_obstacles
        self.collided_with_obstacle = False
        self.obstacles = []
        
        # Replace the default listener with our new, combined one.
        self.world.contactListener = CustomContactListener(self)

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        # Re-set the contact listener after parent reset (parent may recreate the world)
        self.world.contactListener = CustomContactListener(self)
        self.car.hull.userData = "car"
        self.collided_with_obstacle = False
        for obs in self.obstacles:
            self.world.DestroyBody(obs)
        self.obstacles = []
        self._create_obstacles()
        return observation, info

    def _create_obstacles(self):
        track_indices = np.arange(10, len(self.track))
        if len(track_indices) < self.num_obstacles:
            return

        obstacle_indices = np.random.choice(
            track_indices, size=self.num_obstacles, replace=False
        )
        
        for i in obstacle_indices:
            tile_data = self.track[i][0]
            if not isinstance(tile_data, (list, tuple)):
                continue
            
            valid_vertices = [v for v in tile_data if isinstance(v, (list, tuple)) and len(v) >= 2]
            if len(valid_vertices) < 2:
                continue

            center_x = np.mean([v[0] for v in valid_vertices])
            center_y = np.mean([v[1] for v in valid_vertices])
            
            p1 = valid_vertices[0]
            p2 = valid_vertices[1]
            angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
            
            obstacle_body = self.world.CreateStaticBody(
                position=(center_x, center_y), angle=angle, userData="obstacle"
            )
            obstacle_shape = polygonShape(box=(1.5, 0.5))
            obstacle_body.CreateFixture(
                fixtureDef(shape=obstacle_shape, density=1.0, friction=0.3)
            )
            self.obstacles.append(obstacle_body)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        if self.collided_with_obstacle:
            reward -= 100.0
            terminated = True
            self.collided_with_obstacle = False
        return observation, reward, terminated, truncated, info

# --- Main Execution Block ---
if __name__ == "__main__":
    gym.register(
        id='CarRacingWithObstacles-v0',
        entry_point='__main__:CarRacingWithObstacles',
        max_episode_steps=1000,
        reward_threshold=900,
    )

    env = gym.make('CarRacingWithObstacles-v0', render_mode='human', num_obstacles=20)
    
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = np.array([0, 0.5, 0]) 
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"Episode finished. Total Reward: {total_reward}")
    env.close()