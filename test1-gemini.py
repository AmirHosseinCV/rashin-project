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
            print(f"Not enough track tiles: {len(track_indices)}")
            return

        obstacle_indices = np.random.choice(
            track_indices, size=self.num_obstacles, replace=False
        )
        
        print(f"Creating {self.num_obstacles} obstacles on track with {len(self.track)} tiles")
        
        for idx, i in enumerate(obstacle_indices):
            # Get the track tile - it's a tuple of (alpha, beta, x, y)
            # where x, y are the world coordinates
            if i >= len(self.track):
                continue
                
            tile = self.track[i]
            
            # Track tiles have (alpha, beta, x, y) where x, y are world coordinates
            if len(tile) < 4:
                continue
            
            # Extract position from track tile
            x = tile[2]
            y = tile[3]
            
            # Get angle from track direction
            # Look at next tile if available to determine direction
            if i + 1 < len(self.track):
                next_tile = self.track[i + 1]
                dx = next_tile[2] - x
                dy = next_tile[3] - y
                angle = np.arctan2(dy, dx)
            else:
                angle = 0
            
            # Create the obstacle at the track position
            obstacle_body = self.world.CreateStaticBody(
                position=(x, y), 
                angle=angle, 
                userData="obstacle"
            )
            obstacle_shape = polygonShape(box=(3.0, 2.0))  # Made larger and more visible
            obstacle_body.CreateFixture(
                fixtureDef(
                    shape=obstacle_shape, 
                    density=1.0, 
                    friction=0.3,
                    categoryBits=0x0010,  # Custom category
                    maskBits=0xFFFF       # Collides with everything
                )
            )
            self.obstacles.append(obstacle_body)
            print(f"  Obstacle {idx+1}: position=({x:.2f}, {y:.2f}), angle={angle:.2f}")
        
        print(f"Successfully created {len(self.obstacles)} obstacles")

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        if self.collided_with_obstacle:
            reward -= 100.0
            terminated = True
            self.collided_with_obstacle = False
        return observation, reward, terminated, truncated, info

    def render(self):
        result = super().render()
        
        # Draw obstacles if in human render mode
        if self.render_mode == 'human' and hasattr(self, 'surf') and self.surf is not None and hasattr(self, 'car'):
            import pygame.draw
            
            # Get screen dimensions from the surface
            WINDOW_W = self.surf.get_width()
            WINDOW_H = self.surf.get_height()
            
            # Calculate zoom factor - standard value from CarRacing
            ZOOM = 2.7
            PLAYFIELD = 2000 / ZOOM
            
            for obstacle in self.obstacles:
                # Get obstacle position and angle
                pos = obstacle.position
                angle = obstacle.angle
                
                # Get the fixture shape (box)
                fixture = obstacle.fixtures[0]
                shape = fixture.shape
                
                # Transform obstacle vertices to screen coordinates
                vertices = []
                for vertex in shape.vertices:
                    # Rotate and translate vertex
                    x = vertex[0] * np.cos(angle) - vertex[1] * np.sin(angle) + pos[0]
                    y = vertex[0] * np.sin(angle) + vertex[1] * np.cos(angle) + pos[1]
                    
                    # Transform to screen coordinates using the same logic as CarRacing
                    # Center the view on the car
                    trans_x = x - self.car.hull.position[0]
                    trans_y = y - self.car.hull.position[1]
                    
                    # Convert to screen coordinates
                    screen_x = WINDOW_W / 2 + trans_x * (WINDOW_W / PLAYFIELD)
                    screen_y = WINDOW_H / 4 - trans_y * (WINDOW_W / PLAYFIELD)
                    
                    vertices.append((screen_x, screen_y))
                
                # Draw the obstacle as a bright red polygon with yellow border
                if len(vertices) >= 3:
                    pygame.draw.polygon(self.surf, (255, 0, 0), vertices)  # Red fill
                    pygame.draw.polygon(self.surf, (255, 255, 0), vertices, 3)  # Yellow border
        
        return result

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
        # Keyboard controls: [steering, gas, brake]
        # steering: -1 (left) to +1 (right)
        # gas: 0 to 1
        # brake: 0 to 1
        steering = 0.0
        gas = 0.0
        brake = 0.0
        
        # Get keyboard input
        keys = pygame.key.get_pressed()
        
        # Arrow keys or WASD for control
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            steering = -1.0
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            steering = 1.0
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            gas = 1.0
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            brake = 0.8
        
        action = np.array([steering, gas, brake])
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        # Handle pygame events to keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True

    print(f"Episode finished. Total Reward: {total_reward}")
    env.close()