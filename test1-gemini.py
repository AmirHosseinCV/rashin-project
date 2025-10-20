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
        # Call parent render first to create the surface
        result = super().render()
        
        # Draw obstacles AFTER calling super().render()
        if self.render_mode == 'human' and self.screen is not None and self.car is not None:
            import pygame.draw
            
            # CarRacing constants - matching the source code exactly
            WINDOW_W = 1000
            WINDOW_H = 800
            SCALE = 6.0  # World units to pixels
            BASE_ZOOM = 2.7
            # Use the environment's dynamic zoom if available to match camera behavior
            CURRENT_ZOOM = getattr(self, 'zoom', BASE_ZOOM)

            # Camera rotation (CarRacing rotates the view by the negative car angle)
            cam_angle = -float(self.car.hull.angle)
            cos_c = np.cos(cam_angle)
            sin_c = np.sin(cam_angle)
            car_pos = self.car.hull.position
            
            for obstacle in self.obstacles:
                pos = obstacle.position
                angle = obstacle.angle
                
                # Get the box shape vertices
                fixture = obstacle.fixtures[0]
                shape = fixture.shape
                
                # Transform each vertex to screen space
                vertices = []
                for v in shape.vertices:
                    # Transform vertex to world coordinates (apply obstacle's rotation)
                    cos_a = np.cos(angle)
                    sin_a = np.sin(angle)
                    world_x = pos[0] + cos_a * v[0] - sin_a * v[1]
                    world_y = pos[1] + sin_a * v[0] + cos_a * v[1]
                    
                    # Translate relative to car
                    dx = world_x - car_pos[0]
                    dy = world_y - car_pos[1]

                    # Apply camera rotation to match base env rendering (keeps obstacles visually static)
                    rx = cos_c * dx - sin_c * dy
                    ry = sin_c * dx + cos_c * dy
                    
                    # Transform to screen coordinates
                    screen_x = WINDOW_W / 2 + rx * SCALE * CURRENT_ZOOM
                    screen_y = WINDOW_H / 4 - ry * SCALE * CURRENT_ZOOM  # Minus because screen Y goes down
                    
                    vertices.append((int(screen_x), int(screen_y)))
                
                # Only draw if at least one vertex is on screen
                on_screen = any(0 <= x <= WINDOW_W and 0 <= y <= WINDOW_H for x, y in vertices)
                if len(vertices) >= 3 and on_screen:
                    pygame.draw.polygon(self.screen, (255, 0, 0), vertices, 0)  # Red fill
                    pygame.draw.polygon(self.screen, (255, 255, 0), vertices, 5)  # Thick yellow border
            
            # Update the display
            pygame.display.flip()
        
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