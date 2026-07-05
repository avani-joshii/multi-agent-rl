import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class RaceCar:
    def __init__(self, start_pos, heading, color):
        self.color = color
        self.max_speed = 8.0
        self.acceleration = 0.25
        self.turn_speed = 0.09
        self.friction = 0.03
        self.lap = 0
        self.next_checkpoint = 0
        self.debug_reward = {"survival": 0, "speed": 0, "wall": 0, "checkpoint": 0}
        self.trail = []
        self.reset(start_pos, heading)

    def reset(self, start_pos, heading):
        self.pos = np.array(start_pos, dtype=np.float32)
        self.heading = heading
        self.speed = 0.0
        self.alive = True
        self.next_checkpoint = 0
        self.lap = 0
        self.trail=[]


class MultiRaceEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        self.width = 800
        self.height = 600

        self.track_center = np.array([400, 300], dtype=np.float32)

        self.outer_rx = 260
        self.outer_ry = 220
        self.track_width = 100

        self.inner_rx = self.outer_rx - self.track_width
        self.inner_ry = self.outer_ry - self.track_width

        self.num_checkpoints = 20
        self.checkpoints = []

        track_rx = (self.outer_rx + self.inner_rx) / 2
        track_ry = (self.outer_ry + self.inner_ry) / 2

        start_angle = -np.pi / 2

        for i in range(self.num_checkpoints):
            angle = start_angle - 2 * np.pi * i / self.num_checkpoints

            x = self.track_center[0] + track_rx * np.cos(angle)
            y = self.track_center[1] + track_ry * np.sin(angle)

            self.checkpoints.append(np.array([x, y], dtype=np.float32))

        self.max_steps = 1500
        self.steps = 0

        self.num_rays = 7
        self.ray_length = 150

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(12,),
            dtype=np.float32,
        )

        start_x = self.track_center[0]
        start_y = self.track_center[1] - track_ry + 25

        self.car1 = RaceCar([start_x - 15, start_y], 0, (50, 120, 255))
        self.car2 = RaceCar([start_x + 15, start_y], 0, (255, 80, 80))

        self.screen = None
        self.clock = None

    def _point_on_track(self, point):
        x = point[0] - self.track_center[0]
        y = point[1] - self.track_center[1]

        outer = (x / self.outer_rx) ** 2 + (y / self.outer_ry) ** 2
        inner = (x / self.inner_rx) ** 2 + (y / self.inner_ry) ** 2

        return outer <= 1.0 and inner >= 1.0

    def _track_direction(self, car):
        x = car.pos[0] - self.track_center[0]
        y = car.pos[1] - self.track_center[1]

        rx = (self.outer_rx + self.inner_rx) / 2
        ry = (self.outer_ry + self.inner_ry) / 2

        t = np.arctan2(y / ry, x / rx)

        dx = rx * np.sin(t)
        dy = -ry * np.cos(t)

        return np.arctan2(dy, dx)

    def _cast_rays(self, car):
        distances = []

        ray_angles = np.linspace(-np.pi / 2, np.pi / 2, self.num_rays)

        for angle in ray_angles:
            direction = np.array([
                np.cos(car.heading + angle),
                np.sin(car.heading + angle)
            ])

            distance = self.ray_length

            for d in range(0, self.ray_length, 4):
                p = car.pos + direction * d
                if not self._point_on_track(p):
                    distance = d
                    break

            distances.append(distance / self.ray_length)

        return np.array(distances, dtype=np.float32)

    def _get_obs(self, car, opponent):
        rays = self._cast_rays(car)

        speed = np.array([car.speed / car.max_speed], dtype=np.float32)

        relative = (opponent.pos - car.pos) / 300.0

        checkpoint = self.checkpoints[car.next_checkpoint]
        checkpoint_relative = (checkpoint - car.pos) / 300.0

        obs = np.concatenate([rays, speed, relative, checkpoint_relative])

        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0

        track_rx = (self.outer_rx + self.inner_rx) / 2
        track_ry = (self.outer_ry + self.inner_ry) / 2

        start_x = self.track_center[0]
        start_y = self.track_center[1] - track_ry + 25

        self.car1.reset([start_x - 15, start_y], np.pi)
        self.car2.reset([start_x + 15, start_y], np.pi)

        obs1 = self._get_obs(self.car1, self.car2)
        obs2 = self._get_obs(self.car2, self.car1)

        return (obs1, obs2), {}

    def _apply_action(self, car, action):
        if not car.alive:
            return

        car.speed *= (1 - car.friction)

        if action == 0:
            car.speed += car.acceleration
        elif action == 1:
            car.speed += car.acceleration * 0.5
            car.heading -= car.turn_speed
        elif action == 2:
            car.speed += car.acceleration * 0.5
            car.heading += car.turn_speed
        elif action == 3:
            car.speed -= 0.3

        car.speed = np.clip(car.speed, 0, car.max_speed)

        direction = np.array([np.cos(car.heading), np.sin(car.heading)])
        car.pos += direction * car.speed
        car.trail.append(car.pos.copy())
        if len(car.trail) > 20:
            car.trail.pop(0)


    def _reward(self, car):
        if not self._point_on_track(car.pos):
            car.alive = False
            car.speed = 0
            car.pos -= np.array([np.cos(car.heading), np.sin(car.heading)]) * 12
            car.debug_reward = {"survival": 0, "speed": 0, "wall": 0, "checkpoint": -15}
            return -15

        survival = 0.1 if car.speed > 0.5 else -0.1

        target_heading = self._track_direction(car)
        heading_diff = np.abs((car.heading - target_heading + np.pi) % (2 * np.pi) - np.pi)
        alignment = np.cos(heading_diff)

        speed_reward = car.speed * 0.15 * alignment

        rays = self._cast_rays(car)
        closest_wall = np.min(rays)
        wall_penalty = 0
        if closest_wall < 0.3:
            wall_penalty = -(0.3 - closest_wall) * 2.0

        checkpoint_reward = 0
        checkpoint = self.checkpoints[car.next_checkpoint]
        distance = np.linalg.norm(car.pos - checkpoint)

        if distance < 40:
            checkpoint_reward = 25
            car.next_checkpoint += 1
            if car.next_checkpoint >= self.num_checkpoints:
                car.next_checkpoint = 0
                car.lap += 1
                checkpoint_reward += 200

        reward = survival + speed_reward + wall_penalty + checkpoint_reward

        car.debug_reward = {
            "survival": survival,
            "speed": speed_reward,
            "wall": wall_penalty,
            "checkpoint": checkpoint_reward,
        }

        return reward

    def step(self, actions):
        a1, a2 = actions

        self._apply_action(self.car1, a1)
        self._apply_action(self.car2, a2)

        r1 = self._reward(self.car1)
        r2 = self._reward(self.car2)

        self.steps += 1

        both_crashed = (not self.car1.alive) and (not self.car2.alive)

        done = self.steps >= self.max_steps or both_crashed

        obs1 = self._get_obs(self.car1, self.car2)
        obs2 = self._get_obs(self.car2, self.car1)

        if self.render_mode == "human":
            self._render_frame()

        return (obs1, obs2), (r1, r2), done, False, {}

    def _render_frame(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("RL Racing")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self.screen.fill((40, 120, 40))

        pygame.draw.ellipse(
            self.screen,
            (70, 70, 70),
            pygame.Rect(
                int(self.track_center[0] - self.outer_rx),
                int(self.track_center[1] - self.outer_ry),
                int(self.outer_rx * 2),
                int(self.outer_ry * 2),
            ),
        )

        pygame.draw.ellipse(
            self.screen,
            (40, 120, 40),
            pygame.Rect(
                int(self.track_center[0] - self.inner_rx),
                int(self.track_center[1] - self.inner_ry),
                int(self.inner_rx * 2),
                int(self.inner_ry * 2),
            ),
        )

        for checkpoint in self.checkpoints:
            pygame.draw.circle(
                self.screen,
                (0, 255, 0),
                (int(checkpoint[0]), int(checkpoint[1])),
                6,
            )

        self._draw_car(self.car1)
        self._draw_car(self.car2)

        font = pygame.font.SysFont(None, 24)

        t = font.render(f"Steps: {self.steps}", True, (255, 255, 255))
        blue = font.render(
            f"Blue Lap: {self.car1.lap}  CP: {self.car1.next_checkpoint}/{self.num_checkpoints}",
            True, (50, 120, 255),
        )
        red = font.render(
            f"Red Lap: {self.car2.lap}  CP: {self.car2.next_checkpoint}/{self.num_checkpoints}",
            True, (255, 80, 80),
        )

        self.screen.blit(blue, (10, 40))
        self.screen.blit(red, (10, 70))
        self.screen.blit(t, (10, 10))

        def draw_breakdown(car, y_start, color):
            d = car.debug_reward
            speed_txt = font.render(f"speed: {car.speed:.2f}", True, color)
            self.screen.blit(speed_txt, (self.width - 380, y_start))

            reward_txt = font.render(
                f"surv:{d['survival']:.2f} spd:{d['speed']:.2f} wall:{d['wall']:.2f} cp:{d['checkpoint']:.2f}",
                True, color
            )
            self.screen.blit(reward_txt, (self.width - 380, y_start + 30))

        draw_breakdown(self.car1, 20, (50, 120, 255))
        draw_breakdown(self.car2, 100, (255, 80, 80))

        pygame.display.flip()
        self.clock.tick(60)

    def _draw_car(self, car):
        color = car.color if car.alive else (120, 120, 120)

        x = car.pos[0]
        y = car.pos[1]
        h = car.heading
        
        for i, p in enumerate(car.trail):
            fade = int(255 * (i + 1) / max(len(car.trail), 1))
            radius = max(1, int(4 * (i + 1) / max(len(car.trail), 1)))
            s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, fade), (radius, radius), radius)
            self.screen.blit(s, (int(p[0]) - radius, int(p[1]) - radius))

        front = (x + np.cos(h) * 12, y + np.sin(h) * 12)

        front = (x + np.cos(h) * 12, y + np.sin(h) * 12)
        left = (x + np.cos(h + 2.4) * 8, y + np.sin(h + 2.4) * 8)
        right = (x + np.cos(h - 2.4) * 8, y + np.sin(h - 2.4) * 8)

        pygame.draw.polygon(self.screen, color, [front, left, right])

        rays = self._cast_rays(car)
        angles = np.linspace(-np.pi / 2, np.pi / 2, self.num_rays)

        for d, a in zip(rays, angles):
            end = car.pos + np.array([np.cos(h + a), np.sin(h + a)]) * d * self.ray_length
            pygame.draw.line(self.screen, color, car.pos.astype(int), end.astype(int), 1)

    def render(self):
        self._render_frame()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None