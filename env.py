import numpy as np

class MazeEnv:
    def __init__(self, map_size=500):
        self.map_size = map_size
        self.goal = (450, 450)
        self.distance_threshold = 15
        self.step_distance = 10

        # 障礙物清單 (xmin, xmax, ymin, ymax)
        self.obstacles = [
            (100, 150, 100, 400),
            (300, 400, 300, 320),
            (200, 250, 50, 100)
        ]

        # 五個動作 (角度改變)
        self.actions = [-60, -30, 0, +30, +60]
        self.max_steps = 200
        self.reset()

    # --- 基本功能 ---
    def _random_start(self):
        while True:
            x = np.random.uniform(0, self.map_size)
            y = np.random.uniform(0, self.map_size)
            if not self.is_in_obstacle(x, y):
                return (x, y)

    def reset(self):
        self.x, self.y = self._random_start()
        self.angle = np.random.uniform(-180, 180)
        self.steps = 0
        return self._get_state()
    
    def test_reset(self):
        self.x, self.y = (50, 50)
        self.angle = 0
        self.steps = 0
        return self._get_state()

    def _get_beta(self):
        dx = self.goal[0] - self.x
        dy = self.goal[1] - self.y
        goal_angle = np.degrees(np.arctan2(dy, dx))
        beta = goal_angle - self.angle
        beta = (beta + 180) % 360 - 180
        return beta / 180  # normalize [-1, 1]

    def _get_state(self):
    # 只回傳 x, y, beta，不包含 danger
        return np.array([
            self.x / self.map_size,
            self.y / self.map_size,
            self._get_beta()
        ], dtype=np.float32)

    # --- 區域判斷 ---
    def is_in_obstacle(self, x, y):
        for (xmin, xmax, ymin, ymax) in self.obstacles:
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return True
        return False

    def is_in_danger_zone(self, x, y, margin=10):
        # 靠近牆壁
        if x < margin or x > self.map_size - margin:
            return True
        if y < margin or y > self.map_size - margin:
            return True
        # 靠近障礙物
        for (xmin, xmax, ymin, ymax) in self.obstacles:
            dx = max(xmin - x, 0, x - xmax)
            dy = max(ymin - y, 0, y - ymax)
            if np.hypot(dx, dy) < margin:
                return True
        return False

    def dist_to_obstacles(self, x, y):
        dists = []
        for (xmin, xmax, ymin, ymax) in self.obstacles:
            dx = max(xmin - x, 0, x - xmax)
            dy = max(ymin - y, 0, y - ymax)
            dists.append(np.hypot(dx, dy))
        return min(dists) if dists else float("inf")

    def reached_goal(self):
        return np.hypot(self.x - self.goal[0], self.y - self.goal[1]) < self.distance_threshold

    # --- Step 與獎勵 ---
    def step(self, action):
        delta_angle = self.actions[action]
        self.angle = (self.angle + delta_angle + 180) % 360 - 180
        rad = np.radians(self.angle)

        prev_x, prev_y = self.x, self.y
        prev_danger = self.is_in_danger_zone(prev_x, prev_y)
        prev_dist_obs = self.dist_to_obstacles(prev_x, prev_y)

        # 移動
        new_x = self.x + self.step_distance * np.cos(rad)
        new_y = self.y + self.step_distance * np.sin(rad)
        new_x = np.clip(new_x, 0, self.map_size)
        new_y = np.clip(new_y, 0, self.map_size)

        hit_obstacle = self.is_in_obstacle(new_x, new_y)
        if not hit_obstacle:
            self.x, self.y = new_x, new_y

        self.steps += 1
        done = self.steps >= self.max_steps or self.reached_goal()

        # --- reward system ---
        if self.reached_goal():
            reward = 100.0
        elif hit_obstacle:
            reward = -50.0
        else:
            new_danger = self.is_in_danger_zone(self.x, self.y)
            new_dist_obs = self.dist_to_obstacles(self.x, self.y)

            moving_away = new_dist_obs > prev_dist_obs
            moving_closer = new_dist_obs < prev_dist_obs

            if prev_danger == 0 and new_danger == 1:      # S -> U
                reward = -2.0
            elif prev_danger == 1 and new_danger == 1:    # U -> U
                if moving_closer:  # CO
                    reward = -2.0
                else:              # AO
                    reward = -1.0
            elif prev_danger == 0 and new_danger == 0:    # S -> S
                if moving_closer:  # CG
                    reward = +1.0
                else:              # AG
                    reward = 0.0
            elif prev_danger == 1 and new_danger == 0:    # U -> S
                reward = +2.0
            else:
                reward = -0.01  # fallback

        return self._get_state(), reward, done, {}

    # --- API ---
    def state_space(self):
        return 4  # (x, y, beta, danger)

    def action_space(self):
        return 5
