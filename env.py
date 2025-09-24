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

    def reached_goal(self):
        return np.hypot(self.x - self.goal[0], self.y - self.goal[1]) < self.distance_threshold

    # --- Step 與獎勵 ---
    def step(self, action):
        delta_angle = self.actions[action]
        self.angle = (self.angle + delta_angle + 180) % 360 - 180
        rad = np.radians(self.angle)

        # 嘗試移動
        new_x = self.x + self.step_distance * np.cos(rad)
        new_y = self.y + self.step_distance * np.sin(rad)

        # 邊界檢查
        hit_wall = (
            new_x <= 0 or new_x >= self.map_size or
            new_y <= 0 or new_y >= self.map_size
        )
        new_x = np.clip(new_x, 0, self.map_size)
        new_y = np.clip(new_y, 0, self.map_size)

        # 障礙檢查
        hit_obstacle = self.is_in_obstacle(new_x, new_y)

        # 狀態更新（只有沒撞到才移動）
        if not (hit_obstacle or hit_wall):
            self.x, self.y = new_x, new_y

        self.steps += 1

        # --- done 條件 ---
        done = (
            self.steps >= self.max_steps or
            self.reached_goal() or
            hit_obstacle or
            hit_wall
        )

        # --- reward system ---
        if self.reached_goal():
            reward = 10.0
        elif hit_obstacle or hit_wall:
            reward = -1.0
        else:
            reward = -0.1

        return self._get_state(), reward, done, {}


    # --- API ---
    def state_space(self):
        return 3  

    def action_space(self):
        return 5
