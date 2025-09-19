import torch
import matplotlib.pyplot as plt
from model import QNetwork
from env import MazeEnv

def test(env=None, model_path="ddqn_model.pth", fixed_start=True):
    # 如果沒有傳入 env，就建立一個環境
    if env is None:
        env = MazeEnv()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # reset 環境
    if fixed_start:
        state = env.test_reset()   # 固定起點 (50,50)
    else:
        state = env.reset()        # 隨機起點

    path = [(env.x, env.y)]

    for step in range(env.max_steps):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = model(s).argmax().item()

        state, _, done, _ = env.step(action)
        path.append((env.x, env.y))

        if done:
            break

    # 拆出路徑座標
    x_coords, y_coords = zip(*path)

    # 建立圖
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x_coords, y_coords, marker='o', markersize=3, linewidth=1, label="Path")
    ax.plot(x_coords[0], y_coords[0], 'go', label='Start')
    ax.plot(x_coords[-1], y_coords[-1], 'rx', label='End')

    # 畫障礙物
    for (xmin, xmax, ymin, ymax) in env.obstacles:
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color='gray', alpha=0.5)
        ax.add_patch(rect)

    # 畫目標
    ax.plot(env.goal[0], env.goal[1], 'ro', label="Goal")

    # 設定圖屬性
    ax.set_xlim(0, env.map_size)
    ax.set_ylim(0, env.map_size)
    ax.set_aspect('equal')
    ax.set_title("Robot Path")
    ax.legend()
    ax.grid(True)

    # 顯示圖
    plt.show()
