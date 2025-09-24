import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from model import QNetwork, ReplayBuffer
from env import MazeEnv   

def train(env=None,
          episodes=1000,
          batch_size=64,
          gamma=0.99,
          lr=1e-3,
          epsilon_start=1.0,
          epsilon_end=0.05,
          epsilon_decay=0.995,
          target_update=10):

    # 如果外部沒有傳入 env，就自己建立一個隨機起點的環境
    if env is None:
        env = MazeEnv()  # 隨機起點

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 建立 Q-network 與 target network
    q_net = QNetwork().to(device)
    target_net = QNetwork().to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer()

    epsilon = epsilon_start
    reward_history = []
    epsilon_history = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for _ in range(env.max_steps):
            # ε-greedy 探索策略
            if np.random.rand() < epsilon:
                action = np.random.randint(env.action_space())
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = q_net(state_tensor)
                    action = q_values.argmax().item()

            # 執行動作
            next_state, reward, done, _ = env.step(action)
            buffer.push((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # 訓練 Q-network
            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.BoolTensor(dones).unsqueeze(1).to(device)

                q_values = q_net(states).gather(1, actions)

                # Double DQN target
                with torch.no_grad():
                    next_actions = q_net(next_states).argmax(1, keepdim=True)
                    target_q = target_net(next_states).gather(1, next_actions)
                    expected_q = rewards + gamma * target_q * (~dones)

                loss = F.mse_loss(q_values, expected_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        # ε 衰減
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # 更新 target network
        if episode % target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

        reward_history.append(total_reward)
        epsilon_history.append(epsilon)

    # 儲存模型
    torch.save(q_net.state_dict(), "ddqn_model.pth")

    # 畫圖 Reward vs Epsilon
    episodes_range = range(episodes)
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward", color="tab:blue")
    ax1.plot(episodes_range, reward_history, label="Total Reward", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Epsilon", color="tab:red")
    ax2.plot(episodes_range, epsilon_history, label="Epsilon", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    plt.title("Training Progress: Reward vs Epsilon")
    plt.grid(True)
    plt.show()
