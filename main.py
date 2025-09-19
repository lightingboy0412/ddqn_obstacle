from env import MazeEnv
from train import train
from test import test

if __name__ == "__main__":
    env = MazeEnv()
    mode = input("Enter mode (train/test): ").strip().lower()

    if mode == "train":
        train(env)
    elif mode == "test":
        test(env)
    else:
        print("Invalid mode. Please enter 'train' or 'test'")
