import torch
import gym
from QLearning import QNet


def main():
    Q = torch.load("project26_MountainCar/QLearning_model.pth")
    env = gym.make("MountainCar-v0", render_mode="human")
    s = env.reset()[0]
    done = False
    score = 0.0

    while not done:
        a = Q.select_act(s, 0.0)
        s, r, done, _, _ = env.step(a * 2)
        score += r

    print("score:", score)


if __name__ == "__main__":
    main()
