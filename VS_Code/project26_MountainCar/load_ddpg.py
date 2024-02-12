import torch
import gym
from DDPG import DDPG


def main():
    ddpg = torch.load("project26_MountainCar/DDPG2_model.pth")
    env = gym.make("MountainCarContinuous-v0", render_mode="human")
    s = env.reset()[0]
    s = torch.from_numpy(s)
    done = False
    score = 0.0

    while not done:
        a = ddpg.select_act(s).item()
        s, r, done, _, _ = env.step([a])
        s = torch.from_numpy(s)
        #r = r if done else -0.1  # 기존 보상체계처럼 변환
        score += r

    print(f"score: {score:.2f}")


if __name__ == "__main__":
    main()
