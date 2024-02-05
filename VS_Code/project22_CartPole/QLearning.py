import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import gym
import random


# hyperparameters
learning_rate = 0.01
n_episodes = 500
print_interval = 10
gamma = 0.99
goal = 1000


class Q_net(nn.Module):
    def __init__(self, hidden=128):
        super(Q_net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, hidden), nn.ReLU(), 
            nn.Linear(hidden, 2)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        out = self.model(x)
        return out

    def select_act(self, state, epsilon):
        t = random.random()
        if t < epsilon:
            return random.randint(0, 1)

        out = self(state)
        return out.argmax().item()


def main():
    model = Q_net()
    env = gym.make("CartPole-v1")
    score = 0.0

    for epi in range(n_episodes):
        epsilon = 1.0 / (10 + epi // 10)
        state = env.reset()[0]
        state = torch.from_numpy(state)
        done = False

        for _ in range(goal):
            act = model.select_act(state, epsilon)
            state2, reward, done, _, _ = env.step(act)
            state2 = torch.from_numpy(state2)

            # train
            model.optimizer.zero_grad()
            out = model(state)
            target = torch.tensor(reward)
            if not done:
                out2 = model(state2)
                target += gamma * torch.max(out2).item()

            loss = F.mse_loss(out[act], target.detach())
            loss.backward()
            model.optimizer.step()
            
            state = deepcopy(state2)
            score += reward
            if done: break

        if epi % print_interval == print_interval - 1:
            print(f"episode: {epi+1}, average score: {score/print_interval}")
            score = 0.0

    # test
    env = gym.make("CartPole-v1", render_mode="human")
    state = env.reset()[0]
    state = torch.from_numpy(state)
    done = False
    score = 0.0
    for _ in range(goal):
        act = model.select_act(state, epsilon)
        state, reward, done, _, _ = env.step(act)
        state = torch.from_numpy(state)
        score += reward
        if done: break

    print(f"final result: {score}")


if __name__ == "__main__":
    main()
