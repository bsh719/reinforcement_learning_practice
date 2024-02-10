import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# hyperparameters
n_episodes = 300
interval = 30
gamma = 0.99
learning_rate = 0.01


class QNet(nn.Module):
    def __init__(self, hidden1=64, in_dim=2, out_dim=2):
        super(QNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden1), nn.ReLU(), nn.Linear(hidden1, out_dim)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.episode = []

    def forward(self, x):
        x = self.layers(x)
        return x

    def select_act(self, state, epsilon=0.1):
        p = random.random()
        if p < epsilon:
            return random.randint(0, 1)

        out = self(torch.from_numpy(state))
        return out.argmax().item()

    def train(self):
        self.optimizer.zero_grad()

        for s, a, r, s2, done in self.episode:
            out = self(torch.from_numpy(s))
            target = torch.tensor(r)
            if not done:
                out2 = self(torch.from_numpy(s2))
                target += gamma * torch.max(out2)

            loss = F.mse_loss(out[a], target.detach())
            loss.backward()

        self.optimizer.step()
        self.episode = []


def main():
    env1 = gym.make("MountainCar-v0")
    env2 = gym.make("MountainCar-v0", render_mode="human")
    Q = QNet()
    average_score = 0.0

    for epi in range(n_episodes):
        E = env2 if epi == n_episodes - 1 else env1
        state = E.reset()[0]
        done = False
        epsilon = max(1.0 / (1 + epi // 10), 0.1)
        score = 0.0

        while not done:
            act = Q.select_act(state, epsilon)
            state2, r, done, _, _ = E.step(act * 2)
            score += r
            if score < -5000:
                done = True

            Q.episode.append((state, act, r, state2, done))
            state = state2
            Q.train()

        average_score += score / interval
        print(score)

        if epi % interval == interval - 1:
            print(f"\nepisode {epi-interval+2} ~ {epi+1}, avg score: {average_score}")
            average_score = 0.0

    torch.save(Q, "project26_MountainCar/QLearning_model.pth")


if __name__ == "__main__":
    main()
