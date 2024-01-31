import gym
import torch
import torch.nn as nn
from torch.distributions import Categorical


# hyperparameters
gamma = 0.99
learning_rate = 0.01
n_episodes = 500
interval = 10
goal = 1000


class Pi(nn.Module):
    def __init__(self, in_dim=4, out_dim=2, hidden_size=128):
        super(Pi, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim),
            nn.Softmax(dim=0),
        )
        self.reset()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def reset(self):
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.model(x)
        return x

    def act(self, state):
        x = torch.tensor(state)
        out = self.forward(x)
        pd = Categorical(probs=out)
        action = pd.sample()
        log_prob = pd.log_prob(action)
        self.log_probs.append(log_prob)
        return action.item()

    def train(self):
        T = len(self.rewards)
        R = 0.0
        self.optimizer.zero_grad()

        for t in reversed(range(T)):
            R = self.rewards[t] + gamma * R
            loss = -R * self.log_probs[t]
            loss.backward()

        self.optimizer.step()


def main():
    env = gym.make("CartPole-v1")
    pi = Pi()
    tot_rew = 0.0

    for epi in range(n_episodes):
        state = env.reset()[0]
        done = False
        for _ in range(goal):
            action = pi.act(state)
            state, reward, done, _, _ = env.step(action)
            pi.rewards.append(reward)
            if done: break

        pi.train()
        tot_rew += sum(pi.rewards)
        pi.reset()

        if epi % interval == interval - 1:
            print(f"Episode {epi+1}, average reward: {tot_rew/interval}")
            tot_rew = 0.

    env = gym.make("CartPole-v1", render_mode="human")
    state = env.reset()[0]
    done = False
    score = 0.
    for _ in range(goal * 2):
        action = pi.act(state)
        state, reward, done, _, _ = env.step(action)
        score += reward
        if done: break

    print("final score:", score)


if __name__ == "__main__":
    main()
