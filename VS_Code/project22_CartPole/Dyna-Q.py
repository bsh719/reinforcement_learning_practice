import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import random
from collections import deque


# hyperparameters
learning_rate_T = 0.005
learning_rate_Q = 0.005
n_episodes = 250
print_interval = 10
gamma = 0.99
goal = 1000
buffer_limit = 5000


class Transition(nn.Module):
    def __init__(self, hidden=256):
        super(Transition, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, hidden), nn.ReLU(), 
            nn.Linear(hidden, 8)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate_T)
        self.buffer = deque(maxlen=buffer_limit)
    
    def forward(self, x, a):
        x = self.model(x)
        if a == 0:
            return x[0:4]
        return x[4:8]


class Q_net(nn.Module):
    def __init__(self, hidden=128):
        super(Q_net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, hidden), nn.ReLU(), 
            nn.Linear(hidden, 2)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate_Q)
        self.buffer = deque(maxlen=buffer_limit)

    def forward(self, x):
        out = self.model(x)
        return out

    def select_act(self, state, epsilon=-1.0):
        t = random.random()
        if t < epsilon:
            return random.randint(0, 1)

        out = self(state)
        return out.argmax().item()


def main():
    Q = Q_net()
    T = Transition()
    env = gym.make("CartPole-v1")
    score = 0.0

    for epi in range(n_episodes):
        epsilon = 1.0 / (5 + epi // 10)
        state = env.reset()[0]
        state = torch.from_numpy(state)
        done = False

        for _ in range(goal):
            act = Q.select_act(state, epsilon)
            state2, rew, done, _, _ = env.step(act)
            state2 = torch.from_numpy(state2)
            score += rew
            Q.buffer.append((state, act, done))
            T.buffer.append((state, act, state2))
            
            # Update T
            for _ in range(4):
                T.optimizer.zero_grad()
                s, a, s2 = random.sample(T.buffer, 1)[0]
                out = T(s, a)
                loss = F.mse_loss(out, s2.detach())
                loss.backward()
                T.optimizer.step()
            
            # Update Q
            Q.optimizer.zero_grad()
            out = Q(state)
            target = torch.tensor(1.0)
            if not done:
                out2 = Q(state2)
                target += gamma * torch.max(out2).item()
            loss = F.mse_loss(out[act], target.detach())
            loss.backward()
            Q.optimizer.step()
            
            for _ in range(2):
                Q.optimizer.zero_grad()
                s, a, d = random.sample(Q.buffer, 1)[0]
                out = Q(s)
                target = torch.tensor(1.0)
                if not d:
                    s2 = T(s, a)
                    out2 = Q(s2)
                    target += gamma * torch.max(out2).item()
                loss = F.mse_loss(out[a], target.detach())
                loss.backward()
                Q.optimizer.step()

            state = state2
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
        act = Q.select_act(state)
        state, r, done, _, _ = env.step(act)
        state = torch.from_numpy(state)
        score += r
        if done: break

    print(f"final result: {score}")


if __name__ == "__main__":
    main()
