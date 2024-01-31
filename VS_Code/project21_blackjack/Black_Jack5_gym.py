import gym 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.distributions import Categorical

import matplotlib.pyplot as plt
from tqdm import tqdm 


# REINFORCE algorithm
# hyper-parameters
learning_rate = 0.001
n_episodes = 50000
interval = 1000
gamma = 0.99


class Policy(nn.Module):
    def __init__(self, hidden1=256, hidden2=64):
        super(Policy, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, 2)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        self.rewards = []
        self.log_probs = []
    
    def forward(self, x, softmax_dim=0):
        x = self.layers(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def select_act(self, state):
        usable_ace = 1. if state[2] else -1.
        state = torch.tensor([state[0], state[1], usable_ace], dtype=torch.float)
        out = self(state)
        
        m = Categorical(out)
        act = m.sample()
        
        log_prb = m.log_prob(act)
        self.log_probs.append(log_prb)
        return act.item()
    
    def train(self):
        self.optimizer.zero_grad()
        
        L = self.log_probs.__len__()
        R = 0.
        for i in reversed(range(L)):
            R = self.rewards[i] + gamma * R
            loss = -R * self.log_probs[i]
            loss.backward()
        
        self.optimizer.step()
        self.rewards = []
        self.log_probs = []


def main():
    pi = Policy()
    env = gym.make("Blackjack-v1")
    win_cnt = 0
    win_lose_cnt = 0
    result_winrate = []
    
    for epi in tqdm(range(n_episodes)):
        state = env.reset()[0]
        done = False
        score = 0
        
        while not done:
            act = pi.select_act(state)
            state, reward, done, _, _ = env.step(act)
            pi.rewards.append(reward)
            score += reward
        
        if score < -0.9:
            win_lose_cnt += 1
        elif score > 0.9:
            win_cnt += 1
            win_lose_cnt += 1
        
        pi.train()
        
        if epi % interval == interval - 1:
            result_winrate.append(win_cnt / win_lose_cnt)
            win_cnt = 0
            win_lose_cnt = 0

    plt.plot(range(len(result_winrate)), result_winrate, color='red')
    plt.show()

if __name__ == "__main__":
    main()
