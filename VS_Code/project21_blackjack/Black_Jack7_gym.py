import gym 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import matplotlib.pyplot as plt
from tqdm import tqdm 
import random 


# QLearning algorithm
# hyper-parameters
learning_rate = 0.001
n_episodes = 50000
interval = 1000
gamma = 0.99

def preprocess(state):
    usable_ace = 1. if state[2] else -1.
    return torch.tensor([state[0], state[1], usable_ace], dtype=torch.float)


class QNet(nn.Module):
    def __init__(self, hidden1=256, hidden2=64):
        super(QNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, 2)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.episode = []
    
    def forward(self, x):
        x = self.layers(x)
        return x

    def select_act(self, state, epsilon=0.01):
        p = random.random()
        if p < epsilon:
            return random.randint(0, 1)
        out = self(preprocess(state))
        return out.argmax().item()
    
    def train(self):
        self.optimizer.zero_grad()
        
        for s, a, r, s2, done in self.episode:
            out = self(preprocess(s))
            target = torch.tensor(r)
            if not done:
                out2 = self(preprocess(s2))
                target += gamma * torch.max(out2)
            loss = F.smooth_l1_loss(out[a], target.detach())
            loss.backward()
        
        self.optimizer.step()
        self.episode = []


def main():
    Q = QNet()
    env = gym.make("Blackjack-v1")
    win_cnt = 0
    win_lose_cnt = 0
    result_winrate = []
    
    for epi in tqdm(range(n_episodes)):
        state = env.reset()[0]
        done = False
        score = 0
        
        while not done:
            act = Q.select_act(state)
            state2, reward, done, _, _ = env.step(act)
            Q.episode.append((state, act, reward, state2, done))
            score += reward
            state = state2
        
        if score < -0.9:
            win_lose_cnt += 1
        elif score > 0.9:
            win_cnt += 1
            win_lose_cnt += 1
        
        Q.train()
        
        if epi % interval == interval - 1:
            result_winrate.append(win_cnt / win_lose_cnt)
            win_cnt = 0
            win_lose_cnt = 0

    plt.plot(range(len(result_winrate)), result_winrate, color='red')
    plt.show()

if __name__ == "__main__":
    main()
