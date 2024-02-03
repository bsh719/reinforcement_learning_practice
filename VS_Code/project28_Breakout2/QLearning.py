import gym 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import random 
# from collections import deque
from copy import deepcopy
from tqdm import tqdm


# hyperparmeters
learning_rate = 0.0001
gamma = 0.98
n_episodes = 10000
interval = 200
# buffer_limit = 50000
# mini_batch_size = 512


class QNet(nn.Module):
    def __init__(self, hidden1=512, hidden2=64, out_dim=3):
        super(QNet, self).__init__()
        # input: 2x152x88
        self.Conv_layers = nn.Sequential(
            nn.MaxPool2d(2, 2),  # 2x76x44
            nn.Conv2d(2, 4, kernel_size=3, padding=1),  # 4x76x44
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 4x38x22
            nn.Conv2d(4, 6, kernel_size=3),  # 6x36x20
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 6x18x10
            nn.Flatten()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(6*18*10, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, out_dim)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.episodes = []
    
    def forward(self, x):
        x = x.reshape(-1, 2, 152, 88)
        x = self.Conv_layers(x)
        x = self.fc_layers(x)
        return x
    
    def get_act(self, state, epsilon):
        p = random.random()
        if p < epsilon:
            return random.randint(0, 2)

        out = self(state)
        return out.argmax().item()
    
    def train_net(self):
        self.optimizer.zero_grad()

        for transition in self.episodes:
            s, a, r, s2, done = transition
            out = self(s)
            target = torch.tensor(r)
            if not done:
                out2 = self(s2)
                target += gamma * torch.max(out2)
            
            loss = F.mse_loss(out[0, a], target.detach())
            loss.backward()
        
        self.optimizer.step()
        self.episodes = []


def preprocess_state(state, state2):
    # 210 x 160 x 3 -> 1 x 152 x 88
    s_tensor = torch.from_numpy(state[32:208:2, 4:156, 0:1]).float()
    s2_tensor = torch.from_numpy(state2[32:208:2, 4:156, 0:1]).float()
    return torch.transpose(torch.cat([s_tensor, s2_tensor], dim=2), 0, 2)


def main():
    Q = QNet()
    env1 = gym.make("ALE/Breakout-v5")
    env2 = gym.make("ALE/Breakout-v5", render_mode="human")
    best_score = 0.0
    avg_score = 0.0
    
    for epi in tqdm(range(n_episodes)):
        epsilon = max(10. / (10 + epi // 100), 0.01)
        score = 5.0
        E = env2 if epi % interval == 0 else env1
        E.reset()
        
        s, _, _, _, _ = E.step(1)
        s2, _, done, _, _ = E.step(1)
        s_tensor = preprocess_state(s, s2)
        lives = 5
        
        while not done:
            act = Q.get_act(s_tensor, epsilon)
            s, r, _, _, _ = E.step(act + 1)
            s2, r2, done, _, new_info = E.step(1)
            r += r2
            s2_tensor = preprocess_state(s, s2)
            
            new_lives = new_info['lives']
            if lives > new_lives:
                r = -1.0
                lives -= 1
            score += r
            
            Q.episodes.append((s_tensor, act, r, s2_tensor, done))
            s_tensor = deepcopy(s2_tensor)
        
        avg_score += score / interval
        Q.train_net()
        
        if epi % interval == interval - 1:
            print(f"\nepisode: {epi+1}, average score: {avg_score:.2f}")
            avg_score = 0.0
        
        if best_score + 0.1 < score:
            print(f"\nepisode: {epi+1}, best score: {score}")
            best_score = score

    print(f"\n\nResult - - best score: {best_score}!")
    torch.save(Q, "project28_Breakout2/QLearning_model.pth")


if __name__ == "__main__":
    main()
