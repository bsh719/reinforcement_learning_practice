# 실패
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
import random


# hyperparameters
learning_rate = 0.001
n_episodes = 10000
gamma = 0.99
interval = 100


class PPO(nn.Module):
    def __init__(self, hidden1=512, hidden2=128, n_out=2):
        super(PPO, self).__init__()
        # 환경: 210x160x3 2개 -> 2x140x100
        self.base = nn.Sequential(
            nn.Conv2d(2, 6, kernel_size=8, stride=4),   # 6x34x24
            nn.Conv2d(6, 16, kernel_size=4, stride=2),  # 16x16x11
            nn.Flatten(),
            nn.Linear(256*11, hidden1),
            nn.ReLU(),
        )
        self.value_layers = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )
        self.policy_layers = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, n_out),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        self.old_param = nn.Sequential(
            deepcopy(self.base), deepcopy(self.policy_layers), nn.Softmax(dim=1)
        )
        self.s_lst = []
        self.a_lst = []
        self.r_lst = []
        self.s2_lst = []
        self.mask_lst = []

    def Policy(self, x):
        x = x.reshape(-1, 2, 140, 100)
        x = self.base(x)
        x = self.policy_layers(x)
        x = F.softmax(x, dim=1)
        return x

    def Value(self, x):
        x = self.base(x)
        x = self.value_layers(x)
        return x
    
    def act(self, state):
        out = self.Policy(state)
        p = random.random()
        if p < out[0, 0].item(): return 0
        return 1

    def train(self):
        s = torch.stack(self.s_lst)
        a = torch.tensor(self.a_lst)
        r = torch.tensor(self.r_lst)
        s2 = torch.stack(self.s2_lst)
        mask = torch.tensor(self.mask_lst)
        
        td_target = r + gamma * self.Value(s2) * mask
        s_value = self.Value(s)
        advs = td_target - s_value
        advs = advs.detach()

        pi = self.Policy(s)
        pi_a = pi.gather(1, a)
        old_pi = self.old_param(s).detach()
        old_a = old_pi.gather(1, a)

        ratios = torch.div(pi_a, old_a)
        sur_1 = ratios * advs
        sur_2 = torch.clamp(ratios, 0.9, 1.1) * advs
        clip_loss = -torch.min(sur_1, sur_2)
        self.old_param = nn.Sequential(
            deepcopy(self.base), deepcopy(self.policy_layers), nn.Softmax(dim=1)
        )

        self.optimizer.zero_grad()
        loss = clip_loss + F.mse_loss(s_value, td_target.detach())
        loss = torch.sum(loss)
        loss.backward()
        self.optimizer.step()
        
        self.s_lst = []
        self.a_lst = []
        self.r_lst = []
        self.s2_lst = []
        self.mask_lst = []


def preprocess_state(state):
    # 210 x 160 x 3 -> 1 x 140 x 100
    state = state[5:205:2, 20:, 0:1]
    s_tensor = torch.from_numpy(state).float()
    return torch.transpose(s_tensor, 0, 2)

def main():
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    env2 = gym.make("ALE/Breakout-v5")
    model = PPO()
    best_score = 0.0
    average_score = 0.0

    for epi in tqdm(range(n_episodes)):
        E = env if epi % interval == 0 else env2
        score = 0.
        
        E.reset()
        state, r, done, _, _ = E.step(1)
        state_prev = preprocess_state(state)
        state, r, done, _, _ = E.step(1)
        state = preprocess_state(state)

        while not done:
            act = model.act(torch.cat([state_prev, state]))
            s2, r, done, _, _ = E.step(act + 2)
            if not done:
                s2, r2, done, _, _ = E.step(1)
                r += r2
            s2 = preprocess_state(s2)
            
            model.s_lst.append(torch.cat([state_prev, state]))
            model.a_lst.append([act])
            model.r_lst.append([r / 100.0])
            model.s2_lst.append(torch.cat([state, s2]))
            model.mask_lst.append([0.0 if done else 1.0])
            state_prev = deepcopy(state)
            state = s2
            score += r
        
        average_score += score / interval
        model.train()

        if score > best_score + 0.5:
            print(f"\nEpisode {epi+1}, best score: {score}")
            best_score = score
        
        if epi % interval == interval - 1:
            print(f"\nEpisode {epi+1}, average score: {average_score}")
            average_score = 0.0

    print(f"\n\nResult - - best score: {best_score}!")
    torch.save(model, "project24_Breakout/PPO_model.pt")


if __name__ == "__main__":
    main()
