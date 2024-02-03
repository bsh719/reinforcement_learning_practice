import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
import random


# hyperparameters
learning_rate = 0.0001
n_episodes = 10000
gamma = 0.98
interval = 200


class PPO(nn.Module):
    def __init__(self, hidden1=512, hidden2=128, n_out=3):
        super(PPO, self).__init__()
        # 환경: 210x160x3 2개 -> 2x152x88
        self.base = nn.Sequential(
            nn.MaxPool2d(2, 2),  # 2x76x44
            nn.Conv2d(2, 4, kernel_size=3, padding=1),  # 4x76x44
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 4x38x22
            nn.Conv2d(4, 6, kernel_size=3),  # 6x36x20
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 6x18x10
            nn.Flatten(),
            nn.Linear(6*18*10, hidden1),
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
            nn.Softmax(dim=1)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        self.old_param = nn.Sequential(
            deepcopy(self.base), deepcopy(self.policy_layers)
        )
        self.s_lst = []
        self.a_lst = []
        self.r_lst = []
        self.s2_lst = []
        self.mask_lst = []

    def Policy(self, x):
        x = x.reshape(-1, 2, 152, 88)
        x = self.base(x)
        x = self.policy_layers(x)
        return x

    def Value(self, x):
        x = self.base(x)
        x = self.value_layers(x)
        return x
    
    def get_act(self, state):
        out = self.Policy(state)
        p = random.random()
        if p < out[0, 0].item():
            return 0
        elif p + out[0, 0].item() < out[0, 1].item():
            return 1
        return 2

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
        sur_2 = torch.clamp(ratios, 0.99, 1.01) * advs
        clip_loss = -torch.min(sur_1, sur_2)
        self.old_param = nn.Sequential(
            deepcopy(self.base), deepcopy(self.policy_layers)
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


def preprocess_state(state, state2):
    # 210 x 160 x 3 -> 1 x 152 x 88
    s_tensor = torch.from_numpy(state[32:208:2, 4:156, 0:1]).float()
    s2_tensor = torch.from_numpy(state2[32:208:2, 4:156, 0:1]).float()
    return torch.transpose(torch.cat([s_tensor, s2_tensor], dim=2), 0, 2)


def main():
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    env2 = gym.make("ALE/Breakout-v5")
    model = PPO()
    best_score = 0.0
    average_score = 0.0

    for epi in tqdm(range(n_episodes)):
        E = env if epi % interval == 0 else env2
        score = 5.0
        E.reset()
        
        s, _, _, _, _ = E.step(1)
        s2, _, done, _, _ = E.step(1)
        s_tensor = preprocess_state(s, s2)
        lives = 5

        while not done:
            act = model.get_act(s_tensor)
            s, r, _, _, _ = E.step(act + 1)
            s2, r2, done, _, new_info = E.step(1)
            r += r2
            s2_tensor = preprocess_state(s, s2)
            
            new_lives = new_info['lives']
            if lives > new_lives:
                r = -1.0
                lives -= 1
            score += r
            
            model.s_lst.append(s_tensor)
            model.a_lst.append([act])
            model.r_lst.append([r])
            model.s2_lst.append(s2_tensor)
            model.mask_lst.append([0.0 if done else 1.0])
            s_tensor = s2_tensor
        
        average_score += score / interval
        model.train()

        if best_score + 0.1 < score:
            print(f"\nEpisode {epi+1}, best score: {score}")
            best_score = score
        
        if epi % interval == interval - 1:
            print(f"\nEpisode {epi+1}, average score: {average_score:.2f}")
            average_score = 0.0

    print(f"\n\nResult - - best score: {best_score}!")
    torch.save(model, "project28_Breakout2/PPO_model.pth")


if __name__ == "__main__":
    main()
