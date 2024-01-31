import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from torch.distributions import Categorical
from copy import deepcopy


# hyperparameters
learning_rate = 0.01
gamma = 0.99
n_episodes = 500
interval = 10
goal = 1000


class PPO(nn.Module):
    def __init__(self, hidden=128):
        super(PPO, self).__init__()
        self.episode = []

        self.fc1 = nn.Sequential(nn.Linear(4, hidden), nn.ReLU())
        self.fc_policy2 = nn.Linear(hidden, 2)
        self.fc_value2 = nn.Linear(hidden, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.old_param = nn.Sequential(
            deepcopy(self.fc1), deepcopy(self.fc_policy2), nn.Softmax(dim=1)
        )

    def Policy(self, x, softmax_dim=0):
        x = self.fc1(x)
        x = self.fc_policy2(x)
        x = F.softmax(x, dim=softmax_dim)
        return x

    def Value(self, x):
        x = self.fc1(x)
        x = self.fc_value2(x)
        return x

    def put_data(self, x):
        self.episode.append(x)

    def make_batch(self):
        s_lst, a_lst, r_lst, s2_lst, mask_lst = [], [], [], [], []
        for transition in self.episode:
            s, a, r, s2, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r / 100.0])
            s2_lst.append(s2)
            done_mask = 0.0 if done else 1.0
            mask_lst.append([done_mask])

        s = torch.tensor(s_lst)
        a = torch.tensor(a_lst)
        r = torch.tensor(r_lst)
        s2 = torch.tensor(s2_lst)
        m = torch.tensor(mask_lst)
        self.episode = []
        return s, a, r, s2, m

    def train(self):
        s, a, r, s2, mask = self.make_batch()
        td_target = r + gamma * self.Value(s2) * mask
        s_value = self.Value(s)
        advs = td_target - s_value
        advs = advs.detach()

        pi = self.Policy(s, softmax_dim=1)
        pi_a = pi.gather(1, a)
        old_pi = self.old_param(s).detach()
        old_a = old_pi.gather(1, a)

        ratios = torch.div(pi_a, old_a)
        sur_1 = ratios * advs
        sur_2 = torch.clamp(ratios, 0.99, 1.01) * advs
        clip_loss = -torch.min(sur_1, sur_2)
        self.old_param = nn.Sequential(
            deepcopy(self.fc1), deepcopy(self.fc_policy2), nn.Softmax(dim=1)
        )

        self.optimizer.zero_grad()
        loss = clip_loss + F.smooth_l1_loss(s_value, td_target.detach())
        loss = torch.sum(loss)
        loss.backward()
        self.optimizer.step()


def main():
    env = gym.make("CartPole-v1")
    model = PPO()
    score = 0.0

    for n_epi in range(n_episodes):
        done = False
        state = env.reset()[0]

        for _ in range(goal):
            prob = model.Policy(torch.tensor(state))
            m = Categorical(prob)
            action = m.sample().item()
            s2, r, done, _, _ = env.step(action)

            model.put_data((state, action, r, s2, done))
            state = s2
            score += r
            if done: break

        model.train()

        if n_epi % interval == interval - 1:
            print(f"Episode: {n_epi+1}, average score: {score/interval}")
            score = 0.0
    
    env = gym.make("CartPole-v1", render_mode="human")
    state = env.reset()[0]
    done = False
    for _ in range(goal * 2):
        prob = model.Policy(torch.tensor(state))
        m = Categorical(prob)
        action = m.sample().item()
        s2, r, done, _, _ = env.step(action)

        state = s2
        score += r
        if done: break

    print("final score:", score)

if __name__ == "__main__":
    main()
