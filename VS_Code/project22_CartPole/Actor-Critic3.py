import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from torch.distributions import Categorical


# hyperparameters
learning_rate = 0.01
gamma = 0.99
n_episodes = 500
interval = 10
goal = 1000


class ActorCritic(nn.Module):
    def __init__(self, hidden=128):
        super(ActorCritic, self).__init__()
        self.episode = []

        self.fc1 = nn.Sequential(nn.Linear(4, hidden), nn.ReLU())
        self.fc_policy2 = nn.Linear(hidden, 2)
        self.fc_value2 = nn.Linear(hidden, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

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

        pi = self.Policy(s, softmax_dim=1)
        pi_a = pi.gather(1, a)

        self.optimizer.zero_grad()
        loss = -torch.log(pi_a) * advs.detach() + \
            F.mse_loss(
            s_value, td_target.detach()
        )
        loss = torch.sum(loss)
        loss.backward()
        self.optimizer.step()


def main():
    env = gym.make("CartPole-v1")
    model = ActorCritic()
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
    score = 0.
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
