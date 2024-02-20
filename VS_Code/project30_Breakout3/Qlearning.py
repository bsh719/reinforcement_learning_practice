import gym 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import random
from tqdm import tqdm
from torchvision import transforms


# hyperparmeters
learning_rate = 0.001
gamma = 0.98
n_episodes = 10000
interval = 200


class QNet(nn.Module):
    def __init__(self, hidden1=512, hidden2=64, out_dim=3):
        super(QNet, self).__init__()
        # input: 2x64x64
        self.Conv_layers = nn.Sequential(
            nn.Conv2d(2, 6, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),  # 6x32x32
            nn.ReLU(),
            nn.Conv2d(6, 12, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),  # 12x16x16
            nn.ReLU(),
            nn.Conv2d(12, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),  # 16x8x8
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(16*8*8, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, out_dim)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.episodes = []
    
    def forward(self, x):
        x = x.reshape(-1, 2, 64, 64)
        x = self.Conv_layers(x)
        x = self.fc_layers(x)
        return x
    
    def get_act(self, state, epsilon=-1.0):
        p = random.random()
        if p < epsilon:
            return random.randint(0, 2)

        out = self(state)
        return out.argmax().item()
    
    def train_net(self):
        self.optimizer.zero_grad()
        s_, a_, r_, s2_, m_ = [], [], [], [], []
        
        for transition in self.episodes:
            s, a, r, s2, done = transition
            s_.append(s)
            a_.append([a])
            r_.append([r])
            s2_.append(s2)
            m_.append([0.0 if done else 1.0])
        
        s = torch.stack(s_)
        a = torch.tensor(a_)
        r = torch.tensor(r_)
        s2 = torch.stack(s2_)
        m = torch.tensor(m_)

        out = self(s)
        out_a = out.gather(1, a)
        out2 = self(s2)
        target = r + gamma * torch.amax(out2, dim=1).reshape(-1, 1) * m
        
        loss = F.mse_loss(out_a, target.detach())
        loss.backward()
        self.optimizer.step()
        self.episodes = []


def preprocess_state(s1, s2):
    # 210 x 160 x 3 -> 1 x 144 x 168 -> 3 x 64 x 64
    t1 = torch.from_numpy(s1[32:200, 8:152, 0:1]) / 256
    t2 = torch.from_numpy(s2[32:200, 8:152, 0:1]) / 256
    state = torch.transpose(torch.cat([t1, t2], dim=2), 0, 2)
    return transforms.Resize([64, 64])(state)


def main():
    Q = QNet()
    env = gym.make("ALE/Breakout-v5")
    env_vis = gym.make("ALE/Breakout-v5", render_mode="human")
    best_score = 0.0
    avg_score = 0.0
    
    for epi in tqdm(range(n_episodes)):
        E = env_vis if epi % interval == 0 else env
        E.reset()
        epi += 1
        epsilon = max(10. / (10 + epi // 100), 0.05)
        
        s1, _, _, _, _ = E.step(1)
        s2, _, done, _, _ = E.step(1)
        s_tensor = preprocess_state(s1, s2)
        lives = 5
        score = 5.0
        
        while not done:
            act = Q.get_act(s_tensor, epsilon)
            s1, r, _, _, _ = E.step(act + 1)
            s2, r2, done, _, new_info = E.step(1)
            r += r2
            s2_tensor = preprocess_state(s1, s2)
            
            new_lives = new_info['lives']
            if lives > new_lives:
                r = -1.0
                lives -= 1
            score += r
            
            Q.episodes.append((s_tensor, act, r, s2_tensor, done))
            s_tensor = s2_tensor
        
        Q.train_net()
        avg_score += score / interval
        
        if epi % interval == 0:
            print(f"\nepisode: {epi}, average score: {avg_score:.2f}")
            avg_score = 0.0
        
        if best_score + 0.1 < score:
            print(f"\nepisode: {epi}, best score: {score}")
            best_score = score

    print(f"\n\nResult - - best score: {best_score}!")
    torch.save(Q, "project30_Breakout3/Qmodel.pth")


if __name__ == "__main__":
    main()
