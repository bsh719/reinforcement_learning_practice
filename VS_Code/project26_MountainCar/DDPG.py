import torch 
import torch.nn as nn 
import gym 
import torch.nn.functional as F
import random
from collections import deque

# hyperparameters
n_episodes = 300
interval = 20
gamma = 0.95
buffer_limit = 50000
batch_size = 128
learning_rate = 0.001


class ReplayBuffer():
    def __init__(self):
        self.buffer = deque(maxlen=buffer_limit)
    
    def sample(self, n):
        batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s2_lst, m_lst = [], [], [], [], []
        
        for s, a, r, s2, done in batch:
            s_lst.append(s.tolist())
            a_lst.append([a])
            r_lst.append([r])
            s2_lst.append(s2.tolist())
            m_lst.append([0.0 if done else 1.0])
        
        s = torch.tensor(s_lst)
        a = torch.tensor(a_lst)
        r = torch.tensor(r_lst)
        s2 = torch.tensor(s2_lst)
        m = torch.tensor(m_lst)
        return s, a, r, s2, m


class DDPG(nn.Module):
    def __init__(self):
        super(DDPG, self).__init__()
        
        self.critic_layers = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.actor_layers = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def get_Q(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = self.critic_layers(x)
        return x
    
    def select_act(self, s):
        a = self.actor_layers(s)
        a = torch.clamp(a, -1., 1.)
        return a


def main():
    env1 = gym.make("MountainCarContinuous-v0")
    env2 = gym.make("MountainCarContinuous-v0", render_mode="human")
    ddpg = DDPG()
    optimizer_Q = torch.optim.Adam(ddpg.critic_layers.parameters(), lr=learning_rate)
    optimizer_A = torch.optim.Adam(ddpg.actor_layers.parameters(), lr=learning_rate)
    RB = ReplayBuffer()
    average_score = 0.0
    
    for epi in range(n_episodes):
        epi += 1
        E = env2 if epi == n_episodes else env1
        state = E.reset()[0]
        state = torch.from_numpy(state)
        done = False
        score = 0.0
        #epsilon = max(0.1, 1.0 - 0.1 * (epi // 20))
        epsilon = max(0.1, 1.0 / (1 + epi // 20))
        
        for t in range(10000):
            tmp = random.random()
            if tmp < epsilon:
                act = random.randint(0, 1) * 2 - 1
            else:
                act = ddpg.select_act(state).item()
            
            state2, rew, done, _, _ = E.step([act])
            state2 = torch.from_numpy(state2)
            #rew = rew if done else -0.1  # 기존 보상체계처럼 변환
            score += rew
            RB.buffer.append((state, act, rew, state2, done))
            
            # training
            N = min(1 + len(RB.buffer) // 10, batch_size)
            s, a, r, s2, m = RB.sample(N)
            
            q_out = ddpg.get_Q(s, a)
            target = r + gamma * ddpg.get_Q(s2, ddpg.select_act(s2)) * m
            optimizer_Q.zero_grad()
            loss = F.mse_loss(q_out, target.detach())
            loss.backward()
            optimizer_Q.step()
            
            optimizer_A.zero_grad()
            loss2 = -ddpg.get_Q(s, ddpg.select_act(s))
            loss2 = torch.mean(loss2)
            loss2.backward()
            optimizer_A.step()
            
            if done: break
            state = state2
        
        average_score += score / interval
        print(f"t: {t}, score: {score:.2f}")

        if epi % interval == 0:
            print(f"\nepisode {epi-interval+1} ~ {epi}, avg score: {average_score:.2f}\n")
            average_score = 0.0
    
    torch.save(ddpg, "project26_MountainCar/DDPG2_model.pth")
    

if __name__ == "__main__":
    main()
