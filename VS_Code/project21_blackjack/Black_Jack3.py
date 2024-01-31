import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.distributions import Categorical 

import random 
import matplotlib.pyplot as plt 
from copy import deepcopy 
from tqdm import tqdm 

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


# 환경: 상태를 숫자 대신 1~11 카드 보유 상태로 표현
# 딜러: 에이스는 무조건 1
class Environment():
    def __init__(self, disclosed=False):
        self.cnt = 52
        self.remaining = [4, 4, 4, 4, 4, 4, 4, 4, 4, 16]
        self.dealer = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float).to(device)
        self.player = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float).to(device)
        self.dealer_tot = 0
        self.player_tot = 0
        
        open_lst = []
        for _ in range(3):
            N = random.randint(1, self.cnt)
            for i, n in enumerate(self.remaining):
                N -= n
                if N <= 0:
                    open_lst.append(i)
                    self.remaining[i] -= 1
                    break
            self.cnt -= 1
        
        self.dealer[open_lst[0]] = 1
        self.dealer_tot = open_lst[0] + 1

        if open_lst[1] != 0:
            self.player[open_lst[1]] = 1
            self.player_tot = open_lst[1] + 1
        else:
            self.player[10] = 1
            self.player_tot = 11
        if open_lst[2] != 0:
            self.player[open_lst[2]] += 1
            self.player_tot += open_lst[2] + 1
        else:
            if self.player_tot == 11:
                self.player[0] = 1
                self.player_tot = 12
            else:
                self.player[10] = 1
                self.player_tot += 11
        
        if disclosed:
            print('\ndealer:', open_lst[0]+1)
            print('player:', open_lst[1]+1, open_lst[2]+1)
    
    def step(self, action, disclosed=False):
        # action - 0: stand, 1: hit
        if action == 0:
            if disclosed: print('player: stand!')
            
            while self.dealer_tot < 17:
                N = random.randint(1, self.cnt)
                for i, n in enumerate(self.remaining):
                    N -= n
                    if N <= 0:
                        self.remaining[i] -= 1
                        self.dealer_tot += i+1
                        self.dealer[i] += 1
                        if disclosed: print('dealer:', i+1)
                        break
                self.cnt -= 1
            
            reward = -1
            if self.dealer_tot > 21: 
                reward = 1
                if disclosed: print('dealer: bust!')
            elif self.player_tot > self.dealer_tot:
                reward = 1
                if disclosed: print('Win!')
            elif self.player_tot == self.dealer_tot:
                reward = 0
                if disclosed: print('Draw!')
            elif disclosed: print('Lose!')
            
            return torch.cat((self.player, self.dealer)), reward, True
        else:
            N = random.randint(1, self.cnt)
            for i, n in enumerate(self.remaining):
                N -= n
                if N <= 0:
                    self.remaining[i] -= 1
                    if i == 0: i == 10
                    self.player_tot += i+1
                    self.player[i] += 1
                    if disclosed: print('player:', i+1)
                    break
            self.cnt -= 1
            
            while self.player_tot > 21:
                if self.player[10] > 0.1:
                    self.player[10] -= 1
                    self.player[0] += 1
                    self.player_tot -= 10
                else: 
                    if disclosed: print('player: bust!')
                    return torch.cat((self.player, self.dealer)), -1, True
            
            return torch.cat((self.player, self.dealer)), 0, False


class PolicyNet(nn.Module):
    def __init__(self, hidden1=128, hidden2=64, learning_rate=0.001):
        super(PolicyNet, self).__init__()
        self.data = []
        self.fc1 = nn.Linear(22, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 2)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=0)
        return x

    def put_data(self, x):
        self.data.append(x)
    
    def train(self, gamma):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -R * torch.log(prob)
            loss.backward()
        self.optimizer.step()
        self.data = []


def main():
    n_episodes = 100000
    printing_range = 10000
    gamma = 0.99
    learning_rate = 0.001
    Pi = PolicyNet(learning_rate=learning_rate).to(device)
    
    #result_lst = []
    winrate_lst = []  # 무승부를 제외한 승률
    #range_lst = []
    range_lst2 = []  # 무승부를 제외한 결과
    
    for n_epi in tqdm(range(n_episodes)):
        disclosed = True if n_epi%printing_range == printing_range-1 else False
        E = Environment(disclosed)
        
        state = torch.cat((E.player, E.dealer))
        done = False
        
        while not done:
            prob = Pi(state.to(device))
            m = Categorical(prob)
            action = m.sample()
            
            state2, reward, done = E.step(action, disclosed)
            Pi.put_data((reward, prob[action]))
            state = deepcopy(state2)
        
        Pi.train(gamma)
        
        #range_lst.append(reward)
        if reward != 0: range_lst2.append(max(reward, 0))
        
        if n_epi % printing_range == printing_range-1:
            #result_lst.append(sum(range_lst) / len(range_lst))
            winrate_lst.append(sum(range_lst2) / len(range_lst2))
            #range_lst = []
            range_lst2 = []
    
    # visualize
    #plt.plot(range(len(result_lst)), result_lst, color='blue')
    plt.plot(range(len(winrate_lst)), winrate_lst, color='red')
    plt.show()

if __name__ == "__main__":
    main()
