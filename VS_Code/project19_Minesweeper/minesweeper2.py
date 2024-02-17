import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# Q-learning
# 지뢰찾기 9x9 보드에 지뢰 10개, 중앙 5x5에는 지뢰 없음.
# 각 자리의 상태: 10개, 이미 열린 자리도 누르면 실패.

# hyperparameters
n_episodes = 500
gamma = 0.99
interval = 10
learning_rate = 0.001


class Environment:
    def __init__(self):
        # 0~8: 주변 지뢰 숫자, 9: 지뢰
        self.board = torch.zeros(10, 9, 9)
        mine = 0
        while mine < 10:
            x = random.randint(0, 8)
            y = random.randint(0, 8)
            if x >= 2 and x <= 6 and y >= 2 and y <= 6:
                continue
            if self.board[9, x, y] == 1:
                continue
            mine += 1
            self.board[9, x, y] = 1

        for x in range(0, 8):
            for y in range(0, 8):
                if self.board[9, x, y] == 1:
                    continue

                cnt = 0
                for dx in [-1, 0, 1]:
                    if x + dx < 0 or x + dx > 8:
                        continue
                    for dy in [-1, 0, 1]:
                        if y + dy < 0 or y + dy > 8:
                            continue
                        if self.board[9, x + dx, y + dy] == 1:
                            cnt += 1

                self.board[cnt, x, y] = 1

        # 현재 상태
        self.state = torch.zeros(10, 9, 9)
        self.remained = 81 - mine  # 9x9 - 10

    def step(self, action):
        ax = action // 9
        ay = action % 9
        if torch.sum(self.state[:, ax, ay]) != 0:
            # 이미 열린 자리도 누르면 실패
            return (self.state, -100, True)
        
        if self.board[9, ax, ay] == 1:
            return (self.state, -100, True)

        self.state[:, ax, ay] = self.board[:, ax, ay]
        self.remained -= 1
        return (self.state, 1, self.remained == 0)


class Qnet(nn.Module):
    def __init__(self, hidden1=256, hidden2=256):
        super(Qnet, self).__init__()
        self.cv1 = nn.Conv2d(10, 3, 3, padding=1)
        self.fc1 = nn.Linear(243, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 81)

    def forward(self, x):
        x = F.relu(self.cv1(x))
        x = x.reshape(-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def slt_action(self, state, epsilon):
        t = random.random()
        if t < epsilon:
            return random.randint(0, 80)
        else:
            out = self.forward(state.to(device))
            return torch.argmax(out)


def main():
    Q = Qnet().to(device)
    optimizer = torch.optim.Adam(Q.parameters(), lr=learning_rate)
    result_lst = []
    average_lst = []
    average_rew = 0.0

    for n_epi in tqdm(range(n_episodes)):
        env = Environment()
        state = torch.zeros(10, 9, 9)
        epsilon = 1.0 / (n_epi // 5 + 2)
        done = False
        score = 0

        while not done:
            action = Q.slt_action(state, epsilon)
            state2, reward, done = env.step(action)

            # training
            out = Q(state.to(device))
            target = torch.tensor(reward, dtype=torch.float)
            if not done:
                out2 = Q(state2.to(device))
                target += gamma * torch.max(out2)
               
            optimizer.zero_grad() 
            loss = F.mse_loss(out[action], target.detach())
            loss.backward()
            optimizer.step()

            state = state2
            score += reward

        average_rew += score / interval
        if n_epi % interval == interval - 1:
            result_lst.append(score)
            average_lst.append(average_rew)
            average_rew = 0.0

    # visualize
    print(result_lst)
    plt.plot(range(len(result_lst)), result_lst, color="blue")
    plt.plot(range(len(average_lst)), average_lst, color="red")
    plt.show()


if __name__ == "__main__":
    main()
