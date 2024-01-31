import random 
import matplotlib.pyplot as plt 
import numpy as np 
from tqdm import tqdm 


class Environment():
    def __init__(self, disclosed=False):
        self.cnt = 52
        self.remaining = [4, 4, 4, 4, 4, 4, 4, 4, 4, 16]
        self.dealer = [0, 0]
        self.player = [0, 0]
        
        open_lst = []
        for _ in range(3):
            N = random.randint(1, self.cnt)
            for i, n in enumerate(self.remaining):
                N -= n
                if N <= 0:
                    open_lst.append(i+1)
                    self.remaining[i] -= 1
                    break
            self.cnt -= 1
        
        self.dealer[0] = open_lst[0]

        if open_lst[1] != 1:
            self.player[0] = open_lst[1]
        else:
            self.player = [11, 1]
        
        if open_lst[2] != 1:
            self.player[0] += open_lst[2]
        else:
            if open_lst[1] == 1:
                self.player[0] = 12
            else:
                self.player[0] += 11
                self.player[1] = 1
        
        if disclosed:
            print('\ndealer:', open_lst[0])
            print('player:', open_lst[1], open_lst[2])
    
    def get_state(self):
        return (self.dealer[0], self.player[0], self.player[1])
    
    def step(self, action, disclosed=False):
        # action - 0: stand, 1: hit
        if action == 0:
            if disclosed: print('player: stand!')
            
            while self.dealer[0] < 17:
                N = random.randint(1, self.cnt)
                for i, n in enumerate(self.remaining):
                    N -= n
                    if N <= 0:
                        self.remaining[i] -= 1
                        i += 1
                        self.dealer[0] += i
                        if disclosed: print('dealer:', i)
                        break
                self.cnt -= 1
            
            reward = -1
            if self.dealer[0] > 21: 
                reward = 1
                if disclosed: print('dealer: bust!')
            elif self.player[0] > self.dealer[0]:
                reward = 1
                if disclosed: print('Win!')
            elif self.player[0] == self.dealer[0]:
                reward = 0
                if disclosed: print('Draw!')
            elif disclosed: print('Lose!')
            
            return (self.dealer[0], self.player[0], self.player[1]), reward, True
        else:
            N = random.randint(1, self.cnt)
            for i, n in enumerate(self.remaining):
                N -= n
                if N <= 0:
                    self.remaining[i] -= 1
                    if i == 0:
                        i = 11
                        self.player[1] += 1
                    else: i += 1
                    
                    self.player[0] += i
                    if disclosed: print('player:', i)
                    break
            self.cnt -= 1
            
            while self.player[0] > 21:
                if self.player[1] > 0:
                    self.player[1] -= 1
                    self.player[0] -= 10
                else: 
                    if disclosed: print('player: bust!')
                    return (self.dealer[0], self.player[0], self.player[1]), -1, True
            
            return (self.dealer[0], self.player[0], self.player[1]), 0, False
    

def select_action(Q, state, epsilon):
    t = random.random()
    if t < epsilon:
        return random.randint(0, 1)
    else:
        return np.argmax(np.array(Q[state[0], state[1], state[2], :]))

def main():
    Q = np.zeros([12, 22, 2, 2], dtype=float)
    n_episodes = 100000
    printing_range = 10000
    gamma = 0.99
    
    #result_lst = []
    winrate_lst = []  # 무승부를 제외한 승률
    #range_lst = []
    range_lst2 = []  # 무승부를 제외한 결과
    
    for n_epi in tqdm(range(n_episodes)):
        disclosed = True if n_epi%printing_range == printing_range-1 else False
        E = Environment(disclosed)
        
        state = E.get_state()
        epsilon = 1. / (n_epi // 1000 + 1)
        alpha = 1. / (n_epi // 100 + 10)
        done = False
        
        while not done:     
            action = select_action(Q, state, epsilon)
            state2, reward, done = E.step(action, disclosed)
            
            V2 = 0 if done else np.max(Q[state2[0],state2[1],state2[2], :])
            Q[state[0],state[1],state[2], action] = \
                (1-alpha)*Q[state[0],state[1],state[2], action] + \
                alpha*(reward + gamma*V2)
            state = state2
        
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
