import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import numpy as np


# QLearning algorithm
# hyper-parameters
n_episodes = 100000
interval = 10000
gamma = 0.99


def preprocess(state):
    usable_ace = 1 if state[2] else 0
    return [state[0], state[1], usable_ace]

def select_action(Q, state, epsilon):
    t = random.random()
    if t < epsilon:
        return random.randint(0, 1)
    else:
        return np.argmax(np.array(Q[state[0], state[1], state[2], :]))


def main():
    Q = np.zeros([22, 12, 2, 2], dtype=float)
    env = gym.make("Blackjack-v1")
    win_cnt = 0
    win_lose_cnt = 0
    result_winrate = []

    for epi in tqdm(range(n_episodes)):
        state = env.reset()[0]
        state = preprocess(state)
        done = False
        score = 0
        epsilon = 1.0 / (epi // 1000 + 1)
        alpha = 1.0 / (epi // 100 + 10)

        while not done:
            act = select_action(Q, state, epsilon)
            state2, reward, done, _, _ = env.step(act)
            state2 = preprocess(state2)
            score += reward

            # train
            V_prime = 0 if done else np.max(Q[state2[0], state2[1], state2[2], :])
            Q[state[0], state[1], state[2], act] = (1 - alpha) * Q[
                state[0], state[1], state[2], act
            ] + alpha * (reward + gamma * V_prime)

            state = state2

        if score < -0.9:
            win_lose_cnt += 1
        elif score > 0.9:
            win_cnt += 1
            win_lose_cnt += 1

        if epi % interval == interval - 1:
            result_winrate.append(win_cnt / win_lose_cnt)
            win_cnt = 0
            win_lose_cnt = 0

    plt.plot(range(len(result_winrate)), result_winrate, color="red")
    plt.show()


if __name__ == "__main__":
    main()
