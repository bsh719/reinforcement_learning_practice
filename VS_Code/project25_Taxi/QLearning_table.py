import gym
import numpy as np
import random

# hyperparameters
alpha = 0.01
gamma = 0.99
n_episodes = 10000
interval = 500


def decode(state):
    taxi_row = state // 100
    taxi_col = (state % 100) // 20
    passenger_location = (state % 20) // 4
    destination = state % 4
    return taxi_row, taxi_col, passenger_location, destination


def get_act(Q, s, epsilon):
    p = random.random()
    if p < epsilon:
        return random.randint(0, 5)

    return np.argmax(Q[s[0], s[1], s[2], s[3], :])


def main():
    env1 = gym.make("Taxi-v3")
    env2 = gym.make("Taxi-v3", render_mode="human")
    Q = np.zeros([5, 5, 5, 4, 6])
    avg_score = 0.0

    for epi in range(n_episodes):
        E = env2 if epi == n_episodes - 1 else env1
        epsilon = 1.0 / (1 + epi / 20)
        state = E.reset()[0]
        s = decode(state)
        done = False
        score = 0.0

        for _ in range(1000):
            a = get_act(Q, s, epsilon)
            state2, r, done, _, _ = E.step(a)
            score += r
            s2 = decode(state2)

            # train
            target = r + gamma * np.max(Q[s2[0], s2[1], s2[2], s2[3], :])
            Q[s[0], s[1], s[2], s[3], a] = (1 - alpha) * Q[
                s[0], s[1], s[2], s[3], a
            ] + alpha * target

            s = s2
            if done: break

        avg_score += score / interval

        if epi % interval == interval - 1:
            print(f"episode: {epi+2-interval} ~ {epi+1}, average score: {avg_score}")
            avg_score = 0.0


if __name__ == "__main__":
    main()
