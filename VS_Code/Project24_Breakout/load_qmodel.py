import gym 
import torch
from QLearning import QNet


def preprocess_state(state, state2):
    # 210 x 160 x 3 -> 1 x 152 x 92
    s_tensor = torch.from_numpy(state[26:210:2, 4:156, 0:1]).float()
    s2_tensor = torch.from_numpy(state2[26:210:2, 4:156, 0:1]).float()
    return torch.transpose(torch.cat([s_tensor, s2_tensor]), 0, 2)


def main():
    Q = torch.load("project24_Breakout/QLearning_model.pth")
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    env.reset()
    s, _, _, _, _ = env.step(1)
    s2, _, done, _, _ = env.step(1)
    s_tensor = preprocess_state(s, s2)
    score = 0.0
    
    while not done:
        act = Q.get_act(s_tensor, 0.0)
        s, r, _, _, _ = env.step(act + 1)
        s2, r2, done, _, _ = env.step(1)
        s_tensor = preprocess_state(s, s2)
        score += r + r2
    
    print("score:", score)
    

if __name__ == "__main__":
    main()
