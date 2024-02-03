import gym 
import torch
from QLearning import QNet
from PPO import PPO
from tqdm import tqdm


def preprocess_state(state, state2):
    # 210 x 160 x 3 -> 1 x 152 x 88
    s_tensor = torch.from_numpy(state[32:208:2, 4:156, 0:1]).float()
    s2_tensor = torch.from_numpy(state2[32:208:2, 4:156, 0:1]).float()
    return torch.transpose(torch.cat([s_tensor, s2_tensor], dim=2), 0, 2)


def main():
    model = torch.load("project28_Breakout2/QLearning_model.pth")
    #model = torch.load("project28_Breakout2/PPO_model.pth")
    render = False
    n_test = 100
    avg_score = 0.0
    
    if render:
        env = gym.make("ALE/Breakout-v5", render_mode="human")
    else:
        env = gym.make("ALE/Breakout-v5")
    
    for _ in tqdm(range(n_test)):
        env.reset()
        s, _, _, _, _ = env.step(1)
        s2, _, done, _, _ = env.step(1)
        s_tensor = preprocess_state(s, s2)
        score = 0.0

        while not done:
            act = model.get_act(s_tensor)
            s, r, _, _, _ = env.step(act + 1)
            s2, r2, done, _, _ = env.step(1)
            s_tensor = preprocess_state(s, s2)
            score += r + r2
    
        avg_score += score / n_test
    
    print("avg score:", avg_score)
    

if __name__ == "__main__":
    main()
