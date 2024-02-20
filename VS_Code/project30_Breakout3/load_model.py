import gym 
import torch
from Qlearning import QNet
from tqdm import tqdm
from torchvision import transforms


def preprocess_state(state, state2):
    # 210 x 160 x 3 -> 1 x 144 x 168 -> 3 x 64 x 64
    t1 = torch.from_numpy(state[32:200, 8:152, 0:1]) / 256
    t2 = torch.from_numpy(state2[32:200, 8:152, 0:1]) / 256
    state = torch.transpose(torch.cat([t1, t2], dim=2), 0, 2)
    return transforms.Resize([64, 64])(state)


def main():
    model = torch.load("project30_Breakout3/Qmodel.pth")
    render = False
    n_test = 200
    avg_score = 0.0
    
    if render:
        env = gym.make("ALE/Breakout-v5", render_mode="human")
    else:
        env = gym.make("ALE/Breakout-v5")
    
    for _ in tqdm(range(n_test)):
        env.reset()
        s1, _, _, _, _ = env.step(1)
        s2, _, done, _, _ = env.step(1)
        s_tensor = preprocess_state(s1, s2)
        score = 0.0

        while not done:
            act = model.get_act(s_tensor)
            s1, r, _, _, _ = env.step(act + 1)
            s2, r2, done, _, _ = env.step(1)
            s_tensor = preprocess_state(s1, s2)
            score += r + r2
    
        avg_score += score / n_test
    
    print("avg score:", avg_score)
    

if __name__ == "__main__":
    main()
