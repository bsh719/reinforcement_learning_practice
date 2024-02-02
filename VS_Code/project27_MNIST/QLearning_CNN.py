import torch
import torch.nn as nn 
import torch.nn.functional as F 
import random 
from tqdm import tqdm 

import torchvision 
import torchvision.transforms as transforms

train_dataset = torchvision.datasets.MNIST("datasets", download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_dataset = torchvision.datasets.MNIST("datasets", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))

# hyperparameters
learning_rate = 0.001
#gamma = 1.0
n_epochs = 3
batch_size = 100
interval = 300

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.CV = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(4, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        self.FC = nn.Sequential(
            nn.Linear(16*6*6, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        x = self.CV(x)
        x = self.FC(x)
        return x

    def get_act(self, state, epsilon):
        p = random.random()
        if p < epsilon:
            return random.randint(0, 9)

        out = self(state)
        return out.argmax().item()


def main():
    Q = QNet()
    optimizer = torch.optim.Adam(Q.parameters(), lr=learning_rate)
    epi = 0
    avg_score = 0.0  # 100점 만점
    
    for epoch in range(n_epochs):
        for imgs, lbls in tqdm(train_loader):
            epi += 1
            epsilon = 10. / (10 + epi)
            acts = []
            for i in range(batch_size):
                a = Q.get_act(imgs[i], epsilon)
                acts.append(a)
            
            # train
            optimizer.zero_grad()
            outs = Q(imgs)
            score = 0.0  # 100점 만점
            total_loss = 0.0
            for i in range(batch_size):
                a = acts[i]
                r = 1.0 if a == lbls[i].item() else 0.0
                score += r
                target = torch.tensor(r)
                loss = F.mse_loss(outs[i, a], target.detach())
                total_loss = loss + total_loss
            
            total_loss.backward()
            optimizer.step()
            avg_score += score / interval
            
            if epi % interval == 0:
                print(f"\nepisode {epi}, score: {score}, avg score: {avg_score:.4f}")
                avg_score = 0.0
    
    # test
    test_score = 0.0
    for imgs, lbls in tqdm(test_loader):
        acts = []
        for i in range(batch_size):
            a = Q.get_act(imgs[i], -1.0)
            acts.append(a)
        
        outs = Q(imgs)
        for i in range(batch_size):
            a = acts[i]
            r = 1.0 if a == lbls[i].item() else 0.0
            test_score += r / 100
    
    print(f"\ntest score: {test_score:.4f}")
    # FashionMNIST: 86점대, MNIST: 96점대

if __name__ == "__main__":
    main()
