import torch
import torch.nn as nn
import torch.optim as optim

class IDSModel(nn.Module):
    def __init__(self):
        super(IDSModel, self).__init__()
        self.fc1 = nn.Linear(30, 64)  # change first number according to the number of features (eg. if 30 features, number should be 30)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)   # binary classification for the time being (benign/malicious)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
