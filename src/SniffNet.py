import torch
import torch.nn as nn
import torch.optim as optim

class IDSModel(nn.Module):
    def __init__(self):
        super(IDSModel, self).__init__()
        self.fc1 = nn.Linear(78, 170)  # first number should math # of features (ie. if features are dropped in data_loader.py, update number here)
        self.fc2 = nn.Linear(170, 50)
        self.fc3 = nn.Linear(50, 1)   # binary classification for the time being (benign/malicious)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
