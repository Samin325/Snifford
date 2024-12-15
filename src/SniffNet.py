import torch
import torch.nn as nn
import torch.optim as optim

class IDSModel(nn.Module):
    def __init__(self):
        super(IDSModel, self).__init__()
        self.fc1 = nn.Linear(78, 50) # 3 fully connected layers
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 1) # binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
