import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.fc1 = nn.Linear(78, 50)  # 3 fully connected layers
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 1)  # binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        hidden_size = 64

        # takes about 5 minutes per epoch, accuracy of 94.86%
        self.lstm = nn.LSTM(
            input_size=78,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True  # Bi-LSTM
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, 1),  # hidden_size * 2 for bidirectional LSTM
        )

    def forward(self, x):
        rescaled = x.unsqueeze(1)
        lstm_out, _ = self.lstm(rescaled)  # (batch_size, seq_length, hidden_size*2)
        last_hidden_state = lstm_out[:, -1, :]  # (batch_size, hidden_size*2)
        out = self.fc(last_hidden_state)  # (batch_size, num_classes)
        return out
