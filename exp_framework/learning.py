import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_dim, output_dim, width=64):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, width)
        self.fc2 = nn.Linear(width, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim)  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def predict(self, x):
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
