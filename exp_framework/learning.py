import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # TODO: Should generalize this to work for other datasets
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 output classes for MNIST digits

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def predict(self, x):
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
