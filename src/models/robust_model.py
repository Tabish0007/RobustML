import torch
import torch.nn as nn
import torch.nn.functional as F

class RobustCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(RobustCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64*6*6, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64*6*6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
