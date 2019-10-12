import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.conv2 = nn.Conv2d(12, 32, 3)
        self.conv3 = nn.Conv2d(32, 48, 3)
        self.conv4 = nn.Conv2d(48, 48, 3)
        self.conv5 = nn.Conv2d(48, 32, 3)

        self.linear1 = nn.Linear(2048, 256)
        self.linear2 = nn.Linear(256, 1)

        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.max_pool(x)

        x = x.view((x.size(0), -1))

        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

