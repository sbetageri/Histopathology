import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, 5)
        self.conv2 = nn.Conv2d(48, 128, 3)
        self.conv3 = nn.Conv2d(128, 192, 3)
        self.conv4 = nn.Conv2d(192, 192, 3)
        self.conv5 = nn.Conv2d(192, 128, 3)

        self.linear1 = nn.Linear(8192, 64)
        self.linear2 = nn.Linear(64, 1)

        self.max_pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.45)

    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = self.max_pool(x)
        x = F.selu(self.conv2(x))
        x = self.max_pool(x)
        x = F.selu(self.conv3(x))
        x = F.selu(self.conv4(x))
        x = F.selu(self.conv5(x))
        x = self.max_pool(x)

        x = x.view((x.size(0), -1))

        x = F.selu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)

        return x

