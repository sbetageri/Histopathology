import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, pool=None):
        super(ResBlock, self).__init__()
        if pool is not None:
            self.conv1 = nn.Conv2d(in_dim, out_dim, 3, padding=1, stride=2)
            self.pool = nn.MaxPool2d(2)
        else:
            self.conv1 = nn.Conv2d(in_dim, out_dim, 3, padding=1)
            self.pool = None
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.bn2 = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        if self.pool is not None:
            identity = self.pool(x)
        else:
            identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        out = torch.cat((identity, x), dim=1)
        out = F.relu(out)
        return out

class PreAct_ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, pool=None):
        super(PreAct_ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_dim)

        if pool is not None:
            self.conv1 = nn.Conv2d(in_dim, out_dim, 3, padding=1, stride=2)
            self.pool = nn.MaxPool2d(2)
        else:
            self.conv1 = nn.Conv2d(in_dim, out_dim, 3, padding=1)
            self.pool = None

        self.bn2 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, padding=1)

    def forward(self, x):
        if self.pool is not None:
            identity = self.pool(x)
        else:
            identity = x
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)

        x = torch.cat((identity, x), dim=1)
        return x
