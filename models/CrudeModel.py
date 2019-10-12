import torch
import torch.nn as nn
import torch.nn.functional as F

class CrudeModel(nn.Module):
    def __init__(self):
        super(CrudeModel, self).__init__()

    def forward(self, x):
        op = torch.ones((x.size(0), 1))
        return op
