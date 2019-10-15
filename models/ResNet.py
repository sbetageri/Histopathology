import torch.nn as nn
from models.ResidualBlocks import ResBlock, PreAct_ResBlock

class ResNet(nn.Module):
    def __init__(self, block=ResBlock):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        in_dim = 64

        self.res_block1 = block(in_dim, 64)
        in_dim += 64
        self.res_block2 = block(in_dim, 64)
        in_dim += 64

        self.res_block3 = block(in_dim, 128, pool=True)
        in_dim += 128
        self.res_block4 = block(in_dim, 128)
        in_dim += 128

        self.res_block5 = block(in_dim, 256, pool=True)
        in_dim += 256
        self.res_block6 = block(in_dim, 256)
        in_dim += 256

        self.res_block7 = block(in_dim, 512, pool=True)
        in_dim += 512

        self.res_block8 = block(in_dim, 512)

        self.avg_pool = nn.AvgPool2d(2)
        self.fc = nn.Linear(71424, 1)

    def forward(self, x):
        x = self.conv1(x)

        x = self.res_block1(x)
        x = self.res_block2(x)

        x = self.res_block3(x)
        x = self.res_block4(x)

        x = self.res_block5(x)
        x = self.res_block6(x)

        x = self.res_block7(x)
        x = self.res_block8(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


