import torch
import Dataset
import data
import pandas as pd

from models.ResidualBlocks import ResBlock, PreAct_ResBlock
from models.ResNet import ResNet

if __name__ == '__main__':
    df = pd.read_csv(data.train_csv)
    ds = Dataset.HistoDataset(df, data.train_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resBlock = ResBlock(3, 64)
    resBlock = resBlock.to(device)

    preActResBlock = PreAct_ResBlock(3, 64)
    preActResBlock = preActResBlock.to(device)

    poolResBlock = ResBlock(3, 64, pool=True)
    poolResBlock = poolResBlock.to(device)

    poolPreActResBlock = PreAct_ResBlock(3, 64, pool=True)
    poolPreActResBlock = poolPreActResBlock.to(device)

    resNet = ResNet()
    resNet = resNet.to(device)

    for img, label in ds:
        img = img.view(1, *img.size())
        op = resBlock(img)
        res_out_size = op.size()
        assert res_out_size == (1, 67, 96, 96)
        print('Residual Block tests passed')

        op = preActResBlock(img)
        res_out_size = op.size()
        assert res_out_size == (1, 67, 96, 96)
        print('Pre Activation Residual Block tests passed')

        op = poolResBlock(img)
        res_out_size = op.size()
        assert res_out_size == (1, 67, 48, 48)
        print('Residual Block Pooling tests passed')

        op = poolPreActResBlock(img)
        res_out_size = op.size()
        assert res_out_size == (1, 67, 48, 48)
        print('Pre Activation Residual Block Pooling tests passed')

        op = resNet(img)
        res_out_size = op.size()
        assert res_out_size == (1, 1)
        print('ResNet test passed')

        break




