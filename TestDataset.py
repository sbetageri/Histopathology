import data
import pandas as pd

from Dataset import HistoDataset
from torch.utils.data import DataLoader

## Visually checking for correct dimensions
if __name__ == '__main__':
    df = pd.read_csv(data.train_csv)
    train_ds = HistoDataset(df, data.train_dir)
    for img, label in train_ds:
        print('Train Image size : ', img.size())
        print('Train Label : ', label)
        break

    train_loader = DataLoader(train_ds, batch_size=4)
    for imgs, labels in train_loader:
        print('Train batch images : ', imgs.size())
        print('Train batch labels : ', labels.size())
        print('Labels : ', labels)
        break

    test_df = pd.read_csv(data.test_csv)
    test_ds = HistoDataset(test_df, data.test_dir, flag=HistoDataset.TEST_SET)
    for img in test_ds:
        print('Test Image Size : ', img.size())
        break

    test_loader = DataLoader(test_ds, batch_size=4)
    for imgs in test_loader:
        print('Test batch images : ', imgs.size())
        break
