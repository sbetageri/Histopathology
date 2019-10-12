import data
import torch
import pandas as pd

from models import CrudeModel
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from Dataset import HistoDataset
from tqdm import tqdm

def train_model(model, train_data, val_data, optimizer,
                epochs, loss_fn, device):
    for e in range(epochs):
        print('Training Iteration : ', e + 1)
        running_loss = 0
        model.train()
        for img, label in tqdm(train_data):
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = loss_fn(output, label)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        print('Avg Loss : ', running_loss / (len(train_data) * train_data.batch_size))

        with torch.no_grad():
            print('Validation Iteration : ', e + 1)
            model.eval()
            running_loss = 0
            for img, label in tqdm(val_data):
                img = img.to(device)
                label = label.to(device)
                output = model(img)
                loss = loss_fn(output, label)
                running_loss += loss.item()
        print('Avg Loss : ', running_loss / (len(val_data) * val_data.batch_size))
    return model

def predict(model, test_data):
    outputs = []
    print('Predicting')
    with torch.no_grad():
        model.eval()
        for img in tqmd(test_data):
            output = model(img)
            outputs.append(output.item())
    return outputs


if __name__ == '__main__':
    df = pd.read_csv(data.train_csv)
    train_df, val_df = train_test_split(df, test_size=0.15)
    train_dataset = HistoDataset(train_df, data.train_dir)
    val_dataset = HistoDataset(train_df, data.train_dir)

    test_df = pd.read_csv(data.test_csv)
    test_dataset = HistoDataset(test_df, data.test_dir, flag=HistoDataset.TEST_SET)

    train_loader = DataLoader(train_dataset, batch_size=4)
    val_loader = DataLoader(val_dataset, batch_size=4)

    model = CrudeModel.CrudeModel()

    model = train_model()

    model.save(model.state_dict(), 'models/model.pt')

    predictions = predict(model, test_dataset)

    test_df['label'] = predictions

