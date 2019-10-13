import data
import torch
import pandas as pd

from models import AlexNet
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Dataset import HistoDataset
from tqdm import tqdm

def train_model(model, train_data, val_data, optimizer,
                epochs, loss_fn, device):
    for e in range(epochs):
        print('Training Iteration : ', e + 1)
        running_loss = 0
        running_acc = 0
        model = model.to(device)
        model.train()
        for img, label in tqdm(train_data):
            img = img.to(device)
            label = label.view(label.size(0), 1)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predictions = torch.ceil(output.detach())
            running_acc += accuracy_score(label, predictions)

        print('Avg Loss : ', running_loss / (len(train_data) * train_data.batch_size))
        print('Avg Acc: ', running_acc / (len(train_data) * train_data.batch_size))

        with torch.no_grad():
            print('Validation Iteration : ', e + 1)
            model.eval()
            running_loss = 0
            for img, label in tqdm(val_data):
                img = img.to(device)
                label = label.view(label.size(0), 1)
                label = label.to(device)
                output = model(img)
                loss = loss_fn(output, label)
                predictions = torch.ceil(output.detach())
                running_acc += accuracy_score(label, predictions)
                running_loss += loss.item()
        print('Avg Loss : ', running_loss / (len(val_data) * val_data.batch_size))
        print('Avg Acc: ', running_acc / (len(train_data) * train_data.batch_size))
    return model

def predict(model, test_data, device):
    outputs = []
    print('Predicting')
    with torch.no_grad():
        model.eval()
        for img in tqdm(test_data):
            img = img.view(1, *img.size())
            img = img.to(device)
            output = model(img)
            output = torch.sigmoid(output)
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

    model = AlexNet.AlexNet()

    optimizer = torch.optim.Adadelta(model.parameters())
    loss_fn = torch.nn.BCEWithLogitsLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = train_model(model, train_loader, val_loader, optimizer, 2, loss_fn, device)

    torch.save(model.state_dict(), 'models/model.pt')

    predictions = predict(model, test_dataset, device)

    test_df['label'] = predictions

    test_df.to_csv('predictions.csv', index=False)
