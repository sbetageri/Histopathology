import data
import torch
import pandas as pd

from models import AlexNet
from models import ResNet
from models import ResidualBlocks
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from Dataset import HistoDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def train_model(model, train_data, val_data, optimizer, scheduler,
                epochs, loss_fn, device, writer):
    epoch_val_loss = []
    for e in range(epochs):
        print('Training Iteration : ', e + 1)
        running_loss = 0
        running_acc = 0
        model = model.to(device)
        model.train()
        for i, (img, label) in tqdm(enumerate(train_data)):
            img = img.to(device)
            label = label.view(label.size(0), 1)
            label = label.to(device)

            optimizer.zero_grad()

            output = model(img)
            loss = loss_fn(output, label)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predictions = torch.round(torch.sigmoid(output))
            # print('Prediction : ', predictions)
            # print('Label : ', label)
            # print('Accuracy : ', (predictions == label).sum().item())
            # assert False
            running_acc += (predictions == label).sum().item()
            if (i % 1000) == 999:
                writer.add_scalar('Loss/Train', running_loss / (1000 * train_data.batch_size))
                writer.add_scalar('Acc/Train', running_acc / (1000 * train_data.batch_size))


                running_loss = 0
                running_acc = 0

        val_loss = 0
        with torch.no_grad():
            print('Validation Iteration : ', e + 1)
            model.eval()
            running_loss = 0
            running_acc = 0
            for i, (img, label) in tqdm(enumerate(val_data)):
                img = img.to(device)
                label = label.view(label.size(0), 1)
                label = label.to(device)

                output = model(img)
                loss = loss_fn(output, label)
                running_loss += loss.item()
                val_loss += loss.item()

                predictions = torch.round(torch.sigmoid(output))
                running_acc += (predictions == label).sum().item()

                if (i % 1000) == 999:
                    writer.add_scalar('Loss/Val', running_loss / (1000 * val_data.batch_size))
                    writer.add_scalar('Acc/Val', running_acc / (1000 * val_data.batch_size))

                    running_loss = 0
                    running_acc = 0
        val_loss /= (len(val_data) * val_data.batch_size)
        scheduler.step(val_loss)
        if (len(epoch_val_loss) > 2 and
                within_epsilon(val_loss, epoch_val_loss[-1], 1e-3) and
                within_epsilon(epoch_val_loss[-1], epoch_val_loss[-2], 1e-3)):
            print('Early Stopping')
            return model
        else:
            epoch_val_loss.append(val_loss)
    return model

def within_epsilon(a, b, epsilon):
    diff = abs(a - b)
    if diff <= epsilon:
        return True
    return False

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

    writer = SummaryWriter('runs/histo_run_ResNet_norm_lr_1e-4')

    df = pd.read_csv(data.train_csv)
    train_df, val_df = train_test_split(df, test_size=0.15)
    train_dataset = HistoDataset(train_df, data.train_dir)
    val_dataset = HistoDataset(val_df, data.train_dir)

    test_df = pd.read_csv(data.test_csv)
    test_dataset = HistoDataset(test_df, data.test_dir, flag=HistoDataset.TEST_SET)

    train_loader = DataLoader(train_dataset, batch_size=4)
    val_loader = DataLoader(val_dataset, batch_size=4)

    model = ResNet.ResNet(block=ResidualBlocks.ResBlock)

    img, label = next(iter(train_loader))
    writer.add_graph(model, img)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = train_model(model, train_loader, val_loader, optimizer, scheduler, 20, loss_fn, device, writer)

    torch.save(model.state_dict(), 'models/ResNet_Norm.pt')

    predictions = predict(model, test_dataset, device)

    test_df['label'] = predictions

    test_df.to_csv('predictions.csv', index=False)
