import time

import requests
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, Subset


class Net(nn.Module):
    def __init__(self, num_sensors, num_hidden_units, num_layers, t, dropout):
        super().__init__()
        self.input_size = num_sensors  # number of features
        self.hidden_units = num_hidden_units
        self.num_layers = num_layers
        self.t = t

        # GRU instead of LSTM
        self.gru = nn.GRU(
            input_size=num_sensors,  # the number of expected features in the input x
            hidden_size=num_hidden_units,  # the number of features in the hidden state h
            batch_first=True,
            bidirectional=False,
            dropout=dropout,
            num_layers=self.num_layers  # number of layers that have some hidden units
        )
        self.fc = nn.Linear(num_hidden_units, t)

    def forward(self, x):
        output, hn = self.gru(x)  # GRU returns (output, hidden_state)
        output = output[:, -1, :]  # take the output of the last time step
        output = self.fc(output)  # map to the desired output shape (batch_size, t)
        return output


class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, t, sequence_length=5):
        self.features = features
        self.target = target
        self.t = t
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0] - self.t

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1)]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1)]
            x = torch.cat((padding, x), 0)
        return x, self.y[i + 1: i + self.t + 1]


def mse(prediction, real_value):
    MSE = torch.square(torch.subtract(real_value, prediction)).mean()
    return MSE


def test(model, test_loader, device):
    model.to(device)
    print("TESTING")
    """Validate the network on the entire test set."""
    loss = 0
    model.eval()
    with torch.no_grad():  # do not calculate the gradient
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            result = model(X)
            loss += mse(result, y)
    loss = (loss / len(test_loader)).item()
    print(loss)
    print("LOSS")
    return loss


def train(net, trainloader, epochs, device):
    """Train the network on the training set."""
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)  # momentum=0.9
    net.train()
    for epoch in range(epochs):
        sum_loss = 0
        for i, (X, y) in enumerate(trainloader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()  # sets gradients back to zero: when I start the training loop: zero out the gradients so that I can perform this tracking correctly
            prediction_result = net(X)
            loss = mse(prediction_result, y)
            sum_loss += loss.item()
            loss.backward()  # gradients computed
            optimizer.step()  # to proceed gradient descent


def create_dataloaders(data, column, batch_size, sequence_length, state):
    train_size = 4536
    train_indices = list(range(0 + state, train_size + state))
    val_indices = list(range(train_size + state, 1512 + train_size + state))

    dataset = SequenceDataset(dataframe=data, target=column, features=[column], t=sequence_length,
                              sequence_length=sequence_length)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader


def load_data(id, batch_size, sequence_length, state):
    id = int(id[1:])
    file_path = r"../data/METR-LA.csv"
    data = pd.read_csv(file_path)
    data = data.rename(columns={'Unnamed: 0': 'timestamp'})
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp')
    column = data.columns[id]
    scaler = MinMaxScaler()
    data[[column]] = scaler.fit_transform(data[[column]])
    train_dataloader, val_dataloader = create_dataloaders(data, column, batch_size=batch_size,
                                                          sequence_length=sequence_length, state=state * 24)
    return train_dataloader, val_dataloader


# list(val_dataloader)[-1]
def inference(X, y, model, DEVICE, number_of_requests):
    print("INFERENCE")
    model.to(DEVICE)
    start = time.time()
    for epoch in range(number_of_requests):
        model.eval()
        with torch.no_grad():  # do not calculate the gradient
            X, y = X.to(DEVICE), y.to(DEVICE)
            prediction_result = model(X)
    end = time.time()
    total_time = end - start
    print(total_time)
    return total_time


def send_inference_requests_to_server(X, y, number_of_requests, port):
    print("SEND INFERENCE TO SERVER")
    data = {"X": X.tolist(), "y": y.tolist(), "number_of_requests": number_of_requests}
    response = requests.post(f'http://localhost:{port}/inference', json=data)
    return response
