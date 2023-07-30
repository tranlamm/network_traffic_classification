from collections import OrderedDict
from typing import List, Tuple

import sys
if len(sys.argv) < 2:
    print("Not enough argument!")
    exit(0)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader, random_split
from pathlib import Path
from time import time
import warnings

import flwr as fl
from flwr.common import Metrics

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))

# Config
byte_number = "256"

PACKET_NUM = 20
NUM_FEATURE = int(byte_number)
NUM_CLASSES = 5

client_lr = 0.0003
BATCH_SIZE = 32
NUM_CLIENTS = 8
CLIENT_INDEX = int(sys.argv[1])

# Load datasets
train_dir = './GQUIC_small/Train/GQUIC_train_' + byte_number + '.feather'
test_dir = './GQUIC_small/Test/GQUIC_test_' + byte_number + '.feather'
data = pd.read_feather(train_dir)
test = pd.read_feather(test_dir)

def most_frequent(List):
    return max(set(List), key=List.count)

def load_data_set(data):
    flows = data.groupby('flow_id')['Label'].apply(list).to_dict()
    true_label = []
    for flow in flows:
        true_label.append(most_frequent(flows[flow]))

    true_label = np.array(true_label)
    true_dataset = data.drop(['Label', 'flow_id'], axis=1).to_numpy()/255
    true_dataset = true_dataset.reshape(-1, PACKET_NUM, NUM_FEATURE)
    true_dataset = np.expand_dims(true_dataset, -1)

    true_set = []
    for i in range(true_dataset.shape[0]):
        true_set.append(true_dataset[i].transpose(2, 0, 1))

    true_set = np.array(true_set)
    return true_set, true_label

x_train, y_train = load_data_set(data)
x_test, y_test = load_data_set(test)

def crop(x_train, y_train):
    length = x_train.shape[0] // NUM_CLIENTS
    x_train = x_train[(CLIENT_INDEX - 1)*length:CLIENT_INDEX*length]
    y_train = y_train[(CLIENT_INDEX - 1)*length:CLIENT_INDEX*length]
    return x_train, y_train

x_train, y_train = crop(x_train, y_train)
x_test, y_test = crop(x_test, y_test)

def to_tensor(x_train, y_train):
    tensor_x = torch.Tensor(x_train) # transform to torch tensor
    tensor_y = torch.Tensor(y_train)
    tensor_y = tensor_y.type(torch.LongTensor)

    my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    return my_dataset

train_set = to_tensor(x_train, y_train)
test_set = to_tensor(x_test, y_test)

def load_datasets(train_set, test_set):
    trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_set, batch_size=BATCH_SIZE) 
    return trainloader, testloader

trainloader, testloader = load_datasets(train_set, test_set)

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (5, 5), padding=(2, 2))
        self.conv2 = nn.Conv2d(32, 32, (5, 5), padding=(2, 2))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.dropout1 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(64, 64, (3, 3), padding=(1, 1))
        self.pool2 = nn.MaxPool2d((2, 2), (2, 2))
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 5 * 64, 256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, NUM_CLASSES)
        self.softmax = nn.Softmax(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
net = Net().to(DEVICE)

def train(net, trainloader, train_time, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=client_lr)
    net.train()
    start = time()
    data_iter = iter(trainloader)
    while True:
        if time() - start > train_time:
            break
        try:
            images, labels = next(data_iter)
        except StopIteration:
        # StopIteration is thrown if dataset ends
        # Reinitialize data_iter
            data_iter = iter(trainloader)
            images, labels = next(data_iter)

        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print(config)
        self.set_parameters(parameters)
        train(net, trainloader, train_time=config["train_time"])
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}
    
# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:9090", client=FlowerClient())
