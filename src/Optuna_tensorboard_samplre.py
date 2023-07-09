import optuna
from optuna.integration.tensorboard import TensorBoardCallback
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

# データセットの準備
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)


# モデルの定義
class Net(nn.Module):
    def __init__(self, num_units, dropout_rate):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, num_units)
        self.fc2 = nn.Linear(num_units, 10)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_test_model(num_units, dropout_rate, optimizer):
    model = Net(num_units, dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    
    if optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    else:
        optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(1):  # Run with 1 epoch to speed things up for demo purposes
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total


def objective(trial):
    num_units = trial.suggest_int("NUM_UNITS", 16, 64)
    dropout_rate = trial.suggest_float("DROPOUT_RATE", 0.05, 0.15)
    optimizer = trial.suggest_categorical("OPTIMIZER", ["sgd", "adam"])

    accuracy = train_test_model(num_units, dropout_rate, optimizer)
    return accuracy


tensorboard_callback = TensorBoardCallback("logs/", metric_name="accuracy")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, timeout=600, callbacks=[tensorboard_callback])
