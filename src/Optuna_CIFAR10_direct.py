# NumPy、Matplotlib、PyTorchをインポートする
import datetime
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from optuna.integration.tensorboard import TensorBoardCallback

import optuna
optuna.logging.disable_default_handler()

import random

# set the hyparameter 
# add argparse arguments
import argparse
parser = argparse.ArgumentParser("Welcome to CIFAR10 code")
parser.add_argument("--patch_size", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--EPOCH", type=int, default=None, help="EPOCH used for training")
parser.add_argument("--num_layer", type=int, default=None, help="numlayer used for model setteing")
parser.add_argument("--mid_units", type=int, default=None, help="mid_units used for model setteing")
parser.add_argument("--num_filter", nargs='+', type=int, default=None, help="Seed used for the environment")
parser.add_argument("--activation_name", type=str, default=None, help="Seed used for the environment")
parser.add_argument("--optimizer_name", type=str, default=None, help="Seed used for the environment")
parser.add_argument("--weight_decay", type=float, default=None, help="Seed used for the environment")
parser.add_argument("--adam_lr", type=float, default=None, help="Seed used for the environment")
parser.add_argument("--momentum_sgd_lr", type=float, default=None, help="Seed used for the environment")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
args_cli = parser.parse_args()

#set seeds
def torch_fix_seed(seed=args_cli.seed):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
torch_fix_seed()


#入力画像の高さと幅，畳み込み層のカーネルサイズ
in_height = 32
in_width = 32
kernel = 5
BATCHSIZE = args_cli.patch_size
# BATCHSIZE = 4

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCHSIZE, shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=BATCHSIZE, shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



class Net(nn.Module):
  def __init__(self, num_layer, mid_units, num_filters,activation):
    super(Net, self).__init__()
    self.activation = activation
    #第1層
    self.convs = nn.ModuleList([nn.Conv2d(in_channels=3, out_channels=num_filters[0], kernel_size=5)])
    self.out_height = in_height - kernel +1
    self.out_width = in_width - kernel +1
    #第2層以降
    for i in range(1, num_layer):
      self.convs.append(nn.Conv2d(in_channels=num_filters[i-1], out_channels=num_filters[i], kernel_size=5))
      self.out_height = self.out_height - kernel + 1
      self.out_width = self.out_width - kernel +1
    #pooling層
    self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    self.out_height = int(self.out_height / 2)
    self.out_width = int(self.out_width / 2)
    #線形層
    self.out_feature = self.out_height * self.out_width * num_filters[num_layer - 1]
    self.fc1 = nn.Linear(in_features=self.out_feature, out_features=mid_units)
    self.fc2 = nn.Linear(in_features=mid_units, out_features=10)

  def forward(self, x):
    for i, l in enumerate(self.convs):
      x = l(x)
      x = self.activation(x)
    x = self.pool(x)
    x = x.view(-1, self.out_feature)
    x = self.fc1(x)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 1 - correct / len(test_loader.dataset)


torch_fix_seed()
def directset_get_optimizer(model,optimizer_name,weight_decay,adam_lr,momentum_sgd_lr):
  optimizer_names = ['Adam', 'MomentumSGD', 'rmsprop']
  if optimizer_name == optimizer_names[0]:
    optimizer = optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)
  elif optimizer_name == optimizer_names[1]:
    optimizer = optim.SGD(model.parameters(), lr=momentum_sgd_lr, momentum=0.9, weight_decay=weight_decay)
  else:
    optimizer = optim.RMSprop(model.parameters())

  return optimizer

def directset_get_activation(activation_name):

    if activation_name == "ReLU":
        activation = F.relu
    else:
        activation = F.elu

    return activation
# retrain using hyperparameter derection on terminal
def directset_hyperparameter(num_layer, mid_units, num_filters,activation_name,optimizer_name,weight_decay,adam_lr,momentum_sgd_lr,EPOCH):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    activation = directset_get_activation(activation_name)
    model = Net(num_layer, mid_units, num_filters,activation).to(device)
    optimizer = directset_get_optimizer(model,optimizer_name,weight_decay,adam_lr,momentum_sgd_lr)

    for step in range(EPOCH):
      train(model, device, train_loader, optimizer)
      error_rate = test(model, device, test_loader)
      print(f'{step}fin | error rate {error_rate}')

      # return error_rate





#Execution training
directset_hyperparameter(args_cli.num_layer,
                        args_cli.mid_units,
                        args_cli.num_filter,
                        args_cli.activation_name,
                        args_cli.optimizer_name,
                        args_cli.weight_decay,
                        args_cli.adam_lr,
                        args_cli.momentum_sgd_lr,
                        args_cli.EPOCH)


# python Optuna_CIFAR10_direct.py --num_layer 4 --mid_units 140 --num_filter 128 112 112 112 --activation_name "ReLU" --optimizer_name "MomentumSGD" --weight_decay 5.2182135446336915e-08 --adam_lr 0.00021312 --momentum_sgd_lr 0.0004955865902351846 --seed 42