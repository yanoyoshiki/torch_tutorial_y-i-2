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


#set seeds
def torch_fix_seed(seed=42):
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
BATCHSIZE = 4

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
  def __init__(self, trial, num_layer, mid_units, num_filters,activation):
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

def get_optimizer(trial, model):
  optimizer_names = ['Adam', 'MomentumSGD', 'rmsprop']
  optimizer_name = trial.suggest_categorical('optimizer', optimizer_names)

  weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)

  if optimizer_name == optimizer_names[0]:
    adam_lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
    optimizer = optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)
  elif optimizer_name == optimizer_names[1]:
    momentum_sgd_lr = trial.suggest_loguniform('momentum_sgd_lr', 1e-5, 1e-1)
    optimizer = optim.SGD(model.parameters(), lr=momentum_sgd_lr, momentum=0.9, weight_decay=weight_decay)
  else:
    optimizer = optim.RMSprop(model.parameters())

  return optimizer

def get_activation(trial):
    activation_names = ['ReLU', 'ELU']
    activation_name = trial.suggest_categorical('activation', activation_names)

    if activation_name == activation_names[0]:
        activation = F.relu
    else:
        activation = F.elu

    return activation

def objective(trial):
  EPOCH = 10
  device = "cuda" if torch.cuda.is_available() else "cpu"

  #畳み込み層の数
  num_layer = trial.suggest_int('num_layer', 3, 7)

  #FC層のユニット数
  mid_units = int(trial.suggest_discrete_uniform("mid_units", 100, 300, 10))

  #各畳込み層のフィルタ数
  num_filters = [int(trial.suggest_discrete_uniform("num_filter_"+str(i), 16, 128, 16)) for i in range(num_layer)]

  model = Net(trial, num_layer, mid_units, num_filters).to(device)
  optimizer = get_optimizer(trial, model)

  for step in range(EPOCH):
    train(model, device, train_loader, optimizer)
    error_rate = test(model, device, test_loader)
    print(f'{step}fin | error rate {error_rate}')

  print(f'{trial.number} trial fin')
  return error_rate

TRIAL_SIZE = 50
tensorboard_callback = TensorBoardCallback(f"logs/CIFAR10/{datetime.datetime.now()}/", metric_name="error_rate")
study = optuna.create_study()
study.optimize(objective, n_trials=TRIAL_SIZE, callbacks=[tensorboard_callback])
# ipdb.set_trace()

print(study.best_params)
print(study.best_value)

best_params_result = study.best_params

#output
#{'num_layer': 4, 'mid_units': 140.0, 'num_filter_0': 128.0, 'num_filter_1': 112.0, 'num_filter_2': 112.0, 'num_filter_3': 112.0, 'activation': 'ReLU', 'optimizer': 'MomentumSGD', 'weight_decay': 5.2182135446336915e-08, 'momentum_sgd_lr': 0.0004955865902351846}
#0.2519

#-----------------------------------------------------------------------------------------------------


torch_fix_seed()
def directset_get_optimizer(trial, model,optimizer_name,weight_decay,adam_lr,momentum_sgd_lr):

  if optimizer_name == optimizer_names[0]:
    optimizer = optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)
  elif optimizer_name == optimizer_names[1]:
    optimizer = optim.SGD(model.parameters(), lr=momentum_sgd_lr, momentum=0.9, weight_decay=weight_decay)
  else:
    optimizer = optim.RMSprop(model.parameters())

  return optimizer

def directset_get_activation(trial,activation_name):

    if activation_name == "ReLU":
        activation = F.relu
    else:
        activation = F.elu

    return activation
# retrain using hyperparameter derection on terminal
def directset_hyperparameter(trial, num_layer, mid_units, num_filters,activation_name):
    activation = directset_get_activation(trial,activation_name)
    model = Net(trial, num_layer, mid_units, num_filters,activation).to(device = "cuda" if torch.cuda.is_available() else "cpu")
    optimizer = directset_get_optimizer(trial, model,optimizer_name,weight_decay,adam_lr,momentum_sgd_lr)

    for step in range(EPOCH):
    train(model, device, train_loader, optimizer)
    error_rate = test(model, device, test_loader)
    print(f'{step}fin | error rate {error_rate}')

    print(f'{trial.number + 1} trial fin')
    return error_rate





#set the hyparameter 
# add argparse arguments
import argparse
parser = argparse.ArgumentParser("Welcome to CIFAR10 code")
parser.add_argument("--num_layer", type=int, default=None, help="Seed used for the environment")
parser.add_argument("mid_units", type=int, default=None, help="Seed used for the environment")
parser.add_argument("num_filter", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
args_cli = parser.parse_args()

#seed
#seed =  args_cli.seed

#Indeed, we need to decide the number of num_filter, but the case is based on oputuna firstly. Then you just input each num_layer parameter.
print(study.best_params)
study = optuna.create_study()
study.optimize(objective, n_trials=TRIAL_SIZE, callbacks=[tensorboard_callback])