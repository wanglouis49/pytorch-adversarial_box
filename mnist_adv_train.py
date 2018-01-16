"""
Adversarially train LeNet-5
"""

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test

from models import LeNet5


# Hyper-parameters
param = {
    'batch_size': 128,
    'test_batch_size': 100,
    'num_epochs': 15,
    'delay': 10,
    'learning_rate': 1e-3,
}


# Data loaders
train_dataset = datasets.MNIST(root='../data/',train=True, download=True, 
    transform=transforms.ToTensor())
loader_train = torch.utils.data.DataLoader(train_dataset, 
    batch_size=param['batch_size'], shuffle=True)

test_dataset = datasets.MNIST(root='../data/', train=False, download=True, 
    transform=transforms.ToTensor())
loader_test = torch.utils.data.DataLoader(test_dataset, 
    batch_size=param['test_batch_size'], shuffle=True)


# Setup the model
net = LeNet5()

if torch.cuda.is_available():
    print('CUDA ensabled.')
    net.cuda()
net.train()

# Adversarial training setup
adversary = FGSMAttack(epsilon=0.3)
#adversary = LinfPGDAttack()

# Train the model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(net.parameters(), lr=param['learning_rate'])

for epoch in range(param['num_epochs']):

    print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))

    for t, (x, y) in enumerate(loader_train):

        x_var, y_var = to_var(x), to_var(y.long())
        loss = criterion(net(x_var), y_var)

        # adversarial training
        if epoch+1 > param['delay']:
            # use predicted label to prevent label leaking
            y_pred = pred_batch(x, net)
            x_adv = adv_train(x, y_pred, net, criterion, adversary)
            x_adv_var = to_var(x_adv)
            loss_adv = criterion(net(x_adv_var), y_var)
            loss = (loss + loss_adv) / 2

        if (t + 1) % 100 == 0:
            print('t = %d, loss = %.8f' % (t + 1, loss.data[0]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


test(net, loader_test)

torch.save(net.state_dict(), 'models/adv_trained_lenet5.pkl')
