"""
Adversarial attacks on LeNet5
"""
from time import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.utils import to_var, pred_batch, test, \
    attack_over_test_data, attack_over_test_data_batch

from models import LeNet5


# Hyper-parameters
param = {
    'test_batch_size': 100,
    'epsilon': 0.1,
}


# Data loaders
test_dataset = datasets.MNIST(root='../data/', train=False, download=True,
    transform=transforms.ToTensor())
loader_test = torch.utils.data.DataLoader(test_dataset, 
    batch_size=param['test_batch_size'], shuffle=False)


# Setup model to be attacked
net = LeNet5()
net.load_state_dict(torch.load('models/adv_trained_lenet5.pkl'))

if torch.cuda.is_available():
    print('CUDA ensabled.')
    net.cuda()

for p in net.parameters():
    p.requires_grad = False
net.eval()

test(net, loader_test)


# Adversarial attack
#adversary = FGSMAttack(net, param['epsilon'])
adversary = LinfPGDAttack(net, random_start=False)

t0 = time()
attack_over_test_data(net, adversary, param, loader_test)
print('{}s eclipsed.'.format(time()-t0))
print('')
t0 = time()
attack_over_test_data_batch(net, adversary, param, loader_test)
print('{}s eclipsed.'.format(time()-t0))
