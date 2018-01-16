"""
PyTorch Implementation of Papernot's Black-Box Attack
"""

import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.utils import to_var, pred_batch, test, \
    attack_over_test_data, batch_indices

from models import LeNet5, SubstituteModel


def MNIST_bbox_sub(param, loader_hold_out, loader_test):
    """
    Train a substitute model using Jacobian data augmentation
    """

    # Setup the substitute
    net = SubstituteModel()

    if torch.cuda.is_available():
        print('CUDA ensabled for the substitute.')
        net.cuda()
    net.train()

    # Setup the oracle
    oracle = LeNet5()

    if torch.cuda.is_available():
        print('CUDA ensabled for the oracle.')
        oracle.cuda()
    oracle.load_state_dict(torch.load(param['oracle_name']+'.pkl'))
    oracle.eval()


    # Setup training
    criterion = nn.CrossEntropyLoss()
    # Adam may cause the problem
    # (https://github.com/tensorflow/cleverhans/issues/183)
    optimizer = torch.optim.RMSprop(net.parameters(), 
        lr=param['learning_rate'])


    # Data held out for initial training
    data_iter = iter(loader_hold_out)
    X_sub, y_sub = data_iter.next()
    X_sub, y_sub = X_sub.numpy(), y_sub.numpy()

    # Train the substitute and augment dataset alternatively
    for rho in range(param['data_aug']):
        print("Substitute training epoch #"+str(rho))
        print("Training data: "+str(len(X_sub)))

        rng = np.random.RandomState()

        # model training
        for epoch in range(param['nb_epochs']):

            print('Starting epoch %d / %d' % (epoch + 1, param['nb_epochs']))

            # Compute number of batches
            nb_batches = int(np.ceil(float(len(X_sub)) / 
                param['test_batch_size']))
            assert nb_batches * param['test_batch_size'] >= len(X_sub)

            # Indices to shuffle training set
            index_shuf = list(range(len(X_sub)))
            rng.shuffle(index_shuf)

            for batch in range(nb_batches):

                # Compute batch start and end indices
                start, end = batch_indices(batch, len(X_sub), 
                    param['test_batch_size'])

                x = X_sub[index_shuf[start:end]]
                y = y_sub[index_shuf[start:end]]

                scores = net(to_var(torch.from_numpy(x)))
                loss = criterion(scores, to_var(torch.from_numpy(y).long()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('loss = %.8f' % (loss.data[0]))
        test(net, loader_test)

        # If we are not at last substitute training iteration, augment dataset
        if rho < param['data_aug'] - 1:
            print("Augmenting substitute training data.")
            # Perform the Jacobian augmentation
            X_sub = jacobian_augmentation(net, X_sub, y_sub)

            print("Labeling substitute training data.")
            # Label the newly generated synthetic points using the black-box
            scores = oracle(to_var(torch.from_numpy(X_sub)))
            # Note here that we take the argmax because the adversary
            # only has access to the label (not the probabilities) output
            # by the black-box model
            y_sub = np.argmax(scores.data.cpu().numpy(), axis=1)


    torch.save(net.state_dict(), param['oracle_name']+'_sub.pkl')




if __name__ == "__main__":

    # Hyper-parameters
    param = {
        'hold_out_size': 150,
        'test_batch_size': 128,
        'nb_epochs': 10,
        'learning_rate': 0.001,
        'data_aug': 6,
        'oracle_name': 'models/adv_trained_lenet5',
        'epsilon': 0.3,
    }

    # Data loaders
    # We need to hold out 150 data points from the test data
    # This is a bit tricky in PyTorch
    # We adopt the way from:
    # https://github.com/pytorch/pytorch/issues/1106
    hold_out_data = datasets.MNIST(root='../data/', train=True,
        download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root='../data/', train=False, 
        download=True, transform=transforms.ToTensor())

    indices = list(range(test_dataset.test_data.size(0)))
    split = param['hold_out_size']
    rng = np.random.RandomState()
    rng.shuffle(indices)

    hold_out_idx, test_idx = indices[:split], indices[split:]

    hold_out_sampler = SubsetRandomSampler(hold_out_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    loader_hold_out = torch.utils.data.DataLoader(hold_out_data, 
        batch_size=param['hold_out_size'], sampler=hold_out_sampler,
        shuffle=False)
    loader_test = torch.utils.data.DataLoader(test_dataset, 
        batch_size=param['test_batch_size'], sampler=test_sampler,
        shuffle=False)


    # Train the substitute
    MNIST_bbox_sub(param, loader_hold_out, loader_test)


    # Setup models
    net = SubstituteModel()
    oracle = LeNet5()

    net.load_state_dict(torch.load(param['oracle_name']+'_sub.pkl'))
    oracle.load_state_dict(torch.load(param['oracle_name']+'.pkl'))

    if torch.cuda.is_available():
        net.cuda()
        oracle.cuda()
        print('CUDA ensabled.')

    for p in net.parameters():
        p.requires_grad = False

    net.eval()
    oracle.eval()
    

    # Setup adversarial attacks
    adversary = FGSMAttack(net, param['epsilon'])

    test(net, loader_test)

    # Setup oracle

    attack_over_test_data(net, adversary, param, loader_test, oracle)
