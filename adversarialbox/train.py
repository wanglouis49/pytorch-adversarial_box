"""
Adversarial training
"""

import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn

from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.utils import truncated_normal



def adv_train(X, y, model, criterion, adversary):
    """
    Adversarial training. Returns pertubed mini batch.
    """

    # If adversarial training, need a snapshot of 
    # the model at each batch to compute grad, so 
    # as not to mess up with the optimization step
    model_cp = copy.deepcopy(model)
    for p in model_cp.parameters():
        p.requires_grad = False
    model_cp.eval()
    
    m = X.size()[0] # mini-batch size

    adversary.model = model_cp

    X_adv = []
    for i in range(m):
        X_i, y_i = X[i:i+1].numpy(), y[i]

        X_i_adv = adversary.perturb(X_i, y_i)

        X_adv.append(X_i_adv[0])

    return torch.from_numpy(np.array(X_adv))


def FGSM_train_rnd(X, y, model, criterion, fgsm_adversary, epsilon_max=0.3):
    """
    FGSM with epsilon sampled from a truncated normal distribution.
    Returns pertubed mini batch.
    Kurakin et al, ADVERSARIAL MACHINE LEARNING AT SCALE, 2016
    """

    # If adversarial training, need a snapshot of 
    # the model at each batch to compute grad, so 
    # as not to mess up with the optimization step
    model_cp = copy.deepcopy(model)
    for p in model_cp.parameters():
        p.requires_grad = False
    model_cp.eval()
    
    m = X.size()[0] # mini-batch size

    fgsm_adversary.model = model_cp

    X_adv = []
    for i in range(m):
        X_i, y_i = X[i:i+1].numpy(), y[i]

        # truncated Gaussian
        mean, std = 0., epsilon_max/2
        epsilon = np.abs(truncated_normal(mean, std))

        fgsm_adversary.epsilon = epsilon
        X_i_adv = fgsm_adversary.perturb(X_i, y_i)

        X_adv.append(X_i_adv[0])

    return torch.from_numpy(np.array(X_adv))


