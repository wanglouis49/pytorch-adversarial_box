import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn

from adversarialbox.utils import to_var

# --- White-box attacks ---

class FGSMAttack(object):
    def __init__(self, model=None, epsilon=None):
        """
        One step fast gradient sign method
        """
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, x_nat, y):
        """
        Given one example (x_nat, y), returns its adversarial
        counterpart with an attack length of epsilon.
        """
        x = np.copy(x_nat)

        x_var = to_var(torch.from_numpy(x), requires_grad=True)
        y_var = to_var(torch.LongTensor([int(y)]))

        scores = self.model(x_var)
        loss = self.loss_fn(scores, y_var)
        loss.backward()
        grad_sign = x_var.grad.data.cpu().sign().numpy()

        x += self.epsilon * grad_sign
        x = np.clip(x, 0, 1)

        return x


class LinfPGDAttack(object):
    def __init__(self, model=None, epsilon=0.3, k=40, a=0.01, 
        random_start=True):
        """
        Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point.
        https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, x_nat, y):
        """
        Given one example (x_nat, y), returns an adversarial
        examples within epsilon of x_nat in l_infinity norm.
        """
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, 
                x_nat.shape).astype('float32')
        else:
            x = np.copy(x_nat)

        for i in range(self.k):
            x_var = to_var(torch.from_numpy(x), requires_grad=True)
            y_var = to_var(torch.LongTensor([y]))

            scores = self.model(x_var)
            loss = self.loss_fn(scores, y_var)
            loss.backward()
            grad = x_var.grad.data.cpu().numpy()

            x += self.a * np.sign(grad)

            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = np.clip(x, 0, 1) # ensure valid pixel range

        return x


# --- Black-box attacks ---

def jacobian(model, x, nb_classes=10):
    """
    This function will return a list of PyTorch gradients
    """
    list_derivatives = []
    x_var = to_var(torch.from_numpy(x), requires_grad=True)

    # derivatives for each class
    for class_ind in range(nb_classes):
        score = model(x_var)[:, class_ind]
        score.backward()
        list_derivatives.append(x_var.grad.data.cpu().numpy())
        x_var.grad.data.zero_()

    return list_derivatives


def jacobian_augmentation(model, X_sub_prev, Y_sub, lmbda=0.1):
    """
    Create new numpy array for adversary training data
    with twice as many components on the first dimension.
    """
    X_sub = np.vstack([X_sub_prev, X_sub_prev])

    # For each input in the previous' substitute training iteration
    for ind, x in enumerate(X_sub_prev):
        grads = jacobian(model, x)
        # Select gradient corresponding to the label predicted by the oracle
        grad = grads[Y_sub[ind]]

        # Compute sign matrix
        grad_val = np.sign(grad)

        # Create new synthetic point in adversary substitute training set
        X_sub[len(X_sub_prev)+ind] = X_sub[ind] + lmbda * grad_val #???

    # Return augmented training data (needs to be labeled afterwards)
    return X_sub
