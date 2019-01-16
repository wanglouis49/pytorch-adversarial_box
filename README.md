# Adversarial Box - Pytorch Adversarial Attack and Training

Luyu Wang and Gavin Ding, Borealis AI

## Motivation?
[CleverHans](https://github.com/tensorflow/cleverhans) comes in handy for Tensorflow. However, PyTorch does not have the luck at this moment. [Foolbox](https://github.com/bethgelab/foolbox) supports multiple deep learning frameworks, but it lacks many major implementations (e.g., black-box attack, Carlini-Wagner attack, adversarial training). We feel there is a need to write an easy-to-use and versatile library to help our fellow researchers and engineers.

**We have a much more updated version called [AdverTorch](https://github.com/BorealisAI/advertorch). You can find most of the popular attacks there. This repo will not be maintained anymore.**

## Usage
    from adversarialbox.attacks import FGSMAttack
    adversary = FGSMAttack(model, epsilon=0.1)
    X_adv = adversary.perturb(X_i, y_i)

## Examples
1. MNIST with FGSM ([code](https://github.com/wanglouis49/pytorch-adversarial_box/blob/master/mnist_attack.py))
2. Adversarial Training on MNIST ([code](https://github.com/wanglouis49/pytorch-adversarial_box/blob/master/mnist_adv_train.py))
3. MNIST using a black-box attack ([code](https://github.com/wanglouis49/pytorch-adversarial_box/blob/master/mnist_blackbox.py))

## List of supported attacks
1. FGSM
2. PGD
3. Black-box
