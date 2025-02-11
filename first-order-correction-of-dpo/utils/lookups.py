import torch
from torch import nn


def name_to_distribution(name):
    _name_to_distribution = {
        'categorical': torch.distributions.Categorical
    }
    if name in _name_to_distribution:
        return _name_to_distribution[name]
    else:
        raise ModuleNotFoundError


def name_to_activation(name):
    _name_to_activation = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'leaky_relu': nn.LeakyReLU(),
        'sigmoid': nn.Sigmoid(),
        'selu': nn.SELU(),
        'softplus': nn.Softplus(),
        'identity': nn.Identity(),
    }
    if name in _name_to_activation:
        return _name_to_activation[name]
    else:
        raise ModuleNotFoundError
