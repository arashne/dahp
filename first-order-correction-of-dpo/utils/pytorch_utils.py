from typing import Union

import torch
from torch import nn

from utils.lookups import name_to_activation


#####################################
# NN
#####################################

Activation = Union[str, nn.Module]


def build_mlp(
        input_size: int,
        output_size: int,
        n_layer: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity'
):
    if isinstance(activation, str):
        activation = name_to_activation(activation)
    if isinstance(output_activation, str):
        output_activation = name_to_activation(output_activation)

    layers = []
    in_size = input_size
    for _ in range(n_layer):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)

    return nn.Sequential(*layers)


#####################################
# Device
#####################################

device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device(f'cuda:{gpu_id}')
        print(f'Using GPU id {gpu_id}')
    else:
        device = torch.device('cpu')
        print('GPU not detected. Defaulting to CPU.')


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
