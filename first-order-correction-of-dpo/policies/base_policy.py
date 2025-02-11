import os
import abc
import torch
from torch import nn

from utils.load_and_save_utils import append_class_to_path


class BasePolicy(nn.Module, abc.ABC):
    def get_actions(self, states_n: torch.Tensor) -> torch.Tensor:
        return self(states_n).sample()

    @abc.abstractmethod
    def forward(self, states_n: torch.Tensor) -> torch.distributions.Distribution:
        pass

    @abc.abstractmethod
    def save(self, save_path: str):
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, load_path: str):
        pass

    @classmethod
    def _append_to_path(cls, path):
        return append_class_to_path(path, cls)
