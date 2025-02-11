import os
import abc
import torch

from policies.base_policy import BasePolicy
from utils.load_and_save_utils import append_class_to_path


class BaseSolver(abc.ABC):
    def __init__(self, policy: BasePolicy, ref_policy: BasePolicy):
        self.policy = policy
        self.ref_policy = ref_policy

    def __str__(self):
        return self.__class__.__name__.lower()

    @abc.abstractmethod
    def update(
            self,
            states_n: torch.Tensor,
            actions_1_n: torch.Tensor,
            actions_2_n: torch.Tensor,
            prefs_n: torch.Tensor,
            us_n: torch.Tensor,
    ):
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
