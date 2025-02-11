import abc
import torch
import numpy as np
import random

from utils.lookups import name_to_distribution

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class BaseEnv(abc.ABC):
    def __init__(self, prior: str, prior_params: dict):
        self.prior = name_to_distribution(prior)(**prior_params)

        self.state = None
        self.reset()

    def reset(self):
        self.state = self.prior.sample()

    @abc.abstractmethod
    def step(self, actions_n: torch.Tensor, fix_state_between_actions: bool):
        pass
