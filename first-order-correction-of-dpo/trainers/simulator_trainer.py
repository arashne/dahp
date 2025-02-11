import torch
from typing import Optional

from envs.base_env import BaseEnv
from preferencemodels.base_preference import BasePreference
from solvers.base_solver import BaseSolver
from utils.logger import Logger
from trainers.base_trainer import BaseSimulatorTrainer


class OfflineSimulatorTrainer(BaseSimulatorTrainer):
    def __init__(
            self,
            env: BaseEnv,
            preference_model: BasePreference,
            solver: BaseSolver,
            dataset_size: int,
            batch_size: int,
            logger: Optional[Logger] = None,
    ):
        super().__init__(env, preference_model, solver, logger)

        self.dataset_size = dataset_size
        self.batch_size = batch_size

        self.dataset = self.sample_dataset()

        self.random_indices = torch.randperm(self.dataset_size)
        self.dataset_head = 0

    def sample_dataset(self):
        states_n, actions_1_n, actions_2_n, prefs_n, us_n = self.sample(self.solver.ref_policy, self.dataset_size)

        dataset = {
            'states': states_n,
            'actions_1': actions_1_n,
            'actions_2': actions_2_n,
            'prefs': prefs_n,
            'us': us_n,
        }

        return dataset

    def sample_batch(self):
        self.dataset_head = self.dataset_head

        indices = self.random_indices[self.dataset_head: (self.dataset_head + self.batch_size)]

        self.dataset_head += len(indices)

        if self.dataset_head >= self.dataset_size - self.batch_size - 1:
            self.dataset_head = 0
            self.random_indices = torch.randperm(self.dataset_size)

        return \
            self.dataset['states'][indices], \
            self.dataset['actions_1'][indices], \
            self.dataset['actions_2'][indices], \
            self.dataset['prefs'][indices], \
            self.dataset['us'][indices],

    def train_one_step(self):
        states_n, actions_1_n, actions_2_n, prefs_n, us_n = self.sample_batch()

        return self.solver.update(states_n, actions_1_n, actions_2_n, prefs_n, us_n)
