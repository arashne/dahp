import torch

from policies.base_policy import BasePolicy


class UniformPolicy(BasePolicy):
    def __init__(self, n_action):
        super().__init__()

        self.n_action = n_action

    def forward(self, states_n: torch.Tensor) -> torch.distributions.Distribution:
        n = len(states_n)
        return torch.distributions.Categorical(probs=torch.ones((n, self.n_action)))

    def save(self, save_path: str):
        torch.save(
            {
                'kwargs': {
                    'n_action': self.n_action,
                },
            },
            self._append_to_path(save_path),
        )

    @classmethod
    def load(cls, load_path: str):
        load_path = cls._append_to_path(load_path)
        checkpoint = torch.load(load_path)

        policy = cls(**checkpoint['kwargs'])

        return policy
