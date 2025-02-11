import torch
from torch import nn

from utils import pytorch_utils as ptu
from policies.base_policy import BasePolicy


class DiscreteMLPPolicy(BasePolicy):
    def __init__(
        self,
        n_state,
        n_action,
        n_layer,
        size,
        activation='tanh',
    ):
        super().__init__()

        self.n_state = n_state
        self.n_action = n_action
        self.n_layer = n_layer
        self.size = size
        self.activation = activation

        self.logits_na = ptu.build_mlp(
            input_size=self.n_state,
            output_size=self.n_action,
            n_layer=self.n_layer,
            size=self.size,
            activation=self.activation,
            output_activation='identity',
        )
        self.logits_na.to(ptu.device)

    def forward(self, states_n: torch.Tensor) -> torch.distributions.Distribution:
        if states_n.ndim == 0:
            states_n = states_n.unsqueeze(-1)

        states_ns = nn.functional.one_hot(states_n, num_classes=self.n_state).to(torch.float)

        logits_na = self.logits_na(states_ns)

        return torch.distributions.Categorical(logits=logits_na)

    def save(self, save_path: str):
        torch.save(
            {
                'kwargs': {
                    'n_state': self.n_state,
                    'n_action': self.n_action,
                    'n_layer': self.n_layer,
                    'size': self.size,
                    'activation': self.activation,
                },
                'logits_na_state_dict': self.logits_na.state_dict(),
            },
            self._append_to_path(save_path),
        )

    @classmethod
    def load(cls, load_path: str):
        load_path = cls._append_to_path(load_path)
        checkpoint = torch.load(load_path)

        policy = cls(**checkpoint['kwargs'])
        policy.logits_na.load_state_dict(checkpoint['logits_na_state_dict'])

        return policy
