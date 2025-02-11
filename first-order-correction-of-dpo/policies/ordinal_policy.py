import torch

from utils import pytorch_utils as ptu
from policies.base_policy import BasePolicy


class OrdinalPolicy(BasePolicy):
    def __init__(
        self,
        n_state,
        n_action,
        use_raw_score=False,
    ):
        super().__init__()

        self.n_state = n_state
        self.n_action = n_action

        self.use_raw_score = use_raw_score

        self.scores_sa = torch.ones((self.n_state, self.n_action), device=ptu.device)*1e-6

    def forward(self, states_n: torch.Tensor) -> torch.distributions.Distribution:
        if states_n.ndim == 0:
            states_n = states_n.unsqueeze(-1)

        if self.use_raw_score:
            orders_na = self.scores_sa[states_n]
        else:
            orders_na = torch.argsort(torch.argsort(self.scores_sa[states_n], dim=1), dim=1)

        probs_na = orders_na / orders_na.sum(dim=1, keepdim=True)

        return torch.distributions.Categorical(probs=probs_na)

    def save(self, save_path: str):
        torch.save(
            {
                'kwargs': {
                    'n_state': self.n_state,
                    'n_action': self.n_action,
                    'use_raw_score': self.use_raw_score,
                },
                'scores_sa': self.scores_sa,
            },
            self._append_to_path(save_path),
        )

    @classmethod
    def load(cls, load_path: str):
        load_path = cls._append_to_path(load_path)
        checkpoint = torch.load(load_path)

        policy = cls(**checkpoint['kwargs'])
        policy.scores_sa = checkpoint['scores_sa']

        return policy
