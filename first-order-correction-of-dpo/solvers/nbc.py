import torch

from policies.base_policy import BasePolicy
from policies.ref_policy import UniformPolicy
from policies.ordinal_policy import OrdinalPolicy
from solvers.base_solver import BaseSolver
from utils.load_and_save_utils import append_class_to_path, load_class
from utils import pytorch_utils as ptu


class NBC(BaseSolver):
    def __init__(
            self,
            policy: OrdinalPolicy,
            ref_policy: BasePolicy,
    ):
        super().__init__(policy, ref_policy)

        assert isinstance(ref_policy, UniformPolicy), 'Only implemented for a uniform ref policy.'

        self.win_sa = torch.ones((policy.n_state, policy.n_action)).to(ptu.device)*1e-6
        self.lose_sa = torch.ones((policy.n_state, policy.n_action)).to(ptu.device)*1e-6

    def update(
            self,
            states_n: torch.Tensor,
            actions_1_n: torch.Tensor,
            actions_2_n: torch.Tensor,
            prefs_n: torch.Tensor,
            us_n: torch.Tensor,
    ):
        n = len(states_n)

        actions_n = torch.concat([actions_1_n.unsqueeze(-1), actions_2_n.unsqueeze(-1)], dim=1)
        actions_w_n = actions_n[torch.arange(n), prefs_n]  # Winners
        actions_l_n = actions_n[torch.arange(n), 1 - prefs_n]  # Losers

        self.win_sa.index_put_(
            (states_n, actions_w_n), torch.ones_like(states_n, dtype=self.win_sa.dtype), accumulate=True
        )
        self.lose_sa.index_put_(
            (states_n, actions_l_n), torch.ones_like(states_n, dtype=self.win_sa.dtype), accumulate=True
        )

        self.policy.scores_sa = self.win_sa / (self.win_sa + self.lose_sa)

        return {}  # For compatibility only

    def save(self, save_path: str):
        assert self.policy.__class__.__name__ != self.ref_policy.__class__.__name__

        self.policy.save(save_path)
        self.ref_policy.save(save_path)

        torch.save(
            {
                'policies': {
                    'policy_appendix': append_class_to_path('', self.policy.__class__),
                    'ref_policy_appendix': append_class_to_path('', self.ref_policy.__class__),
                },
                'win_sa': self.win_sa,
                'lose_sa': self.lose_sa,
            },
            self._append_to_path(save_path),
        )

    @classmethod
    def load(cls, load_path: str):
        # Load the solver checkpoint
        solver_load_path = cls._append_to_path(load_path)
        checkpoint = torch.load(solver_load_path)

        # Load the policies
        policy = load_class(load_path + checkpoint['policies']['policy_appendix']).load(load_path)
        ref_policy = load_class(load_path + checkpoint['policies']['ref_policy_appendix']).load(load_path)

        # Load the solver
        solver = cls(policy, ref_policy)
        solver.win_sa = checkpoint['win_sa']
        solver.lose_sa = checkpoint['lose_sa']

        return solver

