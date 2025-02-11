import torch

from policies.base_policy import BasePolicy
from solvers.base_solver import BaseSolver
from utils.load_and_save_utils import append_class_to_path, load_class


class DPO(BaseSolver):
    def __init__(
            self,
            policy: BasePolicy,
            ref_policy: BasePolicy,
            beta: float,
            lr: float,
    ):
        super().__init__(policy, ref_policy)

        self.beta = beta

        self.lr = lr
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr)

    def calc_likelihood(
            self,
            states_n: torch.Tensor,
            actions_1_n: torch.Tensor,
            actions_2_n: torch.Tensor,
            prefs_n: torch.Tensor,
    ):
        n = len(states_n)

        dist_na = self.policy(states_n)
        ref_dist_na = self.ref_policy(states_n)

        actions_n = torch.concat([actions_1_n.unsqueeze(-1), actions_2_n.unsqueeze(-1)], dim=1)
        actions_w_n = actions_n[torch.arange(n), prefs_n]  # Winners
        actions_l_n = actions_n[torch.arange(n), 1 - prefs_n]  # Losers

        log_probs_w_n = dist_na.log_prob(actions_w_n)
        log_probs_l_n = dist_na.log_prob(actions_l_n)

        ref_log_probs_w_n = ref_dist_na.log_prob(actions_w_n)
        ref_log_probs_l_n = ref_dist_na.log_prob(actions_l_n)

        sigmoid_n = torch.sigmoid(
            self.beta*(log_probs_w_n - log_probs_l_n) - self.beta*(ref_log_probs_w_n - ref_log_probs_l_n)
        )

        return sigmoid_n

    def update(
            self,
            states_n: torch.Tensor,
            actions_1_n: torch.Tensor,
            actions_2_n: torch.Tensor,
            prefs_n: torch.Tensor,
            us_n: torch.Tensor,
    ):
        # Calc. likelihood
        sigmoid_n = self.calc_likelihood(states_n, actions_1_n, actions_2_n, prefs_n)

        # Calc. loss
        loss = -torch.log(sigmoid_n).mean()

        # Step the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'training_loss': loss.item()}

    def save(self, save_path: str):
        assert self.policy.__class__.__name__ != self.ref_policy.__class__.__name__

        self.policy.save(save_path)
        self.ref_policy.save(save_path)

        torch.save(
            {
                'kwargs': {
                    'beta': self.beta,
                    'lr': self.lr,
                },
                'policies': {
                    'policy_appendix': append_class_to_path('', self.policy.__class__),
                    'ref_policy_appendix': append_class_to_path('', self.ref_policy.__class__),
                },
                'optimizer_state_dict': self.optimizer.state_dict(),
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
        solver = cls(policy, ref_policy, **checkpoint['kwargs'])
        solver.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return solver
