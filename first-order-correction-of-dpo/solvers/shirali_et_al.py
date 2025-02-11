import torch

from policies.base_policy import BasePolicy
from solvers.dpo import DPO

from utils import pytorch_utils as ptu


class ShiraliEtAl(DPO):
    def __init__(
            self,
            policy: BasePolicy,
            ref_policy: BasePolicy,
            beta: float,
            lr: float,
            method: int = 2,
    ):
        super().__init__(policy, ref_policy, beta, lr)

        self.beta = beta

        self.lr = lr
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr)

        self.method = method  # 1: Use i_func, 2: Use temp-adjusted DPO

    @staticmethod
    def i_func(k):
        if k == 1:
            return i_func_1
        elif k == 2:
            return i_func_2
        elif k == 3:
            return i_func_3
        elif k == 4:
            return i_func_4
        else:
            raise NotImplementedError

    def update(
            self,
            states_n: torch.Tensor,
            actions_1_n: torch.Tensor,
            actions_2_n: torch.Tensor,
            prefs_n: torch.Tensor,
            us_n: torch.Tensor,
    ):
        if prefs_n.ndim == 1:
            prefs_n = prefs_n.unsqueeze(-1)
        assert prefs_n.ndim == 2

        if us_n.ndim == 1:
            us_n = us_n.unsqueeze(-1)
        assert us_n.ndim == 2

        # Calc. h
        n, e = prefs_n.shape
        pseudo_prefs_n = torch.ones((n,), dtype=prefs_n.dtype).to(ptu.device)
        s_n = self.calc_likelihood(states_n, actions_1_n, actions_2_n, pseudo_prefs_n)

        # Calc. loss
        if self.method == 1:
            i_func = self.i_func(e)
            loss = \
                - prefs_n.prod(dim=1) * (s_n + i_func(s_n)) \
                - (1 - prefs_n).prod(dim=1) * (1 - s_n + i_func(1 - s_n))
        elif self.method == 2:
            h_n = torch.logit(s_n)
            loss = \
                - prefs_n.prod(dim=1) * torch.log(torch.sigmoid(e * h_n)) \
                - (1 - prefs_n).prod(dim=1) * torch.log(torch.sigmoid(e * (1 - h_n)))
        else:
            raise NotImplementedError

        loss = loss.mean()

        # Step the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'training_loss': loss.item()}


def i_func_1(x: torch.Tensor) -> torch.Tensor:
    return -x + torch.log(x)


def i_func_2(x: torch.Tensor) -> torch.Tensor:
    return x - 2*torch.log(x) - 1/x


def i_func_3(x: torch.Tensor) -> torch.Tensor:
    return -x + 3*torch.log(x) + 3/x - 1/2/(x**2)


def i_func_4(x: torch.Tensor) -> torch.Tensor:
    return x - 4*torch.log(x) - 6/x + 2/(x**2) - 1/3/(x**3)
