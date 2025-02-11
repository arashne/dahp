import torch

from preferencemodels.base_preference import BasePreference


class BradleyTerry(BasePreference):
    def probs(self, rews_1_n: torch.Tensor, rews_2_n: torch.Tensor) -> torch.Tensor:
        if rews_1_n.ndim == 0:
            rews_1_n.unsqueeze(-1)
        if rews_2_n.ndim == 0:
            rews_2_n.unsqueeze(-1)

        diff_n = rews_2_n - rews_1_n

        return torch.sigmoid(diff_n)
