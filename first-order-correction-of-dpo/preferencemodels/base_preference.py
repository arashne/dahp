import abc
import torch


class BasePreference(abc.ABC):
    @abc.abstractmethod
    def probs(self, rews_1_n: torch.Tensor, rews_2_n: torch.Tensor) -> torch.Tensor:
        # Returns the probability that 2 is preferred to 1 per row
        pass

    def sample(self, rews_1_n: torch.Tensor, rews_2_n: torch.Tensor) -> torch.Tensor:
        probs_n = self.probs(rews_1_n, rews_2_n)

        prefs_n = torch.bernoulli(probs_n).to(torch.long)

        return prefs_n
