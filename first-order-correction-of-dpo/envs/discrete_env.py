from typing import List, Union
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from envs.base_env import BaseEnv
from utils.lookups import name_to_distribution
from utils import pytorch_utils as ptu


class DecayFunc:
    LINEAR = 'linear'
    QUADRATIC = 'quadratic'
    EXPONENTIAL = 'exponential'
    GAUSSIAN = 'gaussian'

    def __init__(self, decay_func: str, rate: float):
        self.rate = rate

        self.forward = self.str_to_decay_func(decay_func)

    def linear(self, x):
        return torch.clamp(1 - self.rate*x, min=0.)

    def quadratic(self, x):
        return torch.clamp(1 - self.rate*(x**2), min=0.)

    def exponential(self, x):
        return self.rate**x

    def gaussian(self, x):
        return self.rate**(x**2)

    def str_to_decay_func(self, decay_func_str):
        if decay_func_str == self.LINEAR:
            return self.linear
        elif decay_func_str == self.EXPONENTIAL:
            return self.exponential
        elif decay_func_str == self.QUADRATIC:
            return self.quadratic
        elif decay_func_str == self.GAUSSIAN:
            return self.gaussian
        else:
            raise NotImplementedError


class DiscreteShiftedProximityEnv(BaseEnv):
    def __init__(
            self,
            n_state: int,
            n_action: int,
            shift: float,
            decay_func: str,
            decay_rate: float,
            rew_scale: float,
    ):
        super().__init__(prior='categorical', prior_params={'probs': torch.ones((n_state,))})

        self.n_state = n_state
        self.n_action = n_action
        self.shift = shift
        self.decay_func_str = decay_func
        self.decay_func = DecayFunc(decay_func, rate=decay_rate)
        self.rew_scale = rew_scale

    def reward(self, action: torch.Tensor):
        ss = (self.state + self.shift) % self.n_action  # Shifted state
        dist = torch.minimum((action - ss) % self.n_action, (ss - action) % self.n_action)
        # noinspection PyArgumentList
        return self.rew_scale * self.decay_func.forward(dist)

    def step(self, actions_n: torch.Tensor, fix_state_between_actions: bool):
        if actions_n.ndim == 0:
            actions_n = actions_n.unsqueeze(-1)

        assert actions_n.ndim == 1

        rews_n = torch.zeros_like(actions_n, dtype=torch.float)
        states_n = torch.zeros_like(actions_n, dtype=torch.long)
        shifts_n = torch.zeros_like(actions_n, dtype=torch.float)
        for i, action in enumerate(actions_n):
            # Find the reward
            rews_n[i] = self.reward(action)

            # Store
            states_n[i] = self.state
            shifts_n[i] = self.shift

            # Update the state
            if not fix_state_between_actions:
                self.reset()

        if fix_state_between_actions:
            self.reset()

        return states_n, rews_n, shifts_n


class DiscreteMultiShiftedProximityEnv(BaseEnv):
    def __init__(
            self,
            n_state: int,
            n_action: int,
            shifts: List[float],
            decay_func: str,
            decay_rates: Union[float, List[float]],
            rew_scales: Union[float, List[float]],
            output_all=False,
    ):
        super().__init__(prior='categorical', prior_params={'probs': torch.ones((n_state,))})

        if isinstance(decay_rates, float):
            decay_rates = [decay_rates]*len(shifts)
        if isinstance(rew_scales, float):
            rew_scales = [rew_scales]*len(shifts)

        self.n_state = n_state
        self.n_action = n_action
        self.shifts = shifts
        self.decay_func_str = decay_func
        self.decay_rates = decay_rates
        self.rew_scales = rew_scales
        self.output_all = output_all

        self.envs = []
        for shift, decay_rate, rew_scale in zip(shifts, decay_rates, rew_scales):
            env = DiscreteShiftedProximityEnv(
                n_state=n_state,
                n_action=n_action,
                shift=shift,
                decay_func=decay_func,
                decay_rate=decay_rate,
                rew_scale=rew_scale,
            )
            self.envs.append(env)

        self.state = self.envs[0].state
        self.update_state(self.state)
        self.env_prior = name_to_distribution('categorical')(probs=torch.ones((len(shifts),)))

    def update_state(self, state):
        self.state = state
        for env in self.envs:
            env.state = self.state

    def reward(self, action: torch.Tensor):
        rews = [env.reward(action) for env in self.envs]
        mean_rew = 0
        for p, rew in zip(self.env_prior.probs, rews):
            mean_rew += p*rew

        return mean_rew

    def step(self, actions_n: torch.Tensor, fix_state_between_actions: bool):
        if not self.output_all:
            # An env will be sampled and used for each call
            idx = self.env_prior.sample().item()
            env = self.envs[idx]

            step_output = env.step(actions_n, fix_state_between_actions)

            self.update_state(env.state)

        else:
            if actions_n.ndim == 0:
                actions_n = actions_n.unsqueeze(-1)

            n = len(actions_n)
            e = len(self.envs)

            states_n, rews_0_n, shifts_0_n = self.envs[0].step(actions_n, fix_state_between_actions)

            rews_ne = torch.zeros((n, e), dtype=rews_0_n.dtype).to(ptu.device)
            shifts_ne = torch.zeros((n, e), dtype=shifts_0_n.dtype).to(ptu.device)

            for idx, state, action, rew_0, shift_0 in zip(range(n), states_n, actions_n, rews_0_n, shifts_0_n):
                rews_ne[idx, 0] = rew_0
                shifts_ne[idx, 0] = shift_0
                for i_e, env in enumerate(self.envs[1:]):
                    env.state = state
                    _, rew_i, shift_i = env.step(action, fix_state_between_actions)
                    rews_ne[idx, i_e + 1] = rew_i
                    shifts_ne[idx, i_e + 1] = shift_i

            self.update_state(self.envs[0].state)

            step_output = states_n, rews_ne, shifts_ne

        return step_output


def plot_sample_env_rew():
    # For debugging
    envi = DiscreteMultiShiftedProximityEnv(1, 40, [-10, 0, 10], 'quadratic', [0.0075, 0.01, 0.0075], [4, 1, 4])
    rew_sa = torch.zeros((envi.n_state, envi.n_action))
    cnt_sa = torch.zeros((envi.n_state, envi.n_action))
    ac_dist = torch.distributions.Categorical(probs=torch.ones((envi.n_action,)))

    for _ in tqdm(range(30000)):
        a = ac_dist.sample()
        s, r, _ = envi.step(a, fix_state_between_actions=True)
        s, r = s[0], r[0]

        cnt_sa[s, a] += 1
        rew_sa[s, a] += r

    rew_sa = rew_sa / cnt_sa

    plt.figure()
    plt.imshow(rew_sa)
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.plot(rew_sa[0])
    plt.show()


def plot_sample_env_output_all_rew():
    # For debugging
    envi = DiscreteMultiShiftedProximityEnv(8, 8, [-2, 2], 'exponential', [0.8, 0.8], [1, 1], output_all=True)
    rew_sae = torch.zeros((envi.n_state, envi.n_action, len(envi.envs)))
    cnt_sa = torch.zeros((envi.n_state, envi.n_action))
    ac_dist = torch.distributions.Categorical(probs=torch.ones((envi.n_action,)))

    for _ in tqdm(range(30000)):
        a = ac_dist.sample()
        s, r, _ = envi.step(a, fix_state_between_actions=True)
        s, r = s[0], r[0]

        cnt_sa[s, a] += 1
        rew_sae[s, a] += r

    rew_sae = rew_sae / cnt_sa.unsqueeze(2)

    plt.figure()
    plt.imshow(rew_sae[:, :, 0])
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(rew_sae[:, :, 1])
    plt.colorbar()
    plt.show()

    plt.figure()
    for i_envi in range(len(envi.envs)):
        plt.plot(rew_sae[0, :, i_envi])
    plt.show()


if __name__ == '__main__':
    plot_sample_env_output_all_rew()
