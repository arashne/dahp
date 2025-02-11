import torch

from envs.discrete_env import DiscreteMultiShiftedProximityEnv
from solvers.base_solver import BaseSolver


def calc_nbc(
        env: DiscreteMultiShiftedProximityEnv,
        solver: BaseSolver
) -> torch.Tensor:
    # Assumes BT

    n_state = solver.policy.n_state
    n_action = solver.policy.n_action

    nbc = torch.zeros((n_state, n_action))

    for s in range(n_state):
        env.update_state(s)
        rews_ea = torch.concat([e.reward(torch.arange(n_action)).view((1, -1)) for e in env.envs], dim=0)

        for a in range(n_action):
            drs_ea = rews_ea[:, a: a + 1] - rews_ea

            env_prior_1e = env.env_prior.probs.view((1, -1))
            ref_probs_1a = solver.ref_policy(torch.tensor([s])).probs

            nbc[s, a] = env_prior_1e @ torch.sigmoid(drs_ea) @ ref_probs_1a.T

    return nbc
