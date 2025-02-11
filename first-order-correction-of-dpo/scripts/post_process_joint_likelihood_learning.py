import os
import torch
import numpy as np
from matplotlib import pyplot as plt

from utils.load_and_save_utils import load_class
from envs.discrete_env import DiscreteMultiShiftedProximityEnv


def sigmoid(x):
    return 1/(1 + np.exp(-x))


if __name__ == '__main__':
    # ===== Settings =====
    env = DiscreteMultiShiftedProximityEnv(
        n_state=20,
        n_action=20,
        shifts=[-5, 0, 5],
        decay_func='linear',
        decay_rates=[0.15, 0.2, 0.15],
        rew_scales=[4, 1.5, 4],
    )

    exp_name = \
        f'offline_size300000_' \
        f'{env.n_state}s{env.n_action}a_' \
        f'shifts{"_".join(["%g" % s for s in env.shifts])}_' \
        f'decay{env.decay_func_str}{"_".join(["%g" % dr for dr in env.decay_rates])}_' \
        f'rscale{"_".join(["%g" % rs for rs in env.rew_scales])}_' \
        f'estvarcorrecteddpo_varmult1'

    log_dir = os.path.join('..', 'data', exp_name)
    figs_dir = os.path.join('..', 'figs')

    save_figs = False

    # ===== Load =====
    solver = load_class(os.path.join(log_dir, 'solvers')).load(log_dir)

    n_state = solver.policy.n_state
    n_action = solver.policy.n_action

    # ===== Eval. policy =====
    probs_saa = np.zeros((n_state, n_action, n_action))
    probs_sa = np.zeros((n_state, n_action))

    for state in range(n_state):
        for action_1 in range(n_action):
            for action_2 in range(n_action):
                probs_saa[state, action_1, action_2] = solver.joint_likelihood_mdl(
                    torch.tensor([state]), torch.tensor([action_1]), torch.tensor([action_2]),
                    torch.tensor([state]), torch.tensor([action_1]), torch.tensor([action_2]),
                )[0].detach().numpy()

    for state in range(n_state):
        for action_1 in range(n_action):
            probs_sa[state, action_1] = probs_saa[state, state, action_1]

    # ===== Visualize policy =====
    # Matrix
    plt.figure(figsize=(5, 4))

    plt.imshow(probs_sa)
    plt.colorbar()

    plt.tight_layout()

    # Vector
    plt.figure(figsize=(5, 4))

    aligned_sa = probs_sa.copy()
    for state in range(n_state):
        aligned_sa[state] = np.roll(probs_sa[state], n_state // 2 - state)

    aligned_a = np.mean(aligned_sa, axis=0)

    d = torch.arange(-(n_action // 2), -(n_action // 2) + n_action)

    opt_a = torch.zeros((n_action,))

    env.update_state(0)
    for e in env.envs:
        opt_a += 1/len(env.envs) * sigmoid(e.reward(d) - e.reward(torch.tensor(0)))**2

    opt_a = opt_a.numpy()

    plt.plot(d, opt_a, 'k--')
    plt.plot(d, aligned_a, 'b')

    plt.ylabel('Joint likelihood')

    plt.legend(['OPT', 'Est.'])

    plt.tight_layout()

    plt.show()

    if save_figs:
        plt.savefig(os.path.join(figs_dir, f'aligned_joint_likelihood_{exp_name}.pdf'))
