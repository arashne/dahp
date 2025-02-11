import os
import torch
import numpy as np
from matplotlib import pyplot as plt

from utils.load_and_save_utils import load_class
from envs.discrete_env import DiscreteMultiShiftedProximityEnv
from utils import consistent_plotting as cp


if __name__ == '__main__':
    # ===== Settings =====
    env = DiscreteMultiShiftedProximityEnv(
        n_state=40,
        n_action=40,
        shifts=[-10, 0, 10],
        decay_func='linear',
        decay_rates=[0.075, 0.1, 0.075],
        rew_scales=[4, 1.5, 4],
    )

    exp_name = \
        f'offline_size500000_' \
        f'{env.n_state}s{env.n_action}a_' \
        f'shifts{"_".join(["%g" % s for s in env.shifts])}_' \
        f'decay{env.decay_func_str}{"_".join(["%g" % dr for dr in env.decay_rates])}_' \
        f'rscale{"_".join(["%g" % rs for rs in env.rew_scales])}_' \
        f'dpo'

    log_dir = os.path.join('..', 'data', exp_name)
    figs_dir = os.path.join('..', 'figs')

    plot_rewards = True
    save_figs = True

    # ===== Load =====
    solver = load_class(os.path.join(log_dir, 'solvers')).load(log_dir)

    n_state = solver.policy.n_state
    n_action = solver.policy.n_action

    # ===== Eval. policy =====
    probs_sa = np.zeros((n_state, n_action))

    for state in range(n_state):
        probs_sa[state] = solver.policy(torch.tensor([state], dtype=torch.long)).probs[0].detach().numpy()

    # ===== Visualize policy =====
    # Matrix
    plt.figure(figsize=(5, 4))

    plt.imshow(probs_sa)
    plt.colorbar()

    plt.tight_layout()

    plt.show()

    # Vector
    cp.figure()

    aligned_sa = probs_sa.copy()
    for state in range(n_state):
        aligned_sa[state] = np.roll(probs_sa[state], n_action // 2 - state)

    aligned_a = np.mean(aligned_sa, axis=0)
    if n_state > 1:
        aligned_se_a = np.std(aligned_sa, axis=0)/np.sqrt(n_state - 1)
    else:
        aligned_se_a = 0.

    # Calc. OPT
    d = torch.arange(-(n_action // 2), -(n_action // 2) + n_action)
    beta = solver.beta if hasattr(solver, 'beta') else 1
    env.update_state(0)
    opt_a = torch.exp(env.reward(d) / beta).numpy()
    opt_a /= opt_a.sum()

    plt.plot(d, opt_a, 'k--')
    plt.plot(d, aligned_a, 'b')
    plt.fill_between(d, aligned_a - aligned_se_a, aligned_a + aligned_se_a, color='b', alpha=0.1)

    plt.legend(['OPT', solver.__class__.__name__], loc='lower center')
    plt.ylabel(r'$\pi(\delta)$')
    plt.xlabel(r'$\delta$')

    cp.subplots_adjust()

    if save_figs:
        plt.savefig(os.path.join(figs_dir, f'aligned_policy_{exp_name}.pdf'))

    # ===== Visualize rewards =====
    if plot_rewards:
        cp.figure()

        cmap = plt.cm.plasma

        d = torch.arange(-(n_action // 2), -(n_action // 2) + n_action)

        cl = lambda i: cmap((i + 0.5)/len(env.envs))
        env.update_state(0)
        for i_e, e in enumerate(env.envs):
            rew = e.reward(d)

            plt.plot(d, rew, color=cl(i_e), label=r'$u = %g$' % e.shift)

        plt.plot(d, env.reward(d), 'k--', label=r'$\mathbb{E}[r^*]$')

        plt.legend(loc='upper center', ncol=2)
        plt.ylabel(r'$r^*(\delta; u)$')
        plt.xlabel(r'$\delta$')

        cp.subplots_adjust()

        if save_figs:
            plt.savefig(os.path.join(figs_dir, f'rewards_{exp_name}.pdf'))

        plt.show()
