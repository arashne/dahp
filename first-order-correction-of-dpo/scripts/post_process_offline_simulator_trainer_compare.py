import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator, AutoLocator, FuncFormatter

from utils.load_and_save_utils import load_class
from envs.discrete_env import DiscreteMultiShiftedProximityEnv
from utils.theory import calc_nbc
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

    correction_vars = [1, 2]

    common_exp_name = \
        f'offline_size500000_' \
        f'{env.n_state}s{env.n_action}a_' \
        f'shifts{"_".join(["%g" % s for s in env.shifts])}_' \
        f'decay{env.decay_func_str}{"_".join(["%g" % dr for dr in env.decay_rates])}_' \
        f'rscale{"_".join(["%g" % rs for rs in env.rew_scales])}'

    postfixes = ['dpo'] + [f'estvarcorrecteddpo_varmult{"%g" % c_var}' for c_var in correction_vars]
    legends = [r'$\pi^*$', r'$\pi_{\rm DPO}$'] + [r'$\pi_{{\rm DPO}(\alpha=%g)}$' % c_var for c_var in correction_vars]
    assert len(legends) == len(postfixes) + 1

    exp_names = [common_exp_name + '_' + postfix for postfix in postfixes]

    log_dirs = [os.path.join('..', 'data', exp_name) for exp_name in exp_names]
    figs_dir = os.path.join('..', 'figs')

    plot_nbc = False
    save_figs = False

    # ===== Load =====
    solvers = [load_class(os.path.join(log_dir, 'solvers')).load(log_dir) for log_dir in log_dirs]

    n_state = solvers[0].policy.n_state
    n_action = solvers[0].policy.n_action

    # ===== Calc. OPT =====
    d = torch.arange(-(n_action // 2), -(n_action // 2) + n_action)
    beta = solvers[0].beta if hasattr(solvers[0], 'beta') else 1
    env.update_state(0)
    opt_a = torch.exp(env.reward(d) / beta).numpy()
    opt_a /= opt_a.sum()

    fig = cp.figure()

    ax1 = fig.add_subplot(111)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=2))

    ax1.plot(d, opt_a, 'k--', label=legends[0])

    # ===== Calc. NBC =====
    if plot_nbc:
        nbc_sa = calc_nbc(env, solvers[0])
        nbc_a = nbc_sa[-(n_action // 2), :]

        ax2 = ax1.twinx()
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1g}'))
        ax2.yaxis.set_major_locator(MultipleLocator(0.09))

        ax2.plot(d, nbc_a, '-.', color='c', label='NBC', zorder=-1)
        ax2.set_ylabel(r'NBC($\delta$)')

    # ===== Eval. policy =====
    cl = lambda i: [i/(len(solvers) - 1), 0, (1 - i/(len(solvers) - 1))] if len(solvers) > 1 else 'b'

    for idx in range(len(postfixes)):
        probs_sa = np.zeros((n_state, n_action))

        for state in range(n_state):
            probs_sa[state] = solvers[idx].policy(torch.tensor([state], dtype=torch.long)).probs[0].detach().numpy()

        aligned_sa = probs_sa.copy()
        for state in range(n_state):
            aligned_sa[state] = np.roll(probs_sa[state], n_state // 2 - state)

        aligned_a = np.mean(aligned_sa, axis=0)

        if n_state > 1:
            aligned_se_a = np.std(aligned_sa, axis=0) / np.sqrt(n_state - 1)
        else:
            aligned_se_a = 0.

        ax1.fill_between(d, aligned_a - aligned_se_a, aligned_a + aligned_se_a, color=cl(idx), alpha=0.1)
        ax1.plot(d, aligned_a, color=cl(idx), label=legends[idx + 1])

    # ax1.legend(loc='lower center')
    ax1.set_ylabel(r'$\pi(\delta)$')
    ax1.set_xlabel(r'$\delta$')

    handles, labels = ax1.get_legend_handles_labels()
    if plot_nbc:
        # noinspection PyUnboundLocalVariable
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles += handles2
        labels += labels2
    plt.legend(handles=handles, labels=labels, loc='lower center', ncol=2)

    cp.subplots_adjust()

    if save_figs:
        plt.savefig(os.path.join(
            figs_dir,
            f'aligned_policy_comparison_{common_exp_name}_{"_".join(postfixes)}{"_nbc" if plot_nbc else ""}.pdf'
        ))

    plt.show()
