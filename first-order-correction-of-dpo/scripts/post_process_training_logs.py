import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
import tensorflow as tf

from envs.discrete_env import DiscreteMultiShiftedProximityEnv


def get_section_tags(file):
    all_tags = set()
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            all_tags.add(v.tag)

    return all_tags


def get_section_results(file, tags):
    data = {tag: [] for tag in tags}
    print(data.keys())
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            for tag in tags:
                if v.tag == tag:
                    data[tag].append(v.simple_value)

    return data


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
        f'dpo'

    log_dir = os.path.join('..', 'data', exp_name)

    smoothness = 100

    # ===== Load logs =====
    files = glob.glob(os.path.join(log_dir, 'events*'))
    assert len(files) >= 1, f'searching for {log_dir} following files are found: ' + '\n'.join(files)
    log_path = sorted(files)[-1]

    tags = sorted([tag for tag in get_section_tags(log_path) if 'loss' in tag])
    section_results = get_section_results(log_path, tags)

    # ===== Plot =====
    plt.figure(figsize=(5*len(tags), 4))

    for i_tag, tag in enumerate(tags):
        loss = section_results[tag]

        loss = gaussian_filter1d(loss, sigma=smoothness)

        plt.subplot(1, len(tags), i_tag + 1)

        plt.plot(range(len(loss)), loss, 'k')

        plt.ylabel(tag)
        plt.xlabel('iter')

    plt.show()
