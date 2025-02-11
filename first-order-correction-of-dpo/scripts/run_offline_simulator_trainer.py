import os

from envs.discrete_env import DiscreteMultiShiftedProximityEnv
from preferencemodels.bradley_terry import BradleyTerry
from policies.mlp_policy import DiscreteMLPPolicy
from policies.ordinal_policy import OrdinalPolicy
from policies.ref_policy import UniformPolicy
from solvers.dpo import DPO
from solvers.corrected_dpo import CorrectedDPO, EstVarCorrectedDPO
from solvers.shirali_et_al import ShiraliEtAl
from solvers.nbc import NBC
from trainers.simulator_trainer import OfflineSimulatorTrainer
from utils.logger import Logger
from utils.pytorch_utils import init_gpu


if __name__ == '__main__':
    # ===== Init. =====
    # GPU
    init_gpu()

    # Solver class
    solver_class = ShiraliEtAl

    # Env
    env = DiscreteMultiShiftedProximityEnv(
        n_state=40,
        n_action=40,
        shifts=[-10, 0, 10],
        decay_func='linear',
        decay_rates=[0.075, 0.1, 0.075],
        rew_scales=[4, 1.5, 4],
        output_all=(solver_class == ShiraliEtAl),
    )

    # Preference
    pref_mdl = BradleyTerry()

    # Polices
    if solver_class in (DPO, CorrectedDPO, EstVarCorrectedDPO, ShiraliEtAl):
        pi = DiscreteMLPPolicy(
            n_state=env.n_state,
            n_action=env.n_action,
            n_layer=2,
            size=15,
        )

    elif solver_class == NBC:
        pi = OrdinalPolicy(
            n_state=env.n_state,
            n_action=env.n_action,
            use_raw_score=True,
        )

    else:
        raise NotImplementedError

    ref_pi = UniformPolicy(n_action=env.n_action)

    # Solver
    if solver_class == DPO:
        solver = DPO(
            policy=pi,
            ref_policy=ref_pi,
            beta=1.,
            lr=1e-3,
        )

    elif solver_class == CorrectedDPO:
        solver = CorrectedDPO(
            policy=pi,
            ref_policy=ref_pi,
            beta=1.,
            lr=1e-3,
            var_multiplier=1.,
        )

    elif solver_class == EstVarCorrectedDPO:
        solver = EstVarCorrectedDPO(
            policy=pi,
            ref_policy=ref_pi,
            beta=1.,
            lr=1e-3,
            var_multiplier=1.,
            start_correction_after_step=8000,
            joint_likelihood_params={'n_layer': 2, 'size': 15, 'latent_dim': len(set(env.shifts))},
            joint_likelihood_lr=1e-2,
            joint_likelihood_lr_scheduler_gamma=1.,
            joint_likelihood_lr_scheduler_step_size=5000,
            use_general_joint_likelihood_mdl=False,
            correction_method=4,
            loss_fn='ce',
            latent_disc_loss_weight=0.,
        )

    elif solver_class == NBC:
        solver = NBC(
            policy=pi,
            ref_policy=ref_pi,
        )

    elif solver_class == ShiraliEtAl:
        solver = ShiraliEtAl(
            policy=pi,
            ref_policy=ref_pi,
            beta=1.,
            lr=1e-3,
            method=2,
        )

    else:
        raise NotImplementedError

    # Logger
    exp_name = \
        f'offline_size500000_' \
        f'{env.n_state}s{env.n_action}a_' \
        f'shifts{"_".join(["%g" % s for s in env.shifts])}_' \
        f'decay{env.decay_func_str}{"_".join(["%g" % dr for dr in env.decay_rates])}_' \
        f'rscale{"_".join(["%g" % rs for rs in env.rew_scales])}_' \
        f'{str(solver)}'

    logger = Logger(
        log_dir=os.path.join('..', 'data', exp_name),
        logging_freq=1,
    )

    # Trainer
    trainer = OfflineSimulatorTrainer(
        env=env,
        preference_model=pref_mdl,
        solver=solver,
        logger=logger,
        dataset_size=500000,
        batch_size=1024,
    )

    # ===== Train =====
    if solver_class in (DPO, CorrectedDPO, EstVarCorrectedDPO, ShiraliEtAl):
        n_step = 10000
    elif solver_class == NBC:
        n_step = 100000
    else:
        raise NotImplementedError

    trainer.train(steps=n_step)
