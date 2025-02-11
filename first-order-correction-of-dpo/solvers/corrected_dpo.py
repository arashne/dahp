import torch
from torch import nn
from typing import Optional

from policies.base_policy import BasePolicy
from policies.mlp_policy import DiscreteMLPPolicy
from solvers.dpo import DPO
from utils.load_and_save_utils import append_class_to_path, load_class
from utils import pytorch_utils as ptu


class CorrectedDPO(DPO):
    def __init__(
            self,
            policy: BasePolicy,
            ref_policy: BasePolicy,
            beta: float,
            lr: float,
            var_multiplier: float,
    ):
        super().__init__(policy, ref_policy, beta, lr)

        self.var_multiplier = var_multiplier

        self.cnt_saau = torch.ones((policy.n_state, policy.n_action, policy.n_action, 2), dtype=torch.long)

    def __str__(self):
        return f'{self.__class__.__name__.lower()}_varmult{"%g" % self.var_multiplier}'

    def calc_likelihood_correction(
            self,
            states_n: torch.Tensor,
            actions_1_n: torch.Tensor,
            actions_2_n: torch.Tensor,
            prefs_n: torch.Tensor,
            sigmoid_n: torch.Tensor,
    ):
        n = len(states_n)

        actions_n = torch.concat([actions_1_n.unsqueeze(-1), actions_2_n.unsqueeze(-1)], dim=1)
        actions_w_n = actions_n[torch.arange(n), prefs_n]  # Winners
        actions_l_n = actions_n[torch.arange(n), 1 - prefs_n]  # Losers

        # Calc. var
        likelihood_1_n = self.cnt_saau[states_n, actions_w_n, actions_l_n, 1] / (
            self.cnt_saau[states_n, actions_w_n, actions_l_n, 1] + self.cnt_saau[states_n, actions_l_n, actions_w_n, 1]
        )

        likelihood_0_n = self.cnt_saau[states_n, actions_w_n, actions_l_n, 0] / (
            self.cnt_saau[states_n, actions_w_n, actions_l_n, 0] + self.cnt_saau[states_n, actions_l_n, actions_w_n, 0]
        )

        mean_likelihood_n = (likelihood_1_n + likelihood_0_n)/2
        p_1 = self.cnt_saau[:, :, :, 1].sum() / self.cnt_saau.sum()
        var_likelihood_n = \
            p_1*(likelihood_1_n - mean_likelihood_n)**2 \
            + (1 - p_1)*(likelihood_0_n - mean_likelihood_n)**2

        # Calc. correction
        correction_n = 0.5*(1 - 2*sigmoid_n)*var_likelihood_n/sigmoid_n/(1 - sigmoid_n)

        return correction_n

    def update(
            self,
            states_n: torch.Tensor,
            actions_1_n: torch.Tensor,
            actions_2_n: torch.Tensor,
            prefs_n: torch.Tensor,
            us_n: torch.Tensor,
    ):
        # Update the counter
        n = len(states_n)

        actions_n = torch.concat([actions_1_n.unsqueeze(-1), actions_2_n.unsqueeze(-1)], dim=1)
        actions_w_n = actions_n[torch.arange(n), prefs_n]  # Winners
        actions_l_n = actions_n[torch.arange(n), 1 - prefs_n]  # Losers

        self.cnt_saau.index_put_(
            (states_n, actions_w_n, actions_l_n, (us_n > 0) * 1),
            torch.ones_like(states_n, dtype=self.cnt_saau.dtype),
            accumulate=True,
        )
        self.cnt_saau[states_n, actions_w_n, actions_l_n, (us_n > 0) * 1] += 1

        # Calc. likelihood
        sigmoid_n = self.calc_likelihood(states_n, actions_1_n, actions_2_n, prefs_n)

        # Calc. correction
        correction_n = self.calc_likelihood_correction(states_n, actions_1_n, actions_2_n, prefs_n, sigmoid_n)

        # Calc. loss
        likelihood_n = sigmoid_n + self.var_multiplier*correction_n

        loss = -torch.log(likelihood_n).mean()

        # Step the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'training_loss': loss.item()}

    def save(self, save_path: str):
        assert self.policy.__class__.__name__ != self.ref_policy.__class__.__name__

        self.policy.save(save_path)
        self.ref_policy.save(save_path)

        torch.save(
            {
                'kwargs': {
                    'beta': self.beta,
                    'lr': self.lr,
                    'var_multiplier': self.var_multiplier,
                },
                'policies': {
                    'policy_appendix': append_class_to_path('', self.policy.__class__),
                    'ref_policy_appendix': append_class_to_path('', self.ref_policy.__class__),
                },
                'optimizer_state_dict': self.optimizer.state_dict(),
                'cnt_saau': self.cnt_saau,
            },
            self._append_to_path(save_path),
        )

    @classmethod
    def load(cls, load_path: str):
        # Load the solver checkpoint
        solver_load_path = cls._append_to_path(load_path)
        checkpoint = torch.load(solver_load_path)

        # Load the policies
        policy = load_class(load_path + checkpoint['policies']['policy_appendix']).load(load_path)
        ref_policy = load_class(load_path + checkpoint['policies']['ref_policy_appendix']).load(load_path)

        # Load the solver
        solver = cls(policy, ref_policy, **checkpoint['kwargs'])
        solver.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        solver.cnt_saau = checkpoint['cnt_saau']

        return solver


class FixedVarCorrectedDPO(DPO):
    def __init__(
            self,
            policy: BasePolicy,
            ref_policy: BasePolicy,
            beta: float,
            lr: float,
            var: float,
    ):
        super().__init__(policy, ref_policy, beta, lr)

        self.var = var

    def __str__(self):
        return f'{self.__class__.__name__.lower()}_var{"%g" % self.var}'

    def update(
            self,
            states_n: torch.Tensor,
            actions_1_n: torch.Tensor,
            actions_2_n: torch.Tensor,
            prefs_n: torch.Tensor,
            us_n: torch.Tensor,
    ):
        # Calc. likelihood
        sigmoid_n = self.calc_likelihood(states_n, actions_1_n, actions_2_n, prefs_n)

        # Calc. loss
        loss = -torch.log(sigmoid_n + 0.5*self.var*(1 - 2*sigmoid_n)/sigmoid_n/(1 - sigmoid_n)).mean()

        # Step the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'training_loss': loss.item()}

    def save(self, save_path: str):
        assert self.policy.__class__.__name__ != self.ref_policy.__class__.__name__

        self.policy.save(save_path)
        self.ref_policy.save(save_path)

        torch.save(
            {
                'kwargs': {
                    'beta': self.beta,
                    'lr': self.lr,
                    'var': self.var,
                },
                'policies': {
                    'policy_appendix': append_class_to_path('', self.policy.__class__),
                    'ref_policy_appendix': append_class_to_path('', self.ref_policy.__class__),
                },
                'optimizer_state_dict': self.optimizer.state_dict(),
            },
            self._append_to_path(save_path),
        )


class EstVarCorrectedDPO(DPO):
    def __init__(
            self,
            policy: DiscreteMLPPolicy,
            ref_policy: BasePolicy,
            beta: float,
            lr: float,
            var_multiplier: float,
            step: int = 0,
            start_correction_after_step: int = 0,
            joint_likelihood_mdl: Optional[nn.Module] = None,
            joint_likelihood_params: Optional[dict] = None,
            joint_likelihood_lr: Optional[float] = None,
            joint_likelihood_lr_scheduler_step_size: int = 5000,
            joint_likelihood_lr_scheduler_gamma: float = 1,
            use_general_joint_likelihood_mdl: bool = True,
            correction_method: int = 4,
            loss_fn: str = 'ce',
            latent_disc_loss_weight: float = 0.,
    ):
        super().__init__(policy, ref_policy, beta, lr)

        self.n_state = policy.n_state
        self.n_action = policy.n_action

        assert (joint_likelihood_mdl is not None) or (joint_likelihood_params is not None)

        self.var_multiplier = var_multiplier

        self.step = step
        self.start_correction_after_step = start_correction_after_step

        self.joint_likelihood_params = joint_likelihood_params
        self.joint_likelihood_lr = joint_likelihood_lr or self.lr
        self.joint_likelihood_lr_scheduler_gamma = joint_likelihood_lr_scheduler_gamma
        self.joint_likelihood_lr_scheduler_step_size = joint_likelihood_lr_scheduler_step_size
        self.use_general_joint_likelihood_mdl = use_general_joint_likelihood_mdl
        if joint_likelihood_mdl is None:
            if self.use_general_joint_likelihood_mdl:
                joint_likelihood_cls = GeneralJointLikelihoodModel
            else:
                joint_likelihood_cls = MultiTabularRewardBasedJointLikelihoodModel

            self.joint_likelihood_mdl = joint_likelihood_cls(
                self.n_state, self.n_action, **self.joint_likelihood_params
            )

        else:
            self.joint_likelihood_mdl = joint_likelihood_mdl

        self.joint_likelihood_optimizer = torch.optim.Adam(
            self.joint_likelihood_mdl.parameters(), lr=self.joint_likelihood_lr
        )
        self.joint_likelihood_scheduler = torch.optim.lr_scheduler.StepLR(
            self.joint_likelihood_optimizer,
            step_size=self.joint_likelihood_lr_scheduler_step_size,
            gamma=self.joint_likelihood_lr_scheduler_gamma,
        )

        self.correction_method = correction_method
        self.loss_fn = loss_fn
        self.latent_disc_loss_weight = latent_disc_loss_weight

    def __str__(self):
        return f'{self.__class__.__name__.lower()}_varmult{"%g" % self.var_multiplier}'

    def calc_likelihood_correction(
            self,
            states_n: torch.Tensor,
            actions_1_n: torch.Tensor,
            actions_2_n: torch.Tensor,
            prefs_n: torch.Tensor,
            sigmoid_n: torch.Tensor,
    ):
        # Calc. joint likelihood and correction
        n = len(states_n)

        actions_n = torch.concat([actions_1_n.unsqueeze(-1), actions_2_n.unsqueeze(-1)], dim=1)
        actions_w_n = actions_n[torch.arange(n), prefs_n]  # Winners
        actions_l_n = actions_n[torch.arange(n), 1 - prefs_n]  # Losers

        a_n, _ = self.joint_likelihood_mdl(
            states_n, actions_l_n, actions_w_n, states_n, actions_l_n, actions_w_n
        )
        a_n = a_n[:, 0].detach()

        correction_a_n = (1 - 2*sigmoid_n) * (a_n - sigmoid_n**2) / (4 - 6*sigmoid_n) / sigmoid_n

        if self.correction_method == 1:
            correction_n = correction_a_n

        elif self.correction_method == 2:
            b_n, _ = self.joint_likelihood_mdl(
                states_n, actions_w_n, actions_l_n, states_n, actions_w_n, actions_l_n
            )
            b_n = b_n[:, 0].detach()

            correction_b_n = (1 - 2*sigmoid_n) * (b_n - (1 - sigmoid_n)**2) / (6*sigmoid_n - 2) / (1 - sigmoid_n)

            corrections_n2 = torch.concat([correction_a_n.view(-1, 1), correction_b_n.view(-1, 1)], dim=1)

            correction_n = corrections_n2[torch.arange(n), (sigmoid_n > 0.5)*1]

        elif self.correction_method == 3:
            b_n, _ = self.joint_likelihood_mdl(
                states_n, actions_w_n, actions_l_n, states_n, actions_w_n, actions_l_n
            )
            b_n = b_n[:, 0].detach()

            var_sigmoid_n = torch.clamp(a_n - 1/4 * (1 + a_n - b_n)**2, min=0.)

            correction_n = 0.5 * (1 - 2*sigmoid_n) * var_sigmoid_n / sigmoid_n / (1 - sigmoid_n)

        elif self.correction_method == 4:
            b_n, _ = self.joint_likelihood_mdl(
                states_n, actions_l_n, actions_w_n, states_n, actions_w_n, actions_l_n
            )
            b_n = b_n[:, 0].detach()

            var_sigmoid_n = torch.clamp(a_n - (a_n + b_n)**2, min=0.)

            # Derivation:
            # var_n = var_sigmoid_n / (sigmoid_n * (1 - sigmoid_n))**2
            # sigmoid'' = sigmoid_n * (1 - sigmoid_n) * (1 - 2*sigmoid_n)
            # correction_n = 0.5 * var_n * sigmoid''

            correction_n = 0.5 * (1 - 2*sigmoid_n) * var_sigmoid_n / sigmoid_n / (1 - sigmoid_n)

        else:
            raise Exception(f'Correction method {self.correction_method} is not valid.')

        correction_n = torch.clamp(correction_n, min=-0.1, max=0.1)

        return correction_n

    def update_joint_likelihood_mdl(
            self,
            states_n: torch.Tensor,
            actions_1_n: torch.Tensor,
            actions_2_n: torch.Tensor,
            prefs_n: torch.Tensor,
            us_n: torch.Tensor,
    ):
        # Find similars
        us_n1 = us_n.unsqueeze(-1)
        sim_nn = torch.triu((us_n1 == us_n1.T)*1, diagonal=1)
        filt = torch.any(sim_nn, dim=1)

        # Find first similar per row (if any)
        sim_idx_fn = torch.argmax(sim_nn[filt], dim=1)

        # Extract the first batch
        states_l_n = states_n[filt]
        actions_1_l_n = actions_1_n[filt]
        actions_2_l_n = actions_2_n[filt]

        # Extract the second batch
        states_r_n = states_n[sim_idx_fn]
        actions_1_r_n = actions_1_n[sim_idx_fn]
        actions_2_r_n = actions_2_n[sim_idx_fn]

        # Calc. likelihood and loss
        joint_likelihood_n, joint_likelihood_nl = self.joint_likelihood_mdl(
            states_l_n, actions_1_l_n, actions_2_l_n, states_r_n, actions_1_r_n, actions_2_r_n
        )
        joint_likelihood_n = joint_likelihood_n[:, 0]

        labels_n = prefs_n[filt] * prefs_n[sim_idx_fn]

        if self.loss_fn == 'mse':
            se_n = (joint_likelihood_n - labels_n) ** 2
            loss = se_n.mean()
        elif self.loss_fn == 'ce':
            ce_n = -labels_n*torch.log(joint_likelihood_n) - (1 - labels_n)*torch.log(1 - joint_likelihood_n)
            loss = ce_n.mean()
        else:
            raise Exception(f'loss_fn {self.loss_fn} is not recognized.')

        # Disc. loss
        disc_loss = self.joint_likelihood_mdl.calc_latent_disc_loss()

        loss += self.latent_disc_loss_weight * disc_loss

        # Step the optimizer
        self.joint_likelihood_optimizer.zero_grad()
        loss.backward()
        self.joint_likelihood_optimizer.step()
        self.joint_likelihood_scheduler.step()

        return {'joint_likelihood_training_loss': loss.item()}

    def update(
            self,
            states_n: torch.Tensor,
            actions_1_n: torch.Tensor,
            actions_2_n: torch.Tensor,
            prefs_n: torch.Tensor,
            us_n: torch.Tensor,
    ):
        # Calc. likelihood
        sigmoid_n = self.calc_likelihood(states_n, actions_1_n, actions_2_n, prefs_n)

        # Calc. loss
        if self.step >= self.start_correction_after_step:
            # Calc. likelihood correction
            correction_n = self.calc_likelihood_correction(
                states_n, actions_1_n, actions_2_n, prefs_n, sigmoid_n
            )

            likelihood_n = sigmoid_n + self.var_multiplier*correction_n
        else:
            likelihood_n = sigmoid_n

        loss = -torch.log(likelihood_n).mean()

        # Step the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step += 1

        # Update the joint likelihood model
        out_dict = self.update_joint_likelihood_mdl(states_n, actions_1_n, actions_2_n, prefs_n, us_n)

        out_dict['training_loss'] = loss.item()

        return out_dict

    def save(self, save_path: str):
        assert self.policy.__class__.__name__ != self.ref_policy.__class__.__name__

        self.policy.save(save_path)
        self.ref_policy.save(save_path)

        torch.save(
            {
                'kwargs': {
                    'beta': self.beta,
                    'lr': self.lr,
                    'var_multiplier': self.var_multiplier,
                    'step': self.step,
                    'start_correction_after_step': self.start_correction_after_step,
                    'joint_likelihood_params': self.joint_likelihood_params,
                    'joint_likelihood_lr': self.joint_likelihood_lr,
                    'joint_likelihood_lr_scheduler_gamma': self.joint_likelihood_lr_scheduler_gamma,
                    'joint_likelihood_lr_scheduler_step_size': self.joint_likelihood_lr_scheduler_step_size,
                    'use_general_joint_likelihood_mdl': self.use_general_joint_likelihood_mdl,
                    'correction_method': self.correction_method,
                    'loss_fn': self.loss_fn,
                    'latent_disc_loss_weight': self.latent_disc_loss_weight,
                },
                'policies': {
                    'policy_appendix': append_class_to_path('', self.policy.__class__),
                    'ref_policy_appendix': append_class_to_path('', self.ref_policy.__class__),
                },
                'optimizer_state_dict': self.optimizer.state_dict(),
                'joint_likelihood_mdl_state_dict': self.joint_likelihood_mdl.state_dict(),
                'joint_likelihood_optimizer_state_dict': self.joint_likelihood_optimizer.state_dict(),
            },
            self._append_to_path(save_path),
        )

    @classmethod
    def load(cls, load_path: str):
        # Load the solver checkpoint
        solver_load_path = cls._append_to_path(load_path)
        checkpoint = torch.load(solver_load_path)

        # Load the policies
        policy = load_class(load_path + checkpoint['policies']['policy_appendix']).load(load_path)
        ref_policy = load_class(load_path + checkpoint['policies']['ref_policy_appendix']).load(load_path)

        # Load the solver
        solver = cls(policy, ref_policy, **checkpoint['kwargs'])
        solver.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        solver.joint_likelihood_optimizer.load_state_dict(checkpoint['joint_likelihood_optimizer_state_dict'])
        solver.joint_likelihood_mdl.load_state_dict(checkpoint['joint_likelihood_mdl_state_dict'])

        return solver


class GeneralJointLikelihoodModel(nn.Module):
    def __init__(self, n_state: int, n_action: int, n_layer: int, size: int):
        super().__init__()

        self.n_state = n_state
        self.n_action = n_action

        self.mdl = ptu.build_mlp(
            input_size=2*(self.n_state + 2*self.n_action),
            output_size=1,
            output_activation='sigmoid',
            n_layer=n_layer,
            size=size,
        )

    def forward(
            self,
            states_l_n,
            actions_1_l_n,
            actions_2_l_n,
            states_r_n,
            actions_1_r_n,
            actions_2_r_n,
    ):
        states_l_ns = nn.functional.one_hot(states_l_n, num_classes=self.n_state).to(torch.float)
        actions_1_l_na = nn.functional.one_hot(actions_1_l_n, num_classes=self.n_action).to(torch.float)
        actions_2_l_na = nn.functional.one_hot(actions_2_l_n, num_classes=self.n_action).to(torch.float)

        states_r_ns = nn.functional.one_hot(states_r_n, num_classes=self.n_state).to(torch.float)
        actions_1_r_na = nn.functional.one_hot(actions_1_r_n, num_classes=self.n_action).to(torch.float)
        actions_2_r_na = nn.functional.one_hot(actions_2_r_n, num_classes=self.n_action).to(torch.float)

        joint_inputs = torch.concat(
            [states_l_ns, actions_1_l_na, actions_2_l_na, states_r_ns, actions_1_r_na, actions_2_r_na],
            dim=1,
        )

        prob_n = self.mdl(joint_inputs)

        return prob_n, None


class MultiRewardBasedJointLikelihoodModel(nn.Module):
    def __init__(self, n_state: int, n_action: int, n_layer: int, size: int, latent_dim: int):
        super().__init__()

        self.n_state = n_state
        self.n_action = n_action
        self.latent_dim = latent_dim

        self.rew_mdl = ptu.build_mlp(
            input_size=(self.n_state + self.n_action + self.latent_dim),
            output_size=1,
            output_activation='identity',
            n_layer=n_layer,
            size=size,
        )

        self.pref_mdl = nn.functional.sigmoid

        self.latent_logits = nn.Parameter(torch.zeros((self.latent_dim,), device=ptu.device), requires_grad=True)

    def calc_latent_disc_loss(self):
        u_ne = self.rew_mdl[0].weight.T[-self.latent_dim:]

        u_ne = u_ne / u_ne.norm(dim=1, keepdim=True)
        dissimilarity_nn = 1 - torch.matmul(u_ne, u_ne.T)

        msk = torch.triu(torch.ones_like(dissimilarity_nn, device=ptu.device), diagonal=1) == 1
        loss = -dissimilarity_nn[msk].mean()

        return loss

    def forward(
            self,
            states_l_n,
            actions_1_l_n,
            actions_2_l_n,
            states_r_n,
            actions_1_r_n,
            actions_2_r_n,
    ):
        latent_probs = torch.softmax(self.latent_logits, dim=0)

        n = len(states_l_n)

        states_l_ns = nn.functional.one_hot(states_l_n, num_classes=self.n_state).to(torch.float)
        actions_1_l_na = nn.functional.one_hot(actions_1_l_n, num_classes=self.n_action).to(torch.float)
        actions_2_l_na = nn.functional.one_hot(actions_2_l_n, num_classes=self.n_action).to(torch.float)

        states_r_ns = nn.functional.one_hot(states_r_n, num_classes=self.n_state).to(torch.float)
        actions_1_r_na = nn.functional.one_hot(actions_1_r_n, num_classes=self.n_action).to(torch.float)
        actions_2_r_na = nn.functional.one_hot(actions_2_r_n, num_classes=self.n_action).to(torch.float)

        prob_nl = []
        for u in range(self.latent_dim):
            u_nu = nn.functional.one_hot(
                torch.ones((n,), dtype=torch.long) * u,
                num_classes=self.latent_dim,
            ).to(torch.float)

            rew_1_l_n = self.rew_mdl(torch.concat([states_l_ns, actions_1_l_na, u_nu], dim=1))
            rew_2_l_n = self.rew_mdl(torch.concat([states_l_ns, actions_2_l_na, u_nu], dim=1))
            prob_l_n = self.pref_mdl(rew_2_l_n - rew_1_l_n)

            rew_1_r_n = self.rew_mdl(torch.concat([states_r_ns, actions_1_r_na, u_nu], dim=1))
            rew_2_r_n = self.rew_mdl(torch.concat([states_r_ns, actions_2_r_na, u_nu], dim=1))
            prob_r_n = self.pref_mdl(rew_2_r_n - rew_1_r_n)

            prob_nl.append(prob_l_n * prob_r_n)

        prob_nl = torch.concat(prob_nl, dim=1)
        prob_n = torch.matmul(prob_nl, latent_probs.view(-1, 1))

        return prob_n, prob_nl


class MultiTabularRewardBasedJointLikelihoodModel(nn.Module):
    def __init__(self, n_state: int, n_action: int, n_layer: int, size: int, latent_dim: int):
        super().__init__()

        self.n_state = n_state
        self.n_action = n_action
        self.latent_dim = latent_dim

        self.rew_mdl = nn.Parameter(
            torch.randn((self.n_state, self.n_action, self.latent_dim,), device=ptu.device),
            requires_grad=True,
        )

        self.pref_mdl = nn.functional.sigmoid

        self.latent_logits = nn.Parameter(torch.zeros((self.latent_dim,), device=ptu.device), requires_grad=True)

    @staticmethod
    def calc_latent_disc_loss():
        return 0.

    def forward(
            self,
            states_l_n,
            actions_1_l_n,
            actions_2_l_n,
            states_r_n,
            actions_1_r_n,
            actions_2_r_n,
    ):
        latent_probs = torch.softmax(self.latent_logits, dim=0)

        prob_nl = []
        for u in range(self.latent_dim):
            rew_1_l_n = self.rew_mdl[states_l_n, actions_1_l_n, u]
            rew_2_l_n = self.rew_mdl[states_l_n, actions_2_l_n, u]
            prob_l_n = self.pref_mdl(rew_2_l_n - rew_1_l_n)

            rew_1_r_n = self.rew_mdl[states_r_n, actions_1_r_n, u]
            rew_2_r_n = self.rew_mdl[states_r_n, actions_2_r_n, u]
            prob_r_n = self.pref_mdl(rew_2_r_n - rew_1_r_n)

            prob_nl.append((prob_l_n * prob_r_n).view(-1, 1))

        prob_nl = torch.concat(prob_nl, dim=1)
        prob_n = torch.matmul(prob_nl, latent_probs.view(-1, 1))

        return prob_n, prob_nl


class MultiLocRewardBasedJointLikelihoodModel(nn.Module):
    def __init__(self, n_state: int, n_action: int, n_layer: int, size: int, latent_dim: int):
        super().__init__()

        self.n_state = n_state
        self.n_action = n_action
        self.latent_dim = latent_dim

        self.rew_mdl = nn.Parameter(
            torch.randn((self.n_action, self.latent_dim,), device=ptu.device),
            requires_grad=True,
        )

        self.pref_mdl = nn.functional.sigmoid

        self.latent_logits = nn.Parameter(torch.zeros((self.latent_dim,), device=ptu.device), requires_grad=True)

    @staticmethod
    def calc_latent_disc_loss():
        return 0.

    def forward(
            self,
            states_l_n,
            actions_1_l_n,
            actions_2_l_n,
            states_r_n,
            actions_1_r_n,
            actions_2_r_n,
    ):
        latent_probs = torch.softmax(self.latent_logits, dim=0)

        prob_nl = []
        for u in range(self.latent_dim):
            rew_1_l_n = self.rew_mdl[(actions_1_l_n - states_l_n) % self.n_action, u]
            rew_2_l_n = self.rew_mdl[(actions_2_l_n - states_l_n) % self.n_action, u]
            prob_l_n = self.pref_mdl(rew_2_l_n - rew_1_l_n)

            rew_1_r_n = self.rew_mdl[(actions_1_r_n - states_r_n) % self.n_action, u]
            rew_2_r_n = self.rew_mdl[(actions_2_r_n - states_r_n) % self.n_action, u]
            prob_r_n = self.pref_mdl(rew_2_r_n - rew_1_r_n)

            prob_nl.append((prob_l_n * prob_r_n).view(-1, 1))

        prob_nl = torch.concat(prob_nl, dim=1)
        prob_n = torch.matmul(prob_nl, latent_probs.view(-1, 1))

        return prob_n, prob_nl


class MultiRewardEncoderBasedJointLikelihoodModel(nn.Module):
    def __init__(self, n_state: int, n_action: int, n_layer: int, size: int, latent_dim: int):
        super().__init__()

        self.n_state = n_state
        self.n_action = n_action
        self.latent_dim = latent_dim

        self.rew_mdl = ptu.build_mlp(
            input_size=3*size,
            output_size=1,
            output_activation='identity',
            n_layer=n_layer,
            size=size,
        )

        self.state_encoder = nn.Linear(self.n_state, size)
        self.action_encoder = nn.Linear(self.n_action, size)
        self.latent_encoder = nn.Linear(self.latent_dim, size)

        self.pref_mdl = nn.functional.sigmoid

        self.latent_logits = nn.Parameter(torch.zeros((self.latent_dim,), device=ptu.device), requires_grad=True)

    def calc_latent_disc_loss(self):
        u_n = torch.arange(self.latent_dim, device=ptu.device)
        u_nu = nn.functional.one_hot(u_n, num_classes=self.latent_dim).to(torch.float)
        u_ne = self.latent_encoder(u_nu)

        u2_ne = (u_ne**2).sum(dim=1, keepdim=True)
        dissimilarity_nn = u2_ne + u2_ne.T - 2*torch.matmul(u_ne, u_ne.T)

        msk = torch.triu(torch.ones_like(dissimilarity_nn, device=ptu.device), diagonal=1) == 1
        loss = -dissimilarity_nn[msk].mean()

        return loss

    def forward(
            self,
            states_l_n,
            actions_1_l_n,
            actions_2_l_n,
            states_r_n,
            actions_1_r_n,
            actions_2_r_n,
    ):
        latent_probs = torch.softmax(self.latent_logits, dim=0)

        n = len(states_l_n)

        states_l_ns = nn.functional.one_hot(states_l_n, num_classes=self.n_state).to(torch.float)
        actions_1_l_na = nn.functional.one_hot(actions_1_l_n, num_classes=self.n_action).to(torch.float)
        actions_2_l_na = nn.functional.one_hot(actions_2_l_n, num_classes=self.n_action).to(torch.float)

        states_l_ne = self.state_encoder(states_l_ns)
        actions_1_l_ne = self.action_encoder(actions_1_l_na)
        actions_2_l_ne = self.action_encoder(actions_2_l_na)

        states_r_ns = nn.functional.one_hot(states_r_n, num_classes=self.n_state).to(torch.float)
        actions_1_r_na = nn.functional.one_hot(actions_1_r_n, num_classes=self.n_action).to(torch.float)
        actions_2_r_na = nn.functional.one_hot(actions_2_r_n, num_classes=self.n_action).to(torch.float)

        states_r_ne = self.state_encoder(states_r_ns)
        actions_1_r_ne = self.action_encoder(actions_1_r_na)
        actions_2_r_ne = self.action_encoder(actions_2_r_na)

        prob_nl = []
        for u in range(self.latent_dim):
            u_nu = nn.functional.one_hot(
                torch.ones((n,), dtype=torch.long) * u,
                num_classes=self.latent_dim,
            ).to(torch.float)

            u_ne = self.latent_encoder(u_nu)

            rew_1_l_n = self.rew_mdl(torch.concat([states_l_ne, actions_1_l_ne, u_ne], dim=1))
            rew_2_l_n = self.rew_mdl(torch.concat([states_l_ne, actions_2_l_ne, u_ne], dim=1))
            prob_l_n = self.pref_mdl(rew_2_l_n - rew_1_l_n)

            rew_1_r_n = self.rew_mdl(torch.concat([states_r_ne, actions_1_r_ne, u_ne], dim=1))
            rew_2_r_n = self.rew_mdl(torch.concat([states_r_ne, actions_2_r_ne, u_ne], dim=1))
            prob_r_n = self.pref_mdl(rew_2_r_n - rew_1_r_n)

            prob_nl.append(prob_l_n * prob_r_n)

        prob_nl = torch.concat(prob_nl, dim=1)
        prob_n = torch.matmul(prob_nl, latent_probs.view(-1, 1))

        return prob_n, prob_nl
