"""PPGA Emitter"""
import itertools
from typing import Optional, List

import numpy as np
import torch
import wandb

from ribs.emitters._dqd_emitter_base import DQDEmitterBase
from ribs.emitters.opt import AdamOpt, GradientAscentOpt
from ribs.emitters.opt._xnes import ExponentialES
from ribs.emitters.rankers import _get_ranker
from baselines.PPGA.utils.utilities import log
from ribs.archives import ArchiveBase
from baselines.PPGA.RL.ppo import PPO


class PPGAEmitter(DQDEmitterBase):
    def __init__(self,
                 ppo: PPO,
                 archive: ArchiveBase,
                 x0: np.ndarray,
                 sigma0: float,
                 batch_size: int = 100,
                 ranker: str = "2imp",
                 restart_rule: str = 'no_improvement',
                 normalize_grad: bool = True,
                 epsilon: float = 1e-8,
                 seed: Optional[int] = None,
                 use_wandb: bool = False,
                 bounds: Optional[List[float]] = None,
                 *,
                 grad_opt: str = 'ppo',
                 step_size: Optional[float] = None,
                 normalize_obs: bool = True,
                 normalize_returns: bool = True,
                 adaptive_stddev: bool = True
                 ):
        DQDEmitterBase.__init__(self, archive, len(x0), bounds)
        self._epsilon = epsilon
        self._rng = np.random.default_rng(seed)
        self._x0 = np.array(x0, dtype=archive.dtype)
        self._sigma0 = sigma0
        self._grad_coefficients = None
        self._normalize_grads = normalize_grad
        self._jacobian_batch = None
        self._ranker = _get_ranker(ranker)
        self._ranker.reset(self, archive, self._rng)
        self.ppo = ppo

        assert grad_opt in ['ppo', 'adam', 'gradient_ascent'], f'Invalid gradient optimizer passed in, {grad_opt=}'
        self._grad_opt = None
        if grad_opt == 'adam':
            self._grad_opt = AdamOpt(self._x0, step_size)
        elif grad_opt == 'gradient_ascent':
            self._grad_opt = GradientAscentOpt(self._x0, step_size)
        elif grad_opt == 'ppo':
            self.ppo.theta = self._x0
            self._grad_opt = self.ppo
        else:
            raise ValueError(f"Invalid Gradient Ascent Optimizer {grad_opt}")

        self._restart_rule = restart_rule
        self._restarts = 0
        self._itrs = 0
        # Check if the restart_rule is valid.
        _ = self._check_restart(0)
        self._restart_rule = restart_rule

        # We have a coefficient for each measure and an extra coefficient for
        # the objective.
        self._num_coefficients = archive.measure_dim + 1

        self.opt_seed = seed
        self.opt_lambda = batch_size
        self.device = torch.device('cuda')
        self._initial_bounds = ([-2.0] * (archive.measure_dim + 1), [2.0] * (archive.measure_dim + 1))
        self._initial_bounds[0][0] = 0.0  # restrict on-restart sampling of grad f coefficients to be [0.0, 2.0]
        self.opt = ExponentialES(solution_dim=self._num_coefficients,
                                 device=self.device,
                                 sigma0=sigma0,
                                 popsize=self.opt_lambda,
                                 seed=seed,
                                 initial_bounds=self._initial_bounds)

        self._batch_size = batch_size
        self._restarts = 0
        self._itrs = 0
        self._step_size = step_size
        self._use_wandb = use_wandb
        self._normalize_obs = normalize_obs
        self._normalize_returns = normalize_returns
        self._mean_agent_obs_normalizer = None
        self._mean_agent_return_normalizer = None

    @property
    def x0(self):
        """numpy.ndarray: Initial solution for the optimizer."""
        return self._x0

    @property
    def batch_size(self):
        """int: Number of solutions to return in :meth:`ask`."""
        return self._batch_size

    @property
    def batch_size_dqd(self):
        """int: Number of solutions to return in :meth:`ask_dqd`.

        This is always 1, as we only return the solution point in
        :meth:`ask_dqd`.
        """
        return 1

    @property
    def restarts(self):
        """int: The number of restarts for this emitter."""
        return self._restarts

    @property
    def itrs(self):
        """int: The number of iterations for this emitter."""
        return self._itrs

    @property
    def epsilon(self):
        """int: The epsilon added for numerical stability when normalizing
        gradients in :meth:`tell_dqd`."""
        return self._epsilon

    @property
    def theta(self):
        return self._grad_opt.theta

    @property
    def mean_agent_obs_normalizer(self):
        return self._mean_agent_obs_normalizer

    @mean_agent_obs_normalizer.setter
    def mean_agent_obs_normalizer(self, obs_normalizer):
        self._mean_agent_obs_normalizer = obs_normalizer

    @property
    def mean_agent_return_normalizer(self):
        return self._mean_agent_return_normalizer

    @mean_agent_return_normalizer.setter
    def mean_agent_return_normalizer(self, rew_normalizer):
        self._mean_agent_return_normalizer = rew_normalizer

    def update_theta(self, new_theta):
        self._grad_opt.theta = new_theta

    def ask_dqd(self):
        """Samples a new solution from the gradient optimizer.

        **Call :meth:`ask_dqd` and :meth:`tell_dqd` (in this order) before
        calling :meth:`ask` and :meth:`tell`.**

        Returns:
            a new solution to evaluate.
        """
        return self._grad_opt.theta[None]

    def ask(self):
        """Samples new solutions from a gradient aborescence parameterized by a
        multivariate Gaussian distribution.

        The multivariate Gaussian is parameterized by the evolution strategy
        optimizer ``self.opt``.

        Note that this method returns `batch_size - 1` solution as one solution
        is returned via ask_dqd.

        Returns:
            (batch_size, :attr:`solution_dim`) array -- a batch of new solutions
            to evaluate.
        """
        self._grad_coefficients = self.opt.ask()
        noise = np.expand_dims(self._grad_coefficients, axis=2)
        return self._grad_opt.theta + np.sum(
            np.multiply(self._jacobian_batch, noise), axis=1)

    def _check_restart(self, num_parents):
        """Emitter-side checks for restarting the optimizer.

        The optimizer also has its own checks.

        Args:
            num_parents (int): The number of solution to propagate to the next
                generation from the solutions generated by CMA-ES.
        Raises:
          ValueError: If :attr:`restart_rule` is invalid.
        """
        if isinstance(self._restart_rule, (int, np.integer)):
            return self._itrs % self._restart_rule == 0
        if self._restart_rule == "no_improvement":
            return num_parents == 0
        if self._restart_rule == "basic":
            return False
        raise ValueError(f"Invalid restart_rule {self._restart_rule}")

    def tell_dqd(self,
                 solution_batch,
                 objective_batch,
                 measures_batch,
                 jacobian_batch,
                 status_batch,
                 value_batch,
                 metadata_batch=None):
        """Gives the emitter results from evaluating the gradient of the
        solutions.

        Args:
            solution_batch (numpy.ndarray): (batch_size, :attr:`solution_dim`)
                array of solutions generated by this emitter's :meth:`ask()`
                method.
            objective_batch (numpy.ndarray): 1d array containing the objective
                function value of each solution.
            measures_batch (numpy.ndarray): (batch_size, measure space
                dimension) array with the measure space coordinates of each
                solution.
            jacobian_batch (numpy.ndarray): ``(batch_size, 1 + measure_dim,
                solution_dim)`` array consisting of Jacobian matrices of the
                solutions obtained from :meth:`ask_dqd`. Each matrix should
                consist of the objective gradient of the solution followed by
                the measure gradients.
            status_batch (numpy.ndarray): 1d array of
                :class:`ribs.archive.addstatus` returned by a series of calls to
                archive's :meth:`add()` method.
            value_batch (numpy.ndarray): 1d array of floats returned by a series
                of calls to archive's :meth:`add()` method. for what these
                floats represent, refer to :meth:`ribs.archives.add()`.
            metadata_batch (numpy.ndarray): 1d object array containing a
                metadata object for each solution.
        """
        if self._normalize_grads:
            norms = (np.linalg.norm(jacobian_batch, axis=2, keepdims=True) +
                     self._epsilon)
            jacobian_batch /= norms
        self._jacobian_batch = jacobian_batch

    def tell(self,
             solution_batch,
             objective_batch,
             measures_batch,
             status_batch,
             value_batch,
             metadata_batch=None):
        """Gives the emitter results from evaluating solutions.

        The solutions are ranked based on the `rank()` function defined by
        `self._ranker`.

        Args:
            solution_batch (numpy.ndarray): (batch_size, :attr:`solution_dim`)
                array of solutions generated by this emitter's :meth:`ask()`
                method.
            objective_batch (numpy.ndarray): 1d array containing the objective
                function value of each solution.
            measures_batch (numpy.ndarray): (batch_size, measure space
                dimension) array with the measure space coordinates of each
                solution.
            status_batch (numpy.ndarray): 1d array of
                :class:`ribs.archive.addstatus` returned by a series of calls to
                archive's :meth:`add()` method.
            value_batch (numpy.ndarray): 1d array of floats returned by a series
                of calls to archive's :meth:`add()` method. for what these
                floats represent, refer to :meth:`ribs.archives.add()`.
            metadata_batch (numpy.ndarray): 1d object array containing a
                metadata object for each solution.
        """
        if self._jacobian_batch is None:
            raise RuntimeError("tell() was called without calling tell_dqd().")

        metadata_batch = itertools.repeat(
            None) if metadata_batch is None else metadata_batch

        # Count number of new solutions.
        new_sols = status_batch.astype(bool).sum()

        # Sort the solutions using ranker.
        indices, ranking_values = self._ranker.rank(
            self, self.archive, self._rng, solution_batch, objective_batch,
            measures_batch, status_batch, value_batch, metadata_batch)

        num_parents = self._batch_size

        value_batch_parents = value_batch[indices][:num_parents]
        mean_value = np.mean(value_batch_parents)
        max_value = np.max(value_batch)
        log.debug(f'{mean_value=}, {max_value=}')
        if self._use_wandb:
            wandb.log({
                'QD/mean_value': mean_value,
                'QD/max_value': max_value,
                'QD/iteration': self._itrs,
                'QD/new_sols': new_sols
            })

        # Update Evolution Strategy.
        coeffs, coeff_rankings = self._grad_coefficients[indices], ranking_values[indices]
        coeffs, coeff_rankings = coeffs[:num_parents], coeff_rankings[:num_parents]
        self.opt.tell(value_batch)  # XNES

        # Check for reset and maybe reset
        stop_status = self.opt.check_stop(ranking_values) or self._check_restart(new_sols)
        if stop_status:
            new_elite = self.archive.sample_elites(1)
            new_theta, measures, obj = new_elite.solution_batch[0], new_elite.measures_batch[0], new_elite.objective_batch[0]
            log.debug(f'XNES is restarting with a new solution whose measures are {measures} and objective is {obj}')
            if self._normalize_obs:
                self.mean_agent_obs_normalizer.load_state_dict(metadata_batch[0]['obs_normalizer'])
            if self._normalize_returns:
                self._mean_agent_return_normalizer.load_state_dict(metadata_batch[0]['return_normalizer'])

            self._grad_opt.theta = new_theta
            self.opt = ExponentialES(self._num_coefficients,
                                     self.device,
                                     self._sigma0,
                                     self.opt_lambda,
                                     seed=self.opt_seed,
                                     initial_bounds=self._initial_bounds)
            self._ranker.reset(self, self.archive, self._rng)
            self._restarts += 1

        # Increase iteration counter.
        self._itrs += 1

        return stop_status
