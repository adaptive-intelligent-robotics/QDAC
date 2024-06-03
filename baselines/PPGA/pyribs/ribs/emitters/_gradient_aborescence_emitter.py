"""Provides the GradientImprovementEmitter."""
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


class GradientAborescenceEmitter(DQDEmitterBase):
    """Generates solutions with a gradient arborescence, with coefficients
    parameterized by CMA-ES.

    This emitter originates in `Fontaine 2021
    <https://arxiv.org/abs/2106.03894>`_. It leverages the gradient information
    of the objective and measure functions, generating new solutions around a
    "solution point" using gradient aborescence with coefficients drawn from a
    Gaussian distribution. Based on how the solutions are ranked after being
    inserted into the archive (see ``ranker``), the solution point is updated
    with gradient ascent, and the distribution is updated with CMA-ES.

    Note that unlike non-gradient emitters, GradientAborescenceEmitter requires
    calling :meth:`ask_dqd` and :meth:`tell_dqd` (in this order) before calling
    :meth:`ask` and :meth:`tell` to communicate the gradient information to the
    emitter.

    Args:
        archive (ribs.archives.ArchiveBase): An archive to use when creating and
            inserting solutions. For instance, this can be
            :class:`ribs.archives.GridArchive`.
        x0 (np.ndarray): Initial solution.
        sigma0 (float): Initial step size / standard deviation.
        step_size (float): Step size for the gradient optimizer
        ranker (Callable or str): The ranker is a :class:`RankerBase` object
            that orders the solutions after they have been evaluated in the
            environment. This parameter may be a callable (e.g. a class or a
            lambda function) that takes in no parameters and returns an instance
            of :class:`RankerBase`, or it may be a full or abbreviated ranker
            name as described in :meth:`ribs.emitters.rankers.get_ranker`.
        selection_rule ("mu" or "filter"): Method for selecting parents in
            CMA-ES. With "mu" selection, the first half of the solutions will be
            selected as parents, while in "filter", any solutions that were
            added to the archive will be selected.
        restart_rule (int, "no_improvement", and "basic"): Method to use when
            checking for restarts. If given an integer, then the emitter will
            restart after this many iterations, where each iteration is a call
            to :meth:`tell`. With "basic", only the default CMA-ES convergence
            rules will be used, while with "no_improvement", the emitter will
            restart when none of the proposed solutions were added to the
            archive.
        grad_opt ("adam" or "gradient_ascent"): Gradient optimizer to use for
            the gradient ascent step of the algorithm. Defaults to `adam`.
        normalize_grad (bool): If true (default), then gradient infomation will
            be normalized. Otherwise, it will not be normalized.
        bounds (None or array-like): Bounds of the solution space. As suggested
            in `Biedrzycki 2020
            <https://www.sciencedirect.com/science/article/abs/pii/S2210650219301622>`_,
            solutions are resampled until they fall within these bounds.  Pass
            None to indicate there are no bounds. Alternatively, pass an
            array-like to specify the bounds for each dim. Each element in this
            array-like can be None to indicate no bound, or a tuple of
            ``(lower_bound, upper_bound)``, where ``lower_bound`` or
            ``upper_bound`` may be None to indicate no bound.
        batch_size (int): Number of solutions to return in :meth:`ask`. If not
            passed in, a batch size will be automatically calculated using the
            default CMA-ES rules. Note that `batch_size` **does not** include
            the number of solutions returned by :meth:`ask_dqd`, but also note
            that :meth:`ask_dqd` always returns one solution, i.e. the solution
            point.
        epsilon (float): For numerical stability, we add a small epsilon when
            normalizing gradients in :meth:`tell_dqd` -- refer to the
            implementation `here
            <../_modules/ribs/emitters/_gradient_aborescence_emitter.html#GradientAborescenceEmitter.tell_dqd>`_.
            Pass this parameter to configure that epsilon.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
    Raises:
        ValueError: If ``restart_rule`` is invalid.
    """

    def __init__(self,
                 archive,
                 x0,
                 sigma0,
                 step_size,
                 ranker="2imp",
                 selection_rule="filter",
                 restart_rule="no_improvement",
                 grad_opt="adam",
                 normalize_grad=True,
                 bounds=None,
                 coeff_bounds: Optional[List[tuple[float, float]]] = None,
                 batch_size=None,
                 epsilon=1e-8,
                 seed=None,
                 use_wandb=False,
                 normalize_obs=False,
                 normalize_rewards=False):
        self._epsilon = epsilon
        self._rng = np.random.default_rng(seed)
        self._x0 = np.array(x0, dtype=archive.dtype)
        self._sigma0 = sigma0
        self._normalize_grads = normalize_grad
        self._jacobian_batch = None
        self._grad_coeff_bounds = coeff_bounds
        (self._grad_coeff_lower_bounds, self._grad_coeff_upper_bounds) = \
            self._process_bounds(coeff_bounds, archive.measure_dim + 1, archive.dtype)
        DQDEmitterBase.__init__(
            self,
            archive,
            len(self._x0),
            bounds,
        )

        self._ranker = _get_ranker(ranker)
        self._ranker.reset(self, archive, self._rng)

        # Initialize gradient optimizer
        self._grad_opt = None
        if grad_opt == "adam":
            self._grad_opt = AdamOpt(self._x0, step_size)
        elif grad_opt == "gradient_ascent":
            self._grad_opt = GradientAscentOpt(self._x0, step_size)
        else:
            raise ValueError(f"Invalid Gradient Ascent Optimizer {grad_opt}")

        if selection_rule not in ["mu", "filter"]:
            raise ValueError(f"Invalid selection_rule {selection_rule}")
        self._selection_rule = selection_rule

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
        self._normalize_rewards = normalize_rewards
        self._mean_agent_obs_normalizer = None
        self._mean_agent_reward_normalizer = None

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
    def mean_agent_reward_normalizer(self):
        return self._mean_agent_reward_normalizer

    @mean_agent_reward_normalizer.setter
    def mean_agent_reward_normalizer(self, rew_normalizer):
        self._mean_agent_reward_normalizer = rew_normalizer

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
                self.mean_agent_obs_normalizer = metadata_batch[0]['obs_normalizer']
            if self._normalize_rewards:
                self._mean_agent_reward_normalizer = metadata_batch[0]['reward_normalizer']

            self._grad_opt.reset(new_theta)
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
