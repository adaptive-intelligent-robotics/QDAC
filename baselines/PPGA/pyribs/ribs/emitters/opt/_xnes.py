import torch
import numpy as np

from evotorch import Problem, SolutionBatch
from evotorch.algorithms import XNES
from baselines.PPGA.utils.utilities import log


class ExponentialES(XNES):
    def __init__(self,
                 solution_dim,
                 device,
                 sigma0,
                 popsize,
                 seed,
                 initial_bounds=None):
        problem = Problem(objective_sense='max', solution_length=solution_dim, initial_bounds=initial_bounds, device=device, seed=int(seed))
        XNES.__init__(self, problem, stdev_init=sigma0, popsize=popsize)
        self._first_iter = True

    def _step(self):
        pass

    @property
    def mu(self):
        return self._distribution.mu.detach().cpu().numpy()

    @property
    def A(self):
        return self._distribution.A.detach()

    def ask(self):
        if self._population is None:
            # init the population
            self._population = SolutionBatch(
                self.problem, popsize=self._popsize, device=self._distribution.device, empty=True
            )

        # sample from the pop
        self._distribution.sample(out=self._population.access_values(), generator=self.problem)

        # return the raw data to the dqd algorithm
        return self._population.access_values(keep_evals=True)[:].cpu().numpy()

    def tell(self, fitnesses):
        fitnesses = torch.from_numpy(fitnesses).to(self._distribution.device)
        samples = self._population.access_values(keep_evals=True)
        obj_sense = 'max'
        ranking_method = self._ranking_method
        gradients = self._distribution.compute_gradients(
            samples, fitnesses, objective_sense=obj_sense, ranking_method=ranking_method
        )
        self._update_distribution(gradients)

    def check_stop(self, ranking_values):
        """Checks if the optimization should stop and be reset.

        Tolerances come from CMA-ES.

        Args:
            ranking_values (np.ndarray): Array of objective values of the
                solutions, sorted in the same order that the solutions were
                sorted when passed to tell().
        Returns:
            True if any of the stopping conditions are satisfied.
        """

        # Fitness is too flat (only applies if there are at least 2 parents).
        if len(ranking_values) >= 2 and np.linalg.norm(ranking_values[0] - ranking_values[-1]) < 1e-12:
            log.debug(f'Fitness is too flat. Restarting...')
            return True

        return False





