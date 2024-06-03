import wandb
import numpy as np
from cma import CMAEvolutionStrategy
from baselines.PPGA.utils.utilities import log


class PyCMAEvolutionStrategy(CMAEvolutionStrategy):
    def __init__(self, x0, sigma0, popsize, seed, dtype):
        CMAEvolutionStrategy.__init__(self, x0, sigma0, {'CSA_squared': 'True',
                                                         'popsize': popsize,
                                                         'seed': seed})
        self.solution_dim = len(x0)
        self.dtype = dtype

    # def ask_cma(self, lower_bounds, upper_bounds, batch_size):
    #     remaining_inds = np.arange(batch_size)
    #     solutions = np.empty((batch_size, self.solution_dim), dtype=self.dtype)
    #     num_resamples = 0
    #     while len(remaining_inds) > 0:
    #         new_sols = self.ask(len(remaining_inds), self.mean, self.sigma)
    #         out_of_bounds = np.logical_or(
    #             new_sols < np.expand_dims(lower_bounds, axis=0),
    #             new_sols > np.expand_dims(upper_bounds, axis=0),
    #         )
    #         solutions[remaining_inds] = new_sols
    #         out_of_bounds_inds = np.where(np.any(out_of_bounds, axis=1))[0]
    #         remaining_inds = remaining_inds[out_of_bounds_inds]
    #         num_resamples += len(remaining_inds)
    #
    #     # wandb.log({'QD/Num Resamples': num_resamples})
    #     return np.asarray(solutions)

    def ask_cma(self, lower_bounds, upper_bounds, batch_size):
        solutions = self.ask(batch_size, self.mean, self.sigma)
        return np.asarray(solutions)

    def tell_cma(self, solutions, obj_values):
        self.tell(solutions, obj_values)
        # self.sigma = np.clip(self.sigma, 0.0, 2.0)

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
        if self.condition_number > 1e14:
            log.debug(f'{self.condition_number=} is too large. Restarting...')
            return True

        # area of distribution is too small
        area = self.sigma * np.sqrt(np.max(self.sm.eigenspectrum))
        if area < 1e-11:
            log.debug(f'{area=} of distribution is too small. Restarting...')
            return True

        # Fitness is too flat (only applies if there are at least 2 parents).
        if len(ranking_values) >= 2 and np.linalg.norm(ranking_values[0] - ranking_values[-1]) < 1e-12:
            log.debug(f'Fitness is too flat. Restarting...')
            return True

        # also check for any pycma restart conditions
        stop_status = self.stop(check=True)
        if stop_status != {}:
            log.debug(f"Pycma stop condition triggered. {stop_status=} Restarting...")
            return True

        return False



