"""Differntial Evolution optimizer module."""

import numpy as np

MIN_PARAM_VALUE = 1e-3  # guard against zeroing Ki/Kd


class DifferentialEvolutionOptimizer:
    """
    Differential Evolution (DE/rand/1/bin) with feasibility rules
    for explicit constraint handling.

    IMPORTANT (for your paper):
      - `global_bounds` are HARD domain limits (never exceeded).
      - `bounds` are the current adaptive search window, always clipped to global_bounds.

    Expected eval_func signature:
        returns fitness (float)
        OR returns (fitness, violation) where violation <= 0 means feasible.
    """

    def __init__(self, bounds, pop_size=10, mutation_factor=0.5, crossover_rate=0.7):
        global_b = np.array(bounds, dtype=float)
        global_b[:, 0] = np.maximum(global_b[:, 0], MIN_PARAM_VALUE)

        self.global_bounds = global_b
        self.bounds = self.global_bounds.copy()  # adaptive window (refine/expand) inside global

        self.pop_size = int(pop_size)
        self.mutation_factor = float(mutation_factor)
        self.crossover_rate = float(crossover_rate)
        self.dim = int(self.global_bounds.shape[0])

        self.population = self._initialize_population()
        self.fitness_scores = np.full(self.pop_size, np.inf, dtype=float)
        self.violations = np.full(self.pop_size, np.inf, dtype=float)
        self.best_idx = -1

    def _initialize_population(self):
        pop = np.random.rand(self.pop_size, self.dim)
        min_b = self.bounds[:, 0]
        max_b = self.bounds[:, 1]
        return min_b + pop * (max_b - min_b)

    @staticmethod
    def _as_fit_viol(result):
        """Normalize eval_func result to (fitness, violation)."""
        if isinstance(result, (tuple, list)):
            if len(result) >= 2:
                return float(result[0]), float(result[1])
            if len(result) == 1:
                return float(result[0]), 0.0
        return float(result), 0.0

    @staticmethod
    def _is_better(fit_a, viol_a, fit_b, viol_b):
        """
        Feasibility rules (Deb-style):
          1) feasible beats infeasible
          2) among feasible: lower fitness wins
          3) among infeasible: lower violation wins
        """
        a_feas = viol_a <= 0.0
        b_feas = viol_b <= 0.0

        if a_feas and not b_feas:
            return True
        if b_feas and not a_feas:
            return False
        if a_feas and b_feas:
            return fit_a < fit_b
        return viol_a < viol_b

    def evolve(self, eval_func):
        """
        Runs one generation of DE evolution with feasibility-aware selection.

        Returns:
            best_candidate (np.ndarray),
            best_fitness (float),
            best_violation (float)
        """
        # Evaluate initial population if needed
        if np.isinf(self.fitness_scores).all():
            for i in range(self.pop_size):
                fit, viol = self._as_fit_viol(eval_func(self.population[i]))
                self.fitness_scores[i] = fit
                self.violations[i] = viol

        new_population = np.copy(self.population)

        for i in range(self.pop_size):
            idxs = [idx for idx in range(self.pop_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]

            mutant = a + self.mutation_factor * (b - c)
            mutant = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])

            cross_points = np.random.rand(self.dim) < self.crossover_rate
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, self.population[i])

            trial_fit, trial_viol = self._as_fit_viol(eval_func(trial))

            if self._is_better(trial_fit, trial_viol, self.fitness_scores[i], self.violations[i]):
                new_population[i] = trial
                self.fitness_scores[i] = trial_fit
                self.violations[i] = trial_viol

        self.population = new_population
        self._update_best_idx()

        return (
            self.population[self.best_idx].copy(),
            float(self.fitness_scores[self.best_idx]),
            float(self.violations[self.best_idx]),
        )

    def _update_best_idx(self):
        feas = self.violations <= 0.0
        if np.any(feas):
            masked = np.where(feas, self.fitness_scores, np.inf)
            self.best_idx = int(np.argmin(masked))
        else:
            self.best_idx = int(np.argmin(self.violations))

    def best_scores(self):
        """Return (best_fitness, best_violation) using feasibility-aware best."""
        if self.best_idx < 0:
            self._update_best_idx()
        return float(self.fitness_scores[self.best_idx]), float(self.violations[self.best_idx])

    def inject_candidate(self, candidate, eval_func=None, fitness=None, violation=None, protect_best=True):
        """
        Inject a candidate into the population and keep internal score arrays consistent.

        Provide either:
          - (fitness, violation) directly, OR
          - eval_func to compute them.

        Returns:
            (idx, fitness, violation)
        """
        cand = np.array(candidate, dtype=float).reshape(-1)
        cand = np.clip(cand, self.bounds[:, 0], self.bounds[:, 1])
        # bounds are always inside global_bounds, but keep it explicit:
        cand = np.clip(cand, self.global_bounds[:, 0], self.global_bounds[:, 1])

        if fitness is None or violation is None:
            if eval_func is None:
                raise ValueError("inject_candidate requires either (fitness, violation) or eval_func.")
            fitness, violation = self._as_fit_viol(eval_func(cand))
        else:
            fitness = float(fitness)
            violation = float(violation)

        if self.pop_size <= 1:
            idx = 0
        else:
            protected = self.best_idx if (protect_best and self.best_idx >= 0) else None
            choices = [i for i in range(self.pop_size) if i != protected]
            idx = int(np.random.choice(choices))

        self.population[idx] = cand
        self.fitness_scores[idx] = fitness
        self.violations[idx] = violation
        self._update_best_idx()

        return idx, float(fitness), float(violation)

    def reset_bounds_to_global(self):
        """Reset adaptive search window to the hard global bounds."""
        self.bounds = self.global_bounds.copy()

    def refine_search_space(self, center_candidate, shrink_factor=0.5):
        """
        Shrinks bounds around a candidate (still clamped to global bounds).
        """
        center = np.array(center_candidate, dtype=float).reshape(-1)
        shrink_factor = float(shrink_factor)

        current_range = self.bounds[:, 1] - self.bounds[:, 0]
        new_range = current_range * shrink_factor

        min_b = center - (new_range / 2.0)
        max_b = center + (new_range / 2.0)

        # HARD clamp to global bounds
        min_b = np.maximum(min_b, self.global_bounds[:, 0])
        max_b = np.minimum(max_b, self.global_bounds[:, 1])
        max_b = np.maximum(max_b, min_b + 1e-9)

        self.bounds = np.column_stack((min_b, max_b))

        # Reduce mutation to favor exploitation
        self.mutation_factor *= 0.8

        # Reinitialize population inside refined bounds
        self.population = self._initialize_population()
        self.population[0] = np.clip(center, self.bounds[:, 0], self.bounds[:, 1])

        self.fitness_scores[:] = np.inf
        self.violations[:] = np.inf
        self.best_idx = -1

        print(f"[DE] Refine Mode Active. New Bounds (clipped to global): {self.bounds}")

    def expand_search_space(self, expand_factor=1.5):
        """
        Expands the adaptive search window *within* the hard global bounds.

        If already at global bounds, this becomes a "diversify" step:
          - increase mutation
          - restart population (still inside global)
        """
        expand_factor = float(expand_factor)

        center = np.mean(self.bounds, axis=1)
        current_range = self.bounds[:, 1] - self.bounds[:, 0]
        new_range = current_range * expand_factor

        min_b = center - (new_range / 2.0)
        max_b = center + (new_range / 2.0)

        # HARD clamp to global bounds
        min_b = np.maximum(min_b, self.global_bounds[:, 0])
        max_b = np.minimum(max_b, self.global_bounds[:, 1])
        max_b = np.maximum(max_b, min_b + 1e-9)

        prev_bounds = self.bounds.copy()
        self.bounds = np.column_stack((min_b, max_b))

        # Increase mutation to favor exploration (cap at 1.0)
        self.mutation_factor = min(self.mutation_factor * 1.2, 1.0)

        # Restart population inside (expanded or global) bounds
        self.population = self._initialize_population()
        self.fitness_scores[:] = np.inf
        self.violations[:] = np.inf
        self.best_idx = -1

        changed = not np.allclose(prev_bounds, self.bounds)
        if changed:
            print(f"[DE] Search Space Expanded (clipped to global). New Bounds: {self.bounds}")
        else:
            print(f"[DE] Search Space at GLOBAL bounds; diversification restart. Bounds: {self.bounds}")
