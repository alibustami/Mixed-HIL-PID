import numpy as np

MIN_PARAM_VALUE = 1e-3  # guard against zeroing Ki/Kd


class DifferentialEvolutionOptimizer:
    """
    Differential Evolution (DE/rand/1/bin) with feasibility rules
    for explicit constraint handling.

    Expected eval_func signature:
        - returns fitness (float)   OR
        - returns (fitness, violation) where violation <= 0 means feasible.
    """
    def __init__(self, bounds, pop_size=10, mutation_factor=0.5, crossover_rate=0.7):
        self.bounds = np.array(bounds, dtype=float)
        self.bounds[:, 0] = np.maximum(self.bounds[:, 0], MIN_PARAM_VALUE)

        self.pop_size = int(pop_size)
        self.mutation_factor = float(mutation_factor)
        self.crossover_rate = float(crossover_rate)
        self.dim = len(bounds)

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
        Feasibility rules:
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
            # Mutation: pick a,b,c distinct and != i
            idxs = [idx for idx in range(self.pop_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]

            mutant = a + self.mutation_factor * (b - c)
            mutant = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])

            # Crossover (binomial)
            cross_points = np.random.rand(self.dim) < self.crossover_rate
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, self.population[i])

            # Selection (feasibility rules)
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

    def refine_search_space(self, center_candidate, shrink_factor=0.5):
        """
        Tie-to-Refine Logic: Shrinks bounds around a candidate.
        """
        center_candidate = np.array(center_candidate, dtype=float)
        shrink_factor = float(shrink_factor)

        current_range = self.bounds[:, 1] - self.bounds[:, 0]
        new_range = current_range * shrink_factor

        min_b = center_candidate - (new_range / 2.0)
        max_b = center_candidate + (new_range / 2.0)

        min_b = np.maximum(min_b, MIN_PARAM_VALUE)
        max_b = np.maximum(max_b, min_b + 1e-9)

        self.bounds = np.column_stack((min_b, max_b))

        # Reduce mutation to favor exploitation
        self.mutation_factor *= 0.8

        self.population = self._initialize_population()
        self.population[0] = np.clip(center_candidate, self.bounds[:, 0], self.bounds[:, 1])

        self.fitness_scores[:] = np.inf
        self.violations[:] = np.inf
        self.best_idx = -1

        print(f"[DE] Refine Mode Active. New Bounds: {self.bounds}")

    def expand_search_space(self, expand_factor=1.5):
        """
        Expands bounds if user rejects all options.
        """
        expand_factor = float(expand_factor)
        center = np.mean(self.bounds, axis=1)
        current_range = self.bounds[:, 1] - self.bounds[:, 0]
        new_range = current_range * expand_factor

        min_b = np.maximum(center - (new_range / 2.0), MIN_PARAM_VALUE)
        max_b = center + (new_range / 2.0)

        max_b = np.maximum(max_b, min_b + 1e-9)

        self.bounds = np.column_stack((min_b, max_b))

        self.mutation_factor = min(self.mutation_factor * 1.2, 1.0)

        self.population = self._initialize_population()
        self.fitness_scores[:] = np.inf
        self.violations[:] = np.inf
        self.best_idx = -1

        print(f"[DE] Search Space Expanded. Bounds: {self.bounds}")
