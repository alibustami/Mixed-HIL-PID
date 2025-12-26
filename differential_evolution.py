import numpy as np

class DifferentialEvolutionOptimizer:
    def __init__(self, bounds, pop_size=10, mutation_factor=0.8, crossover_rate=0.7):
        """
        Args:
            bounds (list of tuple): [(min, max), ...] for Kp, Ki, Kd.
            pop_size (int): Number of candidates in population.
            mutation_factor (float): F, scales the difference vector (0-2).
            crossover_rate (float): CR, probability of recombination (0-1).
        """
        self.bounds = np.array(bounds)
        self.pop_size = pop_size
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.dim = len(bounds)
        
        # Initialize population
        self.population = self._initialize_population()
        self.fitness_scores = np.full(self.pop_size, np.inf)
        self.best_idx = -1

    def _initialize_population(self):
        """Generates random candidates within bounds."""
        pop = np.random.rand(self.pop_size, self.dim)
        min_b = self.bounds[:, 0]
        max_b = self.bounds[:, 1]
        return min_b + pop * (max_b - min_b)

    def evolve(self, fitness_func):
        """
        Runs one generation of evolution.
        Args:
            fitness_func (callable): Function taking PID params, returning scalar cost.
        Returns:
            best_candidate (np.array), best_fitness (float)
        """
        # Evaluate initial population if not done
        if np.isinf(self.fitness_scores).all():
            for i in range(self.pop_size):
                self.fitness_scores[i] = fitness_func(self.population[i])

        new_population = np.copy(self.population)

        for i in range(self.pop_size):
            # 1. Mutation: Select 3 distinct random agents (a, b, c) != i
            idxs = [idx for idx in range(self.pop_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            
            mutant = a + self.mutation_factor * (b - c)
            
            # Clip to bounds
            mutant = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])

            # 2. Crossover
            cross_points = np.random.rand(self.dim) < self.crossover_rate
            if not np.any(cross_points): 
                cross_points[np.random.randint(0, self.dim)] = True
            
            trial = np.where(cross_points, mutant, self.population[i])

            # 3. Selection
            trial_fitness = fitness_func(trial)
            if trial_fitness < self.fitness_scores[i]:
                new_population[i] = trial
                self.fitness_scores[i] = trial_fitness

        self.population = new_population
        self.best_idx = np.argmin(self.fitness_scores)
        
        return self.population[self.best_idx], self.fitness_scores[self.best_idx]

    def refine_search_space(self, center_candidate, shrink_factor=0.5):
        """
        Tie-to-Refine Logic: Shrinks bounds around a promising candidate.
        """
        current_range = self.bounds[:, 1] - self.bounds[:, 0]
        new_range = current_range * shrink_factor
        
        min_b = center_candidate - (new_range / 2)
        max_b = center_candidate + (new_range / 2)
        
        # Ensure we don't accidentally invert bounds or go negative if physics forbids it
        # Assuming PID values must be positive
        min_b = np.maximum(min_b, 0.0) 
        
        self.bounds = np.column_stack((min_b, max_b))
        
        # Reduce mutation to favor exploitation
        self.mutation_factor *= 0.8 
        print(f"[DE] Refine Mode Active. New Bounds: {self.bounds}")
        
        # Re-initialize portion of population to focus on this area
        self.population = self._initialize_population()
        # Keep the best one
        self.population[0] = center_candidate
        self.fitness_scores = np.full(self.pop_size, np.inf)

    def expand_search_space(self, expand_factor=1.5):
        """Expands bounds if user rejects all options."""
        center = np.mean(self.bounds, axis=1)
        current_range = self.bounds[:, 1] - self.bounds[:, 0]
        new_range = current_range * expand_factor
        
        min_b = np.maximum(center - (new_range / 2), 0)
        max_b = center + (new_range / 2)
        
        self.bounds = np.column_stack((min_b, max_b))
        self.mutation_factor = min(self.mutation_factor * 1.2, 1.0) # Increase mutation
        print(f"[DE] Search Space Expanded. Bounds: {self.bounds}")