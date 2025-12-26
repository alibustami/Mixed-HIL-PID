import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm

class BayesianOptimizer:
    def __init__(self, bounds):
        """
        Args:
            bounds (list of tuple): [(min, max), ...] for Kp, Ki, Kd.
        """
        self.bounds = np.array(bounds)
        self.kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10, alpha=1e-6)
        
        self.X_sample = []
        self.Y_sample = []
        self.dim = len(bounds)

    def propose_location(self, n_restarts=25):
        """
        Proposes the next PID set to try using Expected Improvement (EI).
        """
        if len(self.X_sample) == 0:
            # Random guess if no data
            return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

        # Find best current value (min implementation)
        best_y = np.min(self.Y_sample)

        def min_obj(X):
            # Minus EI because we minimize
            return -self._expected_improvement(X.reshape(1, -1), best_y)

        # Random search for the optimum of the acquisition function
        min_val = 1
        min_x = None
        
        for x0 in np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_restarts, self.dim)):
            res = min_obj(x0)
            if res < min_val:
                min_val = res
                min_x = x0
                
        return min_x if min_x is not None else np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

    def _expected_improvement(self, X, best_y, xi=0.01):
        """
        Calculates EI acquisition function.
        """
        mu, sigma = self.gp.predict(X, return_std=True)
        
        with np.errstate(divide='warn'):
            imp = best_y - mu - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
            
        return ei

    def update(self, X, y):
        """
        Update the internal Gaussian Process model with new data.
        """
        self.X_sample.append(X)
        self.Y_sample.append(y)
        self.gp.fit(np.array(self.X_sample), np.array(self.Y_sample))

    def refine_bounds(self, center_candidate, shrink_factor=0.5):
        """
        Refine logic for BO: focuses the acquisition search space.
        """
        current_range = self.bounds[:, 1] - self.bounds[:, 0]
        new_range = current_range * shrink_factor
        
        min_b = np.maximum(center_candidate - (new_range / 2), 0.0)
        max_b = center_candidate + (new_range / 2)
        
        self.bounds = np.column_stack((min_b, max_b))
        print(f"[BO] Refine Mode Active. New Bounds: {self.bounds}")

    def expand_bounds(self, expand_factor=1.5):
        """Expands bounds if user rejects."""
        center = np.mean(self.bounds, axis=1)
        current_range = self.bounds[:, 1] - self.bounds[:, 0]
        new_range = current_range * expand_factor
        
        min_b = np.maximum(center - (new_range / 2), 0.0)
        max_b = center + (new_range / 2)
        
        self.bounds = np.column_stack((min_b, max_b))