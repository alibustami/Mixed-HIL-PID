"""Bayesian Optmizer Module."""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm

MIN_PARAM_VALUE = 1e-3  # guard against zeroing Ki/Kd


class ConstrainedBayesianOptimizer:
    """
    Constrained Bayesian Optimization for black-box objective f(x) with
    explicit inequality constraint g(x) <= 0.

    PAPER-CONVENTION:
      - `global_bounds` are HARD domain limits (never exceeded)
      - `bounds` are the current adaptive search window (refine/expand),
        always clipped to global_bounds.

    Acquisition: Expected Constrained Improvement (EIC) ≈ EI(x) * PoF(x),
    where PoF is P(g(x) <= 0). :contentReference[oaicite:2]{index=2}
    """

    def __init__(self, bounds, pof_min=0.95):
        self.global_bounds = np.array(bounds, dtype=float)
        self.global_bounds[:, 0] = np.maximum(self.global_bounds[:, 0], MIN_PARAM_VALUE)

        self.bounds = self.global_bounds.copy()
        self.dim = len(bounds)
        self.pof_min = float(pof_min)

        kernel = C(1.0, (1e-3, 1e3)) * RBF(
            length_scale=np.ones(self.dim),
            length_scale_bounds=(1e-2, 1e2),
        )

        self.gp_f = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            alpha=1e-6,
            normalize_y=True,
        )
        self.gp_g = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            alpha=1e-6,
            normalize_y=True,
        )

        self.X_raw = []
        self.Y = []
        self.G = []

        self._f_is_fit = False
        self._g_is_fit = False

    def _to_unit(self, X_raw):
        X = np.asarray(X_raw, dtype=float)
        min_b = self.global_bounds[:, 0]
        span = self.global_bounds[:, 1] - min_b
        return (X - min_b) / (span + 1e-12)

    def _from_unit(self, U):
        U = np.asarray(U, dtype=float)
        min_b = self.global_bounds[:, 0]
        span = self.global_bounds[:, 1] - min_b
        return min_b + U * span

    def _search_bounds_unit(self):
        u_min = self._to_unit(self.bounds[:, 0])
        u_max = self._to_unit(self.bounds[:, 1])
        return u_min, u_max

    def update(self, X, y, g):
        x = np.array(X, dtype=float).reshape(-1)
        y = float(y)
        g = float(g)

        # HARD clamp to global domain
        x = np.clip(x, self.global_bounds[:, 0], self.global_bounds[:, 1])

        self.X_raw.append(x)
        self.Y.append(y)
        self.G.append(g)

        self._fit_models()

    def _fit_models(self):
        X_unit = self._to_unit(np.array(self.X_raw))
        Y = np.array(self.Y, dtype=float)
        G = np.array(self.G, dtype=float)

        try:
            self.gp_g.fit(X_unit, G)
            self._g_is_fit = True
        except Exception:
            self._g_is_fit = False

        feas = G <= 0.0
        try:
            if np.sum(feas) >= 2:
                self.gp_f.fit(X_unit[feas], Y[feas])
            else:
                self.gp_f.fit(X_unit, Y)
            self._f_is_fit = True
        except Exception:
            self._f_is_fit = False

    def _expected_improvement(self, u, best_y, xi=0.01):
        if not self._f_is_fit:
            return 0.0
        u = np.array(u, dtype=float).reshape(1, -1)
        mu, sigma = self.gp_f.predict(u, return_std=True)
        mu = float(mu[0])
        sigma = float(max(sigma[0], 1e-12))
        imp = best_y - mu - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        return float(max(0.0, ei))

    def _probability_feasible(self, u):
        if not self._g_is_fit:
            return 0.0
        u = np.array(u, dtype=float).reshape(1, -1)
        mu, sigma = self.gp_g.predict(u, return_std=True)
        mu = float(mu[0])
        sigma = float(max(sigma[0], 1e-12))
        z = (0.0 - mu) / sigma
        return float(norm.cdf(z))

    def _eic(self, u, best_y, xi=0.01):
        pof = self._probability_feasible(u)
        if pof <= 0.0:
            return 0.0
        return self._expected_improvement(u, best_y, xi=xi) * pof

    def propose_location(self, n_candidates=2048, xi=0.01):
        if len(self.X_raw) == 0 or not self._g_is_fit:
            return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.dim,))

        Y = np.array(self.Y, dtype=float)
        G = np.array(self.G, dtype=float)
        feas = G <= 0.0
        have_feasible = bool(np.any(feas))

        u_min, u_max = self._search_bounds_unit()
        U = np.random.uniform(u_min, u_max, size=(int(n_candidates), self.dim))

        if not have_feasible:
            pofs = np.array([self._probability_feasible(u) for u in U], dtype=float)
            best_u = U[int(np.argmax(pofs))]
            return self._clip_to_bounds(self._from_unit(best_u))

        best_y = float(np.min(Y[feas]))

        best_acq = -np.inf
        best_u = None

        for u in U:
            pof = self._probability_feasible(u)
            if pof < self.pof_min:
                continue
            acq = self._eic(u, best_y, xi=xi)
            if acq > best_acq:
                best_acq = acq
                best_u = u

        if best_u is None:
            for u in U:
                acq = self._eic(u, best_y, xi=xi)
                if acq > best_acq:
                    best_acq = acq
                    best_u = u

        if best_u is None:
            return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.dim,))

        return self._clip_to_bounds(self._from_unit(best_u))

    def _clip_to_bounds(self, x_raw):
        x_raw = np.array(x_raw, dtype=float).reshape(-1)
        x_raw = np.clip(x_raw, self.bounds[:, 0], self.bounds[:, 1])
        # HARD clamp also to global (redundant but explicit)
        x_raw = np.clip(x_raw, self.global_bounds[:, 0], self.global_bounds[:, 1])
        return x_raw

    def best_feasible(self):
        if len(self.X_raw) == 0:
            return None
        Y = np.array(self.Y, dtype=float)
        G = np.array(self.G, dtype=float)
        feas = G <= 0.0
        if not np.any(feas):
            return None
        idxs = np.where(feas)[0]
        best_idx = int(idxs[np.argmin(Y[feas])])
        return np.array(self.X_raw[best_idx], dtype=float), float(Y[best_idx]), float(G[best_idx])

    def refine_bounds(self, center_candidate, shrink_factor=0.5):
        center = np.array(center_candidate, dtype=float).reshape(-1)
        shrink_factor = float(shrink_factor)

        current_range = self.bounds[:, 1] - self.bounds[:, 0]
        new_range = current_range * shrink_factor

        min_b = center - (new_range / 2.0)
        max_b = center + (new_range / 2.0)

        min_b = np.maximum(min_b, self.global_bounds[:, 0])
        max_b = np.minimum(max_b, self.global_bounds[:, 1])
        max_b = np.maximum(max_b, min_b + 1e-9)

        self.bounds = np.column_stack((min_b, max_b))
        print(f"[BO] Refine Mode Active. New Bounds: {self.bounds}")

    def expand_bounds(self, expand_factor=1.5):
        expand_factor = float(expand_factor)

        center = np.mean(self.bounds, axis=1)
        current_range = self.bounds[:, 1] - self.bounds[:, 0]
        new_range = current_range * expand_factor

        min_b = center - (new_range / 2.0)
        max_b = center + (new_range / 2.0)

        min_b = np.maximum(min_b, self.global_bounds[:, 0])
        max_b = np.minimum(max_b, self.global_bounds[:, 1])
        max_b = np.maximum(max_b, min_b + 1e-9)

        self.bounds = np.column_stack((min_b, max_b))
        print(f"[BO] Search Space Expanded. New Bounds: {self.bounds}")

    def nudge_with_preference(self, preferred, preferred_cost, other_cost, preferred_violation, strength=0.2):
        preferred_violation = float(preferred_violation)
        if preferred_violation > 0.0:
            return

        preferred = np.array(preferred, dtype=float).reshape(-1)
        gap = abs(float(other_cost) - float(preferred_cost))
        pseudo_cost = float(preferred_cost) - (strength * gap if gap > 0 else strength * 0.01)

        self.update(preferred, pseudo_cost, preferred_violation)
        print(f"[BO] Preference nudge applied. Gap={gap:.4f}, pseudo_cost={pseudo_cost:.4f}")