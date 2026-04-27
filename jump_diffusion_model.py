"""
Merton Jump-Diffusion model – macro‑conditioned jump intensity.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize

class MertonJumpDiffusion:
    def __init__(self, jump_threshold_std=2.5, lambda_cap=10.0,
                 macro_conditioning=True, vix_avg=20.0):
        self.jump_threshold_std = jump_threshold_std
        self.lambda_cap = lambda_cap
        self.macro_conditioning = macro_conditioning
        self.vix_avg = vix_avg           # baseline VIX level
        self.params = None
        self.fitted = False

    def fit(self, returns: np.ndarray, macro_series: np.ndarray = None):
        if len(returns) < 100:
            return False

        mu0 = np.mean(returns) * 252
        sigma0 = np.std(returns) * np.sqrt(252)

        threshold = self.jump_threshold_std * sigma0 / np.sqrt(252)
        jumps = returns[np.abs(returns) > threshold]

        if len(jumps) > 0:
            lambda0 = min(len(jumps) / len(returns) * 252, self.lambda_cap)
            mu_j0 = np.clip(np.mean(jumps) * 252, -0.5, 0.5)
            sigma_j0 = np.clip(np.std(jumps) * np.sqrt(252), 0.01, 0.5)
        else:
            lambda0 = 0.5
            mu_j0 = 0.0
            sigma_j0 = sigma0 * 0.5

        bounds = [
            (-0.5, 0.5), (0.01, 1.0), (0.0, self.lambda_cap),
            (-0.5, 0.5), (0.01, 0.5)
        ]
        initial = [mu0, sigma0, lambda0, mu_j0, sigma_j0]
        for i, (low, high) in enumerate(bounds):
            initial[i] = np.clip(initial[i], low, high)

        # Macro‑conditioning: adjust λ₀ by recent VIX ratio
        if self.macro_conditioning and macro_series is not None and len(macro_series) > 20:
            recent_vix = np.mean(macro_series[-20:])
            if recent_vix > 0:
                scale = recent_vix / self.vix_avg
                initial[2] = np.clip(initial[2] * scale, 0.0, self.lambda_cap)

        def neg_log_likelihood(params):
            mu, sigma, lam, mu_j, sigma_j = params
            if sigma <= 0 or sigma_j <= 0 or lam < 0:
                return 1e10
            dt = 1/252
            total_ll = 0.0
            for r in returns:
                pdf_diff = stats.norm.pdf(r, loc=mu*dt, scale=sigma*np.sqrt(dt))
                pdf_jump = stats.norm.pdf(r, loc=mu*dt + mu_j*dt,
                                         scale=np.sqrt(sigma**2*dt + sigma_j**2*dt))
                prob = lam * dt
                prob = np.clip(prob, 0.0, 0.99)
                mixture = (1 - prob) * pdf_diff + prob * pdf_jump
                total_ll += -np.log(mixture + 1e-12)
            return total_ll

        try:
            result = minimize(neg_log_likelihood, initial, bounds=bounds,
                              method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-6})
            if result.success:
                self.params = {
                    'mu': result.x[0], 'sigma': result.x[1],
                    'lambda': result.x[2], 'mu_j': result.x[3], 'sigma_j': result.x[4]
                }
                self.fitted = True
                return True
        except Exception:
            pass

        self.params = {'mu': mu0, 'sigma': sigma0, 'lambda': 0.0,
                       'mu_j': 0.0, 'sigma_j': 0.0}
        self.fitted = True
        return True

    def forecast(self) -> dict:
        if not self.fitted:
            return {'expected_return': 0.0, 'jump_adjustment': 0.0}
        mu = self.params['mu']
        lam = self.params['lambda']
        mu_j = self.params['mu_j']
        jump_adjustment = lam * mu_j
        expected_return = mu + jump_adjustment
        return {
            'expected_return': expected_return,
            'diffusion_drift': mu,
            'jump_intensity': lam,
            'jump_mean': mu_j,
            'jump_adjustment': jump_adjustment
        }
