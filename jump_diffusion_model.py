"""
Merton Jump-Diffusion model fitting via maximum likelihood.
Includes safeguards against extreme parameter estimates.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize

class MertonJumpDiffusion:
    def __init__(self, jump_threshold_std=2.5):
        self.jump_threshold_std = jump_threshold_std
        self.params = None
        self.fitted = False
        
    def fit(self, returns: np.ndarray):
        """Fit Merton jump-diffusion parameters via MLE with sensible bounds."""
        if len(returns) < 100:
            return False
        
        # Annualized initial estimates
        mu0 = np.mean(returns) * 252
        sigma0 = np.std(returns) * np.sqrt(252)
        
        # Identify jumps: returns beyond threshold
        threshold = self.jump_threshold_std * sigma0 / np.sqrt(252)
        jumps = returns[np.abs(returns) > threshold]
        
        # Reasonable bounds for lambda (jumps per year)
        if len(jumps) > 0:
            lambda0 = min(len(jumps) / len(returns) * 252, 10.0)  # cap at 10 jumps/year
            mu_j0 = np.clip(np.mean(jumps) * 252, -0.5, 0.5)      # daily jump mean capped at ±50%
            sigma_j0 = np.clip(np.std(jumps) * np.sqrt(252), 0.01, 0.5)
        else:
            lambda0 = 0.5
            mu_j0 = 0.0
            sigma_j0 = sigma0 * 0.5
        
        # Tightened bounds for stability
        bounds = [
            (-0.5, 0.5),       # mu (annualized drift)
            (0.01, 1.0),       # sigma (annualized volatility)
            (0.0, 10.0),       # lambda (jumps per year, capped)
            (-0.5, 0.5),       # mu_j (jump mean, capped)
            (0.01, 0.5)        # sigma_j (jump volatility)
        ]
        initial = [mu0, sigma0, lambda0, mu_j0, sigma_j0]
        # Clip initial to bounds
        for i, (low, high) in enumerate(bounds):
            initial[i] = np.clip(initial[i], low, high)
        
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
                prob = np.clip(prob, 0.0, 0.99)  # prevent prob > 1
                mixture = (1 - prob) * pdf_diff + prob * pdf_jump
                total_ll += -np.log(mixture + 1e-12)
            return total_ll
        
        try:
            result = minimize(neg_log_likelihood, initial, bounds=bounds, method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-6})
            if result.success:
                self.params = {
                    'mu': result.x[0],
                    'sigma': result.x[1],
                    'lambda': result.x[2],
                    'mu_j': result.x[3],
                    'sigma_j': result.x[4]
                }
                self.fitted = True
                return True
        except Exception as e:
            print(f"    Jump-diffusion optimization failed: {e}")
        
        # Fallback: simple historical mean and volatility
        self.params = {
            'mu': mu0,
            'sigma': sigma0,
            'lambda': 0.0,
            'mu_j': 0.0,
            'sigma_j': 0.0
        }
        self.fitted = True
        return True
    
    def forecast(self) -> dict:
        """Return jump-adjusted expected return and components."""
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
