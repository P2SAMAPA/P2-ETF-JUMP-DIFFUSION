"""
Merton Jump-Diffusion model fitting via maximum likelihood.
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
        """Fit Merton jump-diffusion parameters via MLE."""
        if len(returns) < 100:
            return False
        
        # Initial estimates
        mu0 = np.mean(returns) * 252
        sigma0 = np.std(returns) * np.sqrt(252)
        
        # Identify jumps: returns beyond threshold
        threshold = self.jump_threshold_std * sigma0 / np.sqrt(252)
        jumps = returns[np.abs(returns) > threshold]
        lambda0 = len(jumps) / len(returns) * 252 if len(jumps) > 0 else 1.0
        
        if len(jumps) > 0:
            mu_j0 = np.mean(jumps) * 252
            sigma_j0 = np.std(jumps) * np.sqrt(252)
        else:
            mu_j0 = 0.0
            sigma_j0 = sigma0
        
        def neg_log_likelihood(params):
            mu, sigma, lam, mu_j, sigma_j = params
            if sigma <= 0 or sigma_j <= 0 or lam < 0 or lam > 252*10:
                return 1e10
            
            dt = 1/252
            total_ll = 0.0
            for r in returns:
                pdf_diff = stats.norm.pdf(r, loc=mu*dt, scale=sigma*np.sqrt(dt))
                pdf_jump = stats.norm.pdf(r, loc=mu*dt + mu_j*dt, 
                                         scale=np.sqrt(sigma**2*dt + sigma_j**2*dt))
                prob = lam * dt
                if prob > 1:
                    prob = 1.0
                mixture = (1 - prob) * pdf_diff + prob * pdf_jump
                total_ll += -np.log(mixture + 1e-10)
            return total_ll
        
        initial = [mu0, sigma0, lambda0, mu_j0, sigma_j0]
        bounds = [(-1, 1), (1e-6, 2), (0, 252*5), (-2, 2), (1e-6, 2)]
        
        result = minimize(neg_log_likelihood, initial, bounds=bounds, method='L-BFGS-B')
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
        return False
    
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
