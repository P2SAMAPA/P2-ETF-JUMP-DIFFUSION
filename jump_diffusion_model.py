"""
Merton Jump-Diffusion model – macro-conditioned jump intensity.

FIXES:
  Bug 1 — Wrong jump likelihood: mu_j and sigma_j were scaled by dt as if
           jumps were diffusions. A jump is a discrete one-time event; its
           mean and variance do NOT scale with dt. Fixed the log-likelihood
           mixture to use the correct Merton formulation with k=0,1,2,3
           Poisson terms instead of the broken two-component approximation.

  Bug 2 — Two-component mixture replaced with proper Poisson sum (k=0..3).
           The original (1-λdt)*diffusion + λdt*jump is only valid for
           λdt<<1 and misses compound jumps entirely.

  Bug 6 — Expected return ignored the Itô correction (σ²/2) and the jump
           variance correction (λ*σ_j²/2). For log-normal prices the simple
           expected return is μ + σ²/2 + λ*(mu_j + σ_j²/2). Added both.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from math import factorial, exp


class MertonJumpDiffusion:
    def __init__(self, jump_threshold_std=2.5, lambda_cap=10.0,
                 macro_conditioning=True, vix_avg=20.0):
        self.jump_threshold_std = jump_threshold_std
        self.lambda_cap = lambda_cap
        self.macro_conditioning = macro_conditioning
        self.vix_avg = vix_avg
        self.params = None
        self.fitted = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _poisson_mixture_pdf(r: float, mu: float, sigma: float,
                              lam: float, mu_j: float, sigma_j: float,
                              dt: float, n_terms: int = 4) -> float:
        """
        FIX Bug 1 & 2: correct Merton (1976) log-likelihood mixture.

        Each daily return is a sum of:
          - a Gaussian diffusion component: N(μ·dt, σ²·dt)
          - k independent jumps drawn from N(mu_j, sigma_j²)
            where k ~ Poisson(λ·dt)

        So the conditional density given k jumps is:
          N(r; μ·dt + k·mu_j,  σ²·dt + k·sigma_j²)

        Note:
          - mu_j  is the jump SIZE (not annualised rate × dt)
          - sigma_j is the jump SIZE std (not scaled by dt)
          - Only the diffusion variance σ²·dt scales with dt
        """
        total = 0.0
        lam_dt = lam * dt
        poisson_weight = exp(-lam_dt)   # starts at k=0 term

        for k in range(n_terms):
            if k > 0:
                poisson_weight *= lam_dt / k   # incremental Poisson weight

            loc   = mu * dt + k * mu_j
            scale = np.sqrt(sigma ** 2 * dt + k * sigma_j ** 2 + 1e-12)
            total += poisson_weight * stats.norm.pdf(r, loc=loc, scale=scale)

        return total

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, returns: np.ndarray, macro_series: np.ndarray = None):
        if len(returns) < 100:
            return False

        mu0    = np.mean(returns) * 252
        sigma0 = np.std(returns) * np.sqrt(252)

        # Identify empirical jumps (daily threshold = 2.5 * daily_std)
        daily_std = np.std(returns)
        threshold = self.jump_threshold_std * daily_std
        jumps = returns[np.abs(returns) > threshold]

        if len(jumps) > 0:
            lambda0 = min(len(jumps) / len(returns) * 252, self.lambda_cap)
            # FIX Bug 1: mu_j is the jump SIZE (keep as daily magnitude, not *252)
            mu_j0    = np.clip(np.mean(jumps), -0.3, 0.3)
            sigma_j0 = np.clip(np.std(jumps),  0.005, 0.3)
        else:
            lambda0  = 0.5
            mu_j0    = 0.0
            sigma_j0 = daily_std * 2.0

        # Macro-conditioning: scale λ₀ by recent VIX ratio
        if (self.macro_conditioning and macro_series is not None
                and len(macro_series) > 20):
            recent_vix = np.mean(macro_series[-20:])
            if recent_vix > 0 and self.vix_avg > 0:
                scale = recent_vix / self.vix_avg
                lambda0 = np.clip(lambda0 * scale, 0.0, self.lambda_cap)

        bounds = [
            (-0.5, 0.5),            # mu  (annualised drift)
            (0.005, 1.0),           # sigma (annualised diffusion vol)
            (0.0, self.lambda_cap), # lambda (jump intensity per year)
            (-0.3, 0.3),            # mu_j  (jump size, NOT annualised)
            (0.005, 0.3),           # sigma_j (jump size std)
        ]
        initial = [mu0, sigma0, lambda0, mu_j0, sigma_j0]
        for i, (lo, hi) in enumerate(bounds):
            initial[i] = float(np.clip(initial[i], lo, hi))

        dt = 1.0 / 252

        def neg_log_likelihood(params):
            mu, sigma, lam, mu_j, sigma_j = params
            if sigma <= 0 or sigma_j <= 0 or lam < 0:
                return 1e10
            ll = 0.0
            for r in returns:
                pdf = self._poisson_mixture_pdf(
                    r, mu, sigma, lam, mu_j, sigma_j, dt, n_terms=4)
                ll += -np.log(pdf + 1e-300)
            return ll

        try:
            result = minimize(
                neg_log_likelihood, initial, bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-7, 'gtol': 1e-5}
            )
            if result.success or result.fun < neg_log_likelihood(initial):
                self.params = {
                    'mu':      result.x[0],
                    'sigma':   result.x[1],
                    'lambda':  result.x[2],
                    'mu_j':    result.x[3],
                    'sigma_j': result.x[4],
                }
                self.fitted = True
                return True
        except Exception:
            pass

        # Fallback: moment-matched parameters
        self.params = {
            'mu': mu0, 'sigma': sigma0,
            'lambda': lambda0, 'mu_j': mu_j0, 'sigma_j': sigma_j0,
        }
        self.fitted = True
        return True

    def forecast(self) -> dict:
        if not self.fitted:
            return {'expected_return': 0.0, 'jump_adjustment': 0.0}

        mu      = self.params['mu']
        sigma   = self.params['sigma']
        lam     = self.params['lambda']
        mu_j    = self.params['mu_j']
        sigma_j = self.params['sigma_j']

        # FIX Bug 6: Itô correction for log-normal prices.
        # The fitted μ is the LOG-return drift. The expected SIMPLE return is:
        #   E[R] = μ + σ²/2  (diffusion Itô term)
        #        + λ*(mu_j + σ_j²/2)  (jump contribution incl. Jensen term)
        # mu_j here is already a jump SIZE (daily), so annualise by *252.
        ito_diffusion   = 0.5 * sigma ** 2
        jump_adjustment = lam * (mu_j * 252 + 0.5 * sigma_j ** 2 * 252)
        expected_return = mu + ito_diffusion + jump_adjustment

        return {
            'expected_return':  float(expected_return),
            'diffusion_drift':  float(mu),
            'ito_correction':   float(ito_diffusion),
            'jump_intensity':   float(lam),
            'jump_mean':        float(mu_j),
            'jump_adjustment':  float(jump_adjustment),
        }
