from dataclasses import dataclass

import numpy as np
import scipy.stats as st
from scipy import optimize
from scipy.interpolate import interp1d


@dataclass
class SVI:  
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_var(self, log_moneyness):
        return self.a + self.b*(self.rho*(log_moneyness-self.m) + np.sqrt((log_moneyness-self.m)**2 + self.sigma**2))
    
    def implied_vol(self, forward_price, maturity, strike):
        log_moneyness = np.log(strike/forward_price)
        return np.sqrt(self.total_var(log_moneyness)/maturity)
    
    def __call__(self, log_moneyness):
        return self.total_var(log_moneyness)
    
    def cdf(self, s0, s):
        x = np.log(s/s0)
        w = self.total_var(x)
        theta = np.sqrt(w)
        d1 = -x/theta + theta/2
        d2 = d1 - theta
        wp = self.b*(self.rho + (x-self.m) / np.sqrt((x-self.m)**2 + self.sigma**2))
        return st.norm.pdf(d1)*s0/s * 0.5*wp/theta - st.norm.cdf(d2) + 1
    
    def pdf(self, s0, s):
        x = np.log(s/s0)
        w = self.total_var(x)
        theta = np.sqrt(w)
        d1 = -x/theta + theta/2
        wp = self.b*(self.rho + (x-self.m) / np.sqrt((x-self.m)**2 + self.sigma**2))
        wpp = self.b*(1/np.sqrt((x-self.m)**2 + self.sigma**2) -
                      (x-self.m)**2/((x-self.m)**2 + self.sigma**2)**1.5)
        return s0*st.norm.pdf(d1)/(theta*s**2)*((1-0.5*x*wp/w)**2 - 0.25*wp**2*(1/w+0.25) + 0.5*wpp)

    @staticmethod
    def _calibrate_adc(x, w, m, sigma, no_arbitrage):
        """Calibrates the raw parameters `a, d, c` given `m, sigma`.

        Args:
            x (array): Log-moneyness.
            w (array): Total implied variances.
            m (float): Parameter `m` of the model.
            sigma (float): Parameter `sigma` of the model.
            no_arbitrage (bool): If True, imposes the no-arbitrage constraints on the parameters.
                This results in a much slower calibration.

        Returns:
            Tuple `((a, d, c), f)` where `a, d, c` are the calibrated parameters and `f` is the
            minimum of the objective function.
        """
        if no_arbitrage:
            # Objective function; p = (a, d, c)
            def f(p):
                return 0.5*np.linalg.norm(
                    p[0] + p[1]*(x-m)/sigma + p[2]*np.sqrt(((x-m)/sigma)**2+1) -
                    w)**2

            # Gradient of the objective function
            def fprime(p):
                v1 = (x-m)/sigma
                v2 = np.sqrt(((x-m)/sigma)**2+1)
                v = p[0] + p[1]*v1 + p[2]*v2 - w
                return (np.sum(v), np.dot(v1, v), np.dot(v2, v))

            res = optimize.minimize(
                f,
                x0=(np.max(w)/2, 0, 2*sigma),
                method="SLSQP",
                jac=fprime,
                bounds=[(None, np.max(w)), (None, None), (0, 4*sigma)],
                constraints=[
                    {'type': 'ineq',
                        'fun': lambda p: p[2]-p[1],
                        'jac': lambda _: (0, -1, 1)},
                    {'type': 'ineq',
                        'fun': lambda p: p[2]+p[1],
                        'jac': lambda _: (0, 1, 1)},
                    {'type': 'ineq',
                        'fun': lambda p: 4*sigma - p[2]-p[1],
                        'jac': lambda _: (0, -1, -1)},
                    {'type': 'ineq',
                        'fun': lambda p: p[1]+4*sigma-p[2],
                        'jac': lambda _: (0, 1, -1)}])
            return res.x, res.fun
        else:
            # The parameters (a, d, c) are found by the least squares method
            X = np.stack([np.ones_like(x), (x-m)/sigma, np.sqrt(((x-m)/sigma)**2+1)], axis=1)
            # p is the minimizer, s[0] is the sum of squared residuals
            p, s, _, _ = np.linalg.lstsq(X, w, rcond=None)
            return p, s[0]

    @classmethod
    def calibrate(cls, x, w, min_sigma=1e-4, max_sigma=10, no_arbitrage=False,
                  method="differential_evolution"):
        """Calibrates the parameters of the model.

        Args:
            x (array): Array of log-moneynesses
            w (array): Array of total implied variances.
            min_sigma (float): Left bound for the value of `sigma`.
            max_sigma (float): Right bound for the value of `sigma`.
            no_arbitrage (bool): If True, imposes the no-arbitrage constraints on the parameters.
                This results in a much slower calibration.
            method (str): Method used for minimization. Must be a name of a global optimization 
                method from `scipy.optimize` module.

        Returns:
            An instance of the class with the calibrated parameters.
        """
        bounds = [(min(x), max(x)), (min_sigma, max_sigma)]
        res = optimize.differential_evolution(#__dict__[method](
            lambda q: cls._calibrate_adc(x, w, q[0], q[1], no_arbitrage)[1],  # q=(m, sigma)
            bounds=bounds)
        m, sigma = res.x
        a, d, c = cls._calibrate_adc(x, w, m, sigma, no_arbitrage)[0]
        rho = d/c
        b = c/sigma
        return cls(a, b, rho, m, sigma)
