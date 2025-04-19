from dataclasses import dataclass
from math import *
import numpy as np
from scipy.stats import norm


@dataclass
class BlackScholes:
    sigma: float
    r: float = 0.0
    q: float = 0.0

    def _d1(self, s, t, k):
        return (np.log(s/k) + (self.r + 0.5*self.sigma**2)*t) / (self.sigma*np.sqrt(t))

    def _d2(self, s, t, k):
        return (np.log(s/k) + (self.r - 0.5*self.sigma**2)*t) / (self.sigma*np.sqrt(t))

    def call_price(self, spot, maturity, strike):
        return (spot*norm.cdf(self._d1(spot, maturity, strike)) 
                - np.exp(-self.r*maturity)*strike*norm.cdf(self._d2(spot, maturity, strike)))
    
    def put_price(self, spot, maturity, strike):
        return (np.exp(-self.r*maturity)*strike*norm.cdf(-self._d2(spot, maturity, strike)) 
                - spot*norm.cdf(-self._d1(spot, maturity, strike)))

    def simulate(self, s0, t, steps, paths, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        dt = t/steps
        dW = rng.normal(loc=(self.r - 0.5*self.sigma**2)*dt,
                        scale=self.sigma*sqrt(dt),
                        size=(steps, paths))
        logS = np.concatenate([np.zeros((1, paths)), np.cumsum(dW, axis=0)])
        
        return s0*np.exp(logS)
