from dataclasses import dataclass
import math
import numpy as np
from scipy.stats import norm


@dataclass
class BlackScholes:
    s0: float
    sigma: float
    r: float = 0.0

    def d1(self, maturity, strike):
        return (np.log(self.s0/strike) + (self.r + 0.5*self.sigma**2)*maturity) / (self.sigma*np.sqrt(maturity))

    def d2(self, maturity, strike):
        return self.d1(maturity, strike) - self.sigma*np.sqrt(maturity)

    def call_price(self, maturity, strike):
        return (self.s0*norm.cdf(self.d1(maturity, strike)) 
                - np.exp(-self.r*maturity)*strike*norm.cdf(self.d2(maturity, strike)))
    
    def put_price(self, maturity, strike):
        return (np.exp(-self.r*maturity)*strike*norm.cdf(-self.d2(maturity, strike)) 
                - self.s0*norm.cdf(-self.d1(maturity, strike)))

    def simulate(self, t, steps, paths, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        dt = t/steps

        dW = rng.normal(loc=(self.r - 0.5*self.sigma**2)*dt,
                        scale=self.sigma*math.sqrt(dt),
                        size=(steps, paths))
        logS = np.concatenate([np.zeros((1, paths)), np.cumsum(dW, axis=0)])
        
        return self.s0*np.exp(logS)
