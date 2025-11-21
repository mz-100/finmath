from dataclasses import dataclass

import numpy as np
import scipy.stats as st
from scipy import optimize, special

from .implied_volatility import implied_vol


@dataclass
class CEV:
    alpha: float
    beta: float

    def __post_init__(self):
        if not (0.0 < self.beta < 1.0):
            raise ValueError('beta must be between 0 and 1')

    def call_price(self, forward_price, maturity, strike, discount_factor=1.0):
        a = strike**(2*(1-self.beta)) / ((self.alpha*(1-self.beta))**2*maturity)
        b = forward_price**(2*(1-self.beta)) / ((self.alpha*(1-self.beta))**2*maturity)

        #return forward_price*st.ncx2(2 + 1/(1-self.beta), b).sf(a) - strike*(1-st.ncx2(1/(1-self.beta), a).sf(b))
        return forward_price*(1-st.ncx2(2 + 1/(1-self.beta), b).cdf(a)) - strike*st.ncx2(1/(1-self.beta), a).cdf(b)
    
    def put_price(self, forward_price, maturity, strike, discount_factor=1.0, epsabs=1.49e-08, epsrel=1.49e-08, limit=500):
        # Используется паритет колл-пут
        return self.call_price(forward_price, maturity, strike, discount_factor, epsabs, epsrel, limit) - discount_factor*(forward_price - strike)

    def implied_vol(self, forward_price, maturity, strike, iv_eps=1e-12, iv_max_iter=100):
        return implied_vol(forward_price, maturity, strike,
                           self.call_price(forward_price, maturity, strike),
                           call_or_put_flag=1, eps=iv_eps, max_iter=iv_max_iter)        
