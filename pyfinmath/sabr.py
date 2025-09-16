from dataclasses import dataclass

import numpy as np
import scipy.optimize as opt
import scipy.stats as st


@dataclass
class SABR:
    alpha: float
    beta: float
    rho: float
    nu: float

    def implied_vol(self, forward_price, maturity, strike):
        s = forward_price
        k = strike
        if self.beta == 0:
            z = self.nu/self.alpha * np.sqrt(s*k) * np.log(s/k)
            x = np.log((np.sqrt(1-2*self.rho*z + z*z) + z - self.rho) /
                       (1-self.rho))
            return (
                self.alpha *
                # при s = k полагаем log(s/k)/(s-k) = k, z/x = 1
                np.divide(np.log(s/k)*z, (s-k)*x,
                          where=np.abs(s-k) > 1e-12,
                          out=np.array(k, dtype=float)) *
                (1 + maturity*(self.alpha**2/(24*s*k) +
                        (2-3*self.rho**2)/24*self.nu**2)))
        elif self.beta == 1:
            z = self.nu/self.alpha * np.log(s/k)
            x = np.log((np.sqrt(1-2*self.rho*z + z*z) + z - self.rho) /
                       (1-self.rho))
            return (
                self.alpha *
                # при s = k полагаем z/x = 1
                np.divide(z, x, where=np.abs(s-k) > 1e-12, out=np.ones_like(z)) *
                (1 + maturity*(self.rho*self.alpha*self.nu/4 +
                        (2-3*self.rho**2)*self.nu**2/24)))
        else:
            z = (self.nu/self.alpha *
                 (s*k)**((1-self.beta)/2) * np.log(s/k))
            x = np.log((np.sqrt(1-2*self.rho*z + z*z) + z -
                        self.rho)/(1-self.rho))
            return (
                self.alpha /
                (s*k)**((1-self.beta)/2)*(
                    1 +
                    (1-self.beta)**2/24*np.log(s/k)**2 +
                    (1-self.beta)**4/1920*np.log(s/k)**4) *
                # при s = k полагаем z/x = 1
                np.divide(z, x, where=np.abs(z) > 1e-12, out=np.ones_like(z)) *
                (1 + maturity*(
                    ((1-self.beta)**2/24 * self.alpha**2 /
                     (s*k)**(1-self.beta)) +
                    (self.rho*self.beta*self.nu*self.alpha /
                     (4*(s*k)**((1-self.beta)/2))) +
                    (2-3*self.rho**2)/24*self.nu**2)))

    @classmethod
    def calibrate(cls, forward_price,  maturity, strike, implied_vol, beta=1, optimization_params=None):
        if optimization_params is None:
            optimization_params = {}
        if 'x0' not in optimization_params:
            alpha = implied_vol.flat[np.abs(strike-forward_price).argmin()]  # ATM volatility
            optimization_params['x0'] = (alpha, -0.1, 1)  # alpha, rho, nu
        if 'bounds' not in optimization_params:
            optimization_params['bounds'] = [(0, None), (-1, 1), (0, None)]
        if 'method' not in optimization_params:
            optimization_params['method'] = 'Nelder-Mead'

        def fun(p):
            return np.linalg.norm(cls(p[0], beta, p[1], p[2]).implied_vol(forward_price, maturity, strike) - implied_vol)
        res = opt.minimize(fun, **optimization_params)
        if not res.success:
            raise RuntimeError("Minimization failed: " + res.message)
        return cls(alpha=res.x[0], beta=beta, rho=res.x[1], nu=res.x[2])
