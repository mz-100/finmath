import cmath
import math
from dataclasses import dataclass

import numba
import numpy as np
import scipy.integrate as intg
import scipy.optimize as opt
from scipy import LowLevelCallable

from .implied_volatility import implied_vol
from .tools import derivative, vectorize_methods


@numba.njit
def _heston_cf(u, t, v0, kappa, theta, sigma, rho):
    d = cmath.sqrt((1j*rho*sigma*u - kappa)**2 + sigma**2*(1j*u + u**2))
    g = ((1j*rho*sigma*u - kappa + d) / (1j*rho*sigma*u - kappa - d))
    C = (kappa*theta/sigma**2 * (
        (kappa - 1j*rho*sigma*u - d)*t - 2*cmath.log((1 - g*cmath.exp(-d*t))/(1-g))))
    D = ((kappa - 1j*rho*sigma*u - d)/sigma**2 * ((1-cmath.exp(-d*t)) / (1-g*cmath.exp(-d*t))))
    return cmath.exp(C + D*v0)


@numba.cfunc('float64(intc, CPointer(float64))')
def _heston_price_integrand(n, x):
    u, t, omega, v0, kappa, theta, sigma, rho = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
    return (cmath.exp(1j*omega*u)/(1j*u) *
            (_heston_cf(u-1j, t, v0, kappa, theta, sigma, rho) -
                math.exp(-omega)*_heston_cf(u, t, v0, kappa, theta, sigma, rho))).real


@numba.cfunc('float64(intc, CPointer(float64))')
def _heston_cdf_integrand(n, x):
    u, t, omega, v0, kappa, theta, sigma, rho = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]   
    return (cmath.exp(1j*omega*u)/(1j*u) * _heston_cf(u, t, v0, kappa, theta, sigma, rho)).real


@numba.njit
def _heston_call_price_cos(s, t, k, v0, kappa, theta, sigma, rho, epsabs, epsrel, limit):
    c1 = (1-math.exp(-kappa*t))*(theta-v0)/(2*kappa) - 0.5*theta*t
    c2 = 1/(8*kappa**3) * (
        sigma*t*kappa*math.exp(-kappa*t)*(v0-theta)*(8*kappa*rho-4*sigma) + kappa*rho*sigma*(1-math.exp(-kappa*t))*(16*theta-8*v0)
        + 2*theta*kappa*t*(-4*kappa*rho*sigma+sigma**2+4*kappa**2)+sigma**2*((theta-2*v0)*math.exp(-2*kappa*t) + theta*(6*math.exp(-kappa*t)-7)+2*v0)
        + 8*kappa**2*(v0-theta)*(1-math.exp(-kappa*t)))
    a = c1 - 12*math.sqrt(math.fabs(c2))
    b = c1 + 12*math.sqrt(math.fabs(c2))
    U0 = k*2/(b-a)*(math.exp(b)-1-b)
    V = 0.5*U0
    x0 = math.log(s/k)
    n = 1
    next_term = 0
    rel_err = 0
    while (n < 10) or (n < limit and ((next_term >= epsabs) or (rel_err >= epsrel))):
        chi = 1/(1+(math.pi*n/(b-a))**2) * ((-1)**n*math.exp(b) - math.cos(-math.pi*n*a/(b-a)) - math.pi*n/(b-a)*math.sin(-math.pi*n*a/(b-a)))
        psi = math.sin(math.pi*n*a/(b-a))*(b-a)/(n*math.pi)
        U = k*2/(b-a)*(chi - psi)
        next_term = (-_heston_cf(math.pi*n/(b-a), t, v0, kappa, theta, sigma, rho)*cmath.exp(1j*math.pi*n*(x0-a)/(b-a))).real * U
        V += next_term
        rel_err = math.fabs(next_term/V)
        n +=1
    return V


try:
    import cupy as cp
    import cupyx as cpx

    @cpx.jit.rawkernel()
    def _heston_qe_gpu_kernel(S, V, V_tmp, Z, U, mu_dt, theta, K, C, paths, intersteps):
        thread_id = cpx.jit.blockIdx.x * cpx.jit.blockDim.x + cpx.jit.threadIdx.x  # pylint: disable=no-member
        total_threads = cpx.jit.gridDim.x * cpx.jit.blockDim.x                     # pylint: disable=no-member

        # if the number of paths is greater then the number of threads, we make one thread simulate multiple paths
        for j in range(thread_id, paths, total_threads):
            if j < paths:
                for i in range(intersteps):
                    m = V[j]*C[1] + theta*(1-C[1])
                    s_sq = V[j]*C[2] + C[3]
                    psi = s_sq/m**2
                    b_sq = max(2/psi - 1 + cp.sqrt(max(4/psi**2 - 2/psi, 0)), 0)
                    a = m/(1+b_sq)
                    p = (psi-1)/(psi+1)
                    beta = (1-p)/m
                    V_tmp[j] = V[j]
                    V[j] = a*(cp.sqrt(b_sq)+Z[0,i,j])**2 if psi < 1.5 else (0 if U[i,j] < p else cp.log((1-p)/(1-U[i,j]))/beta)
                    S[j] *= cp.exp(mu_dt[i] + K[0] + K[1]*V_tmp[j] + K[2]*V[j] + cp.sqrt(K[3]*(V_tmp[j] + V[j]))*Z[1,i,j])

except ImportError:
    pass

def _heston_qe_cpu_kernel(S, V, V_tmp, Z, U, mu_dt, theta, K, C, paths, intersteps):
    for i in range(intersteps):
        m = V*C[1] + theta*(1-C[1])
        s_sq = V*C[2] + C[3]
        psi = s_sq/m**2
        b_sq = np.maximum(2/psi - 1 + np.sqrt(np.maximum(4/psi**2 - 2/psi, 0)), 0)
        a = m/(1+b_sq)
        p = (psi-1)/(psi+1)
        beta = (1-p)/m
        V_tmp = V
        V = np.where(psi < 1.5,
                        a*(np.sqrt(b_sq)+Z[0,i])**2,
                        np.where(U[i] < p, 0, np.log((1-p)/(1-U[i]))/beta))
        S *= np.exp(mu_dt[i] + K[0] + K[1]*V_tmp + K[2]*V + np.sqrt(K[3]*(V_tmp + V))*Z[1,i])


@vectorize_methods(['call_price', 'put_price', 'cdf', 'quantile'])
@dataclass
class Heston:
    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float

    def call_price(self, forward_price, maturity, strike, discount_factor=1.0, epsabs=1.49e-08, epsrel=1.49e-08, limit=500, method='quad'):
        if method=='quad':
            return discount_factor * forward_price * (
                (1 - strike/forward_price)/2 +
                1/math.pi * intg.quad(
                    LowLevelCallable(_heston_price_integrand.ctypes),
                    0, math.inf,
                    args=(maturity, math.log(forward_price/strike), self.v0, self.kappa, self.theta, self.sigma, self.rho),
                    epsabs=epsabs, 
                    epsrel=epsrel,
                    limit=limit)[0]
                )
        elif method=='cos':
            return discount_factor * _heston_call_price_cos(
                forward_price, maturity, strike, self.v0, self.kappa, self.theta, self.sigma, self.rho, epsabs, epsrel, limit)
        else:
            raise ValueError(f'Unknown method: {method}')


    def put_price(self, forward_price, maturity, strike, discount_factor=1.0, epsabs=1.49e-08, epsrel=1.49e-08, limit=500):
        # Используется паритет колл-пут
        return self.call_price(forward_price, maturity, strike, discount_factor, epsabs, epsrel, limit) - discount_factor*(forward_price - strike)

    def implied_vol(self, forward_price, maturity, strike, price_epsabs=1.49e-08, price_epsrel=1.49e-08, price_limit=500, iv_eps=1e-12, iv_max_iter=100):
        return implied_vol(forward_price, maturity, strike,
                           self.call_price(forward_price, maturity, strike, epsabs=price_epsabs, epsrel=price_epsrel, limit=price_limit),
                           call_or_put_flag=1, eps=iv_eps, max_iter=iv_max_iter)
    
    def local_vol(self, s0, t, s, ds=0.01, dt=1/250):
        return np.sqrt(2*derivative(lambda u: self.call_price(s0, u, s), t, dt, n=1) / (s*s*derivative(lambda y: self.call_price(s0, t, y), s, s*ds, n=2)))
    
    def cdf(self, s0, t, s, epsabs=1.49e-08, epsrel=1.49e-08, limit=500):
        return 0.5 - 1/math.pi * intg.quad(
                LowLevelCallable(_heston_cdf_integrand.ctypes),
                0, math.inf,
                args=(t, math.log(s0/s), self.v0, self.kappa, self.theta, self.sigma, self.rho),
                epsabs=epsabs,
                epsrel=epsrel,
                limit=limit)[0]
    
    def quantile(self, s0, t, p, price_epsabs=1.49e-08, price_epsrel=1.49e-08, price_limit=500, root_eps=1.49e-08):
        return opt.newton(lambda s: self.cdf(s0, t, s, price_epsabs, price_epsrel, price_limit) - p, s0, tol=root_eps)
    
    
    @classmethod
    def calibrate(cls, forward_price, maturity, strike, implied_vol, optimization_params=None, iv_params=None):
        if optimization_params is None:
            optimization_params = {}
        if 'x0' not in optimization_params:
            v0 = implied_vol.flat[np.abs(strike-forward_price).argmin()]**2  # ATM variance
            optimization_params['x0'] = (v0, 1.0, v0, 1.0, -0.5)  # (V0, kappa, theta, sigma, rho)
        if 'bounds' not in optimization_params:
            optimization_params['bounds'] = [(0, None), (0, None), (0, None), (0, None), (-1, 1)]
        if 'method' not in optimization_params:
            optimization_params['method'] = 'Nelder-Mead'

        if iv_params is None:
            iv_params = {}
        
        def fun(params):
            return np.linalg.norm(Heston(*params).implied_vol(forward_price, maturity, strike, **iv_params) - implied_vol),
        res = opt.minimize(fun, **optimization_params)
        if not res.success:
            raise RuntimeError("Minimization failed: " + res.message)
        return cls(v0=res.x[0], kappa=res.x[1], theta=res.x[2], sigma=res.x[3], rho=res.x[4])


    def simulate(self, initial_price, t, steps, paths, intersteps=1, drift=0.0, return_variance=False, seed=None, use_gpu=False, cuda_tpb=512, cuda_blocks=None):
        dt = t/(steps*intersteps)
        K = [-self.rho*self.kappa*self.theta*dt/self.sigma,                      # K0
             0.5*(self.kappa*self.rho/self.sigma-0.5)*dt - self.rho/self.sigma,  # K1
             0.5*(self.kappa*self.rho/self.sigma-0.5)*dt + self.rho/self.sigma,  # K2
             0.5*(1-self.rho**2)*dt]                                             # K3
        # K4 from Andersen's paper is equal to K3, since we use gamma_1 = gamma_2 = 1/2

        C1 = math.exp(-self.kappa*dt)
        C = [0.0,                                                                # C0
             C1,                                                                 # C1
             self.sigma**2*C1*(1-C1)/self.kappa,                                 # C2
             0.5*self.theta*self.sigma**2*(1-C1)**2/self.kappa]                  # C3
        
        if np.isscalar(drift):
            mu_dt = np.full((steps, intersteps), drift*dt)
        elif callable(drift):
            mu_dt = drift(np.linspace(0, t, steps*intersteps+1)[:-1]).reshape((steps, intersteps))*dt
        else:
            raise ValueError('drift parameter must be float or callable')

        if use_gpu:
            if cuda_blocks is None:
                cuda_blocks = (paths+cuda_tpb-1)//cuda_tpb
            def randn(*args):
                return cp.random.randn(*args, dtype=cp.float32)
            def randu(*args):
                return cp.random.rand(*args, dtype=cp.float32)
            kernel = _heston_qe_gpu_kernel[cuda_blocks,cuda_tpb]
            theta = cp.float32(self.theta)
            mu_dt = cp.asarray(mu_dt, dtype=cp.float32)
            K = cp.asarray(K, dtype=cp.float32)
            C = cp.asarray(C, dtype=cp.float32)
            S = cp.empty((steps+1, paths), dtype=cp.float32)
            V = cp.empty_like(S, dtype=cp.float32)
            V_tmp = cp.empty(paths, dtype=cp.float32)
            S[0] = cp.float32(initial_price)
            V[0] = cp.float32(self.v0)
            
            if seed is not None:
                cp.random.seed(seed)
        else:
            randn = np.random.randn
            randu = np.random.rand
            kernel = _heston_qe_cpu_kernel
            theta = self.theta
            S = np.empty((steps+1, paths))
            V = np.empty_like(S)
            V_tmp = np.empty(paths)
            S[0] = initial_price
            V[0] = self.v0
        
            if seed is not None:
                np.random.seed(seed)

        for i in range(steps):
            S[i+1] = S[i]
            V[i+1] = V[i]
            Z = randn(2, intersteps, paths)
            U = randu(intersteps, paths)
            kernel(S[i+1], V[i+1], V_tmp, Z, U, mu_dt[i], theta, K, C, paths, intersteps)

        if return_variance:
            return S, V
        else:
            return S


        
