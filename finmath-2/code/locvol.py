from dataclasses import dataclass, field
import numpy as np
from scipy.special import comb
from scipy.linalg import solve_banded
from scipy.optimize import least_squares
from scipy.interpolate import interp1d


@dataclass
class PiecewiseFunction:
    T: np.ndarray
    F: list[callable]

    def __call__(self, t, x):
        i = np.searchsorted(self.T, t, side='right') - 1
        return self.F[i](x)

@dataclass
class AndreasenHuge:
    local_vol: PiecewiseFunction

    @classmethod
    def calibrate(cls, spot_price, maturities, strikes, call_prices, grid_min, grid_max, grid_n):
        K = np.linspace(grid_min, grid_max, grid_n)
        dk = K[1] - K[0]

        def find_c(s, a, c_prev, dt):
            sigma = interp1d(s, a, kind='linear', bounds_error=False, fill_value=(a[0], a[-1]))
            z = dt/(2*dk**2) * K[1:-1]**2 * sigma(K[1:-1])**2

            A = np.zeros((3, grid_n))
            A[0, 2:] = -z
            A[1, 0] = 1
            A[1, 1:-1] = 1 + 2*z
            A[1, -1] = 1
            A[2, :-2] = -z

            return solve_banded((1, 1), A, c_prev)

        lv = []
        t_prev = 0
        c_prev = np.maximum(spot_price - K, 0)
        for i in range(len(maturities)):
            dt = maturities[i] - t_prev

            res = least_squares(
                fun = lambda a: interp1d(K, find_c(strikes[i], a, c_prev, dt))(strikes[i]) - call_prices[i],
                x0 = np.full(len(strikes[i]), 0.3),
                bounds = (0, np.inf),
                method='trf')

            if res.success:
                lv.append(interp1d(strikes[i], res.x, kind='linear', bounds_error=False, fill_value=(res.x[0], res.x[-1]))) #'extrapolate'))
                t_prev = maturities[i]
                c_prev = find_c(strikes[i], res.x, c_prev, dt)

            else:
                raise RuntimeError(f'least squares method failed at maturity t={maturities[i]}:' + res.message)

        return cls(PiecewiseFunction(maturities, lv))
    
# @dataclass
# class AndreasenHuge:
#     local_vol: PiecewiseFunction

#     @classmethod
#     def calibrate(cls, spot_price, maturities, strikes, call_prices, grid_min_strike, grid_max_strike, grid_n_strikes):
#         K = np.linspace(grid_min_strike, grid_max_strike, grid_n_strikes)
#         dK = K[1] - K[0]

#         def find_c(a, strikes, c_prev, dT):
#             #b = (strikes[i][:-1] + strikes[i][1:])/2
#             z = dT/(2*(dK)**2) * K[1:-1] * interp1d(strikes, a, kind='linear', fill_value=(a[0], a[-1]))(K[1:-1])**2
#             print(a)
#             A = np.zeros((3, grid_n_strikes))
#             A[0, 2:] = -z
#             A[1,0] = 1
#             A[1,1:-1] = 1 + z
#             A[1,-1] = 1
#             A[2, :-2] = -z
#             C =  solve_banded((1, 1), A, c_prev)
#             return interp1d(K, C)(strikes)

#         lv = []
#         for i in range(len(maturities)):
#             if i == 0:
#                 dT = maturities[0]
#                 c_prev = np.maximum(spot_price - K, 0)
#             else:
#                 dT = maturities[i] - maturities[i-1]
#             print(i)

#             res = least_squares(lambda a: find_c(a, strikes[i], c_prev, dT) - call_prices[i],
#                                 x0 = np.full(len(strikes[i]), 0.5),
#                                 method='lm')
#             if res.success:
#                 lv.append(interp1d(strikes, res.x, kind='linear', fill_value=(res.x[0], res.x[-1])))
#                 c_prev = find_c(res.x, strikes[i], c_prev, dT)
#             else:
#                 raise RuntimeError('least squares method failed')
#         return cls(PiecewiseFunction(maturities, lv))


        
