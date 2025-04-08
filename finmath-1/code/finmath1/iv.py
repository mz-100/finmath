import math
import numpy as np
import numba
from numba import float64, int64
from numba_stats.norm import ppf as _normppf


def implied_vol(forward_price, maturity, strike, option_price, call_or_put_flag=1, discount_factor=1.0, eps=1e-12, max_iter=100):
    # Реальные вычисления делаются Numba-функцией _implied_vol
    return _implied_vol(forward_price, maturity, strike, option_price, call_or_put_flag, discount_factor, eps, max_iter)


@numba.njit(float64(float64))
def normpdf(x):
    return 1/math.sqrt(2*math.pi)*math.exp(-0.5*x*x)


@numba.njit(float64(float64))
def normcdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


@numba.njit(float64(float64))
def normppf(x):
    # Преобразование к массиву нужно, так как norm.ppf из numba_stats работает только с массивами
    x_ = np.array([x])
    return _normppf(x_, 0.0, 1.0)[0]


@numba.vectorize([float64(float64, float64, float64, float64, float64, float64, float64, int64)])
def _implied_vol(forward, maturity, strike, option_price, call_or_put_flag, discount_factor, eps, max_iter):
    # Замена переменных
    y = math.log(forward/strike)
    p = option_price/(discount_factor*math.sqrt(forward*strike))
    theta = call_or_put_flag        # для краткости

    # Явное решение для опционов ATMF
    if np.isclose(y, 0):
        return -2*normppf((1-p)/2)/math.sqrt(maturity)

    # Проверка условий существования IV
    lower_bound = theta * (math.exp(y/2) - math.exp(-y/2)) if theta*y > 0 else 0.0
    upper_bound = math.exp(theta*y/2)
    if p <= lower_bound or p >= upper_bound:
        return math.nan

    # Начальное значение метода Ньютона - точка перегиба функции Блэка
    x = math.sqrt(2*abs(y))

    # Метод Ньютона
    for n in range(0, max_iter):
        f = theta*(math.exp(y/2)*normcdf(theta*(y/x+x/2)) -
                   math.exp(-y/2)*normcdf(theta*(y/x-x/2))) - p
        f_ =  math.exp(y/2)*normpdf(y/x + x/2) 
        x -= f/f_
        if abs(f/f_) < eps * math.sqrt(maturity):
            break
    
    if n < max_iter:
        return x/math.sqrt(maturity)
    else:
        return math.nan     # заданная точность не достигнута за разрешенное количество итераций
