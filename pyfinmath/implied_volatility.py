import math

import numba
import numpy as np
from numba import float64, int64


def implied_vol(forward_price, maturity, strike, option_price, call_or_put_flag=1,
                discount_factor=1.0, eps=1e-12, max_iter=100):
    return _implied_vol(forward_price, maturity, strike, option_price, call_or_put_flag,
                        discount_factor, eps, max_iter)


@numba.njit
def normpdf(x):
    return 1 / math.sqrt(2*math.pi) * math.exp(-0.5*x*x)


@numba.njit
def normcdf(x):
    return 0.5*(1 + math.erf(x/math.sqrt(2)))


# Алгоритм Peter John Acklam для квантильной функции нормального распределения
# https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/
@numba.njit
def normppf(p):
    if p <= 0.0 or p >= 1.0:
        return math.nan

    # Коэффициенты рациональной аппроксимации
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02,
         -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01,
         -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]

    if p < 0.02425:
        q = math.sqrt(-2*math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p > 1 - 0.02425:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    else:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1))


@numba.vectorize
def _implied_vol(forward_price, maturity, strike, option_price, call_or_put_flag, discount_factor,
                 eps, max_iter):
    # Замена переменных
    y = math.log(forward_price/strike)
    p = option_price / (discount_factor*math.sqrt(forward_price*strike))
    theta = call_or_put_flag  # для краткости

    # Явное решение для опционов ATMF
    if np.isclose(y, 0):
        return -2*normppf((1-p)/2) / math.sqrt(maturity)

    # Проверка условий существования IV
    lower_bound = theta*(math.exp(y/2) - math.exp(-y/2)) if theta*y > 0 else 0.0
    upper_bound = math.exp(theta*y/2)
    if p <= lower_bound or p >= upper_bound:
        return math.nan

    # Начальное значение метода Ньютона - точка перегиба функции Блэка
    x = math.sqrt(2*abs(y))

    # Метод Ньютона
    for n in range(0, max_iter):
        f = (theta*(math.exp(y / 2) * normcdf(theta * (y / x + x / 2))
                    - math.exp(-y / 2) * normcdf(theta * (y / x - x / 2))) - p)
        f_ = math.exp(y/2) * normpdf(y/x + x/2)
        x -= f/f_
        if abs(f/f_) < eps*math.sqrt(maturity):
            break

    if n < max_iter:
        return x / math.sqrt(maturity)
    return math.nan  # заданная точность не достигнута за разрешенное количество итераций
