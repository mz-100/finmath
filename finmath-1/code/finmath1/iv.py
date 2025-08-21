from math import *
import numpy as np
import numba
from numba import float64, int64


def implied_vol(forward, maturity, strike, option_price, call_or_put_flag=1, discount_factor=1.0, eps=1e-12, max_iter=100):
    # Реальные вычисления делаются Numba-функцией _implied_vol
    return _implied_vol(forward, maturity, strike, option_price, call_or_put_flag, discount_factor, eps, max_iter)


@numba.njit(float64(float64))
def normpdf(x):
    return 1/sqrt(2*pi)*exp(-0.5*x*x)


@numba.njit(float64(float64))
def normcdf(x):
    return (1.0 + erf(x / sqrt(2.0))) / 2.0


# Алгоритм Peter John Acklam для квантильной функции нормального распределения
# https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/
@numba.njit(float64(float64))
def normppf(p): 
    if p <= 0.0 or p >= 1.0:
        return nan

    # Коэффициенты рациональной аппроксимации
    a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00]

    if p < 0.02425:
        q = sqrt(-2*log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p > 1 - 0.02425:
        q = sqrt(-2*log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    else:
        q = p - 0.5
        r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


@numba.vectorize([float64(float64, float64, float64, float64, float64, float64, float64, int64)])
def _implied_vol(forward, maturity, strike, option_price, call_or_put_flag, discount_factor, eps, max_iter):
    # Замена переменных
    y = log(forward/strike)
    p = option_price/(discount_factor*sqrt(forward*strike))
    theta = call_or_put_flag        # для краткости

    # Явное решение для опционов ATMF
    if np.isclose(y, 0):
        return -2*normppf((1-p)/2)/sqrt(maturity)

    # Проверка условий существования IV
    lower_bound = theta * (exp(y/2) - exp(-y/2)) if theta*y > 0 else 0.0
    upper_bound = exp(theta*y/2)
    if p <= lower_bound or p >= upper_bound:
        return nan

    # Начальное значение метода Ньютона - точка перегиба функции Блэка
    x = sqrt(2*abs(y))

    # Метод Ньютона
    for n in range(0, max_iter):
        f = theta*(exp(y/2)*normcdf(theta*(y/x+x/2)) -
                   exp(-y/2)*normcdf(theta*(y/x-x/2))) - p
        f_ =  exp(y/2)*normpdf(y/x + x/2) 
        x -= f/f_
        if abs(f/f_) < eps * sqrt(maturity):
            break
    
    if n < max_iter:
        return x/sqrt(maturity)
    else:
        return nan     # заданная точность не достигнута за разрешенное количество итераций
