import math
from dataclasses import dataclass

import numpy as np


@dataclass
class MCResult:
    x: float | np.ndarray
    error: float | np.ndarray
    iterations: int
    success: bool | np.ndarray | None = None
    cv_coef: float | None = None


def monte_carlo(rv, size, abs_err=0.0, rel_err=0.0, crit_value=3.0):
    x = 0
    x_sq = 0
    n = 0
    error = abs_err + 1

    while np.any(error >= abs_err + np.abs(x) * rel_err) and n < size:
        a = rv()
        x = (x * n + np.sum(a, axis=0)) / (n + len(a))
        x_sq = (x_sq * n + np.sum(a**2, axis=0)) / (n + len(a))
        s = np.sqrt(x_sq - x**2)
        n += len(a)
        error = crit_value * s / math.sqrt(n)

    if abs_err + rel_err > 0:
        return MCResult(
            x=x,
            error=error,
            iterations=n,
            success=error <= abs_err + np.abs(x) * rel_err)
    return MCResult(x=x, error=error, iterations=n)


def monte_carlo_cv(rv, f, f_c, size, size_c=1, abs_err=0.0, rel_err=0.0, crit_value=3.0):
    # Оценка контрольного коэффициента
    a = np.concatenate([rv() for i in range(size_c)])
    x = f(a)
    y = f_c(a)
    x0 = x - np.mean(x, axis=0)
    y0 = y - np.mean(y, axis=0)
    cv_coef = np.sum(x0 * y0, axis=0) / np.sum(y0**2, axis=0)

    # Новая случайная величина для симуляции
    def new_rv():
        a = rv()
        return f(a) - cv_coef * f_c(a)

    res = monte_carlo(new_rv, size, abs_err, rel_err, crit_value)
    res.cv_coef = cv_coef
    return res
