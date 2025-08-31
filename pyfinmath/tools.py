import functools

import numpy as np

def vectorize_methods(methods_list):
    """
    Decorator to vectorize specified methods of a class, with optional parallel execution.

    Args:
        methods_list (list of str): List of method names to vectorize.

    Returns:
        function: Class decorator that replaces methods with vectorized versions.
    """

    def decorate(cls):
        # Create a wrapper for `method_name`
        def make_wrapper(method_name):
            @functools.wraps(getattr(cls, method_name))
            def wrapper(self, *args, **kwargs):
                b = np.broadcast(*args, *kwargs.values())

                # Check if broadcasting is needed
                if b.shape:
                    nargs = len(args)
                    kwargs_keys = list(kwargs.keys())
                    results = [getattr(self, f"_orig_{method_name}")(
                        *x[:nargs], **dict(zip(kwargs_keys, x[nargs:]))) for x in b]
                    return np.asarray(results).reshape(b.shape)
                else:
                    # Scalar case: call original method directly
                    return getattr(self, f"_orig_{method_name}")(*args, **kwargs)

            return wrapper

        # Replace specified methods with their vectorized wrappers
        for method_name in methods_list:
            setattr(cls, f"_orig_{method_name}", getattr(cls, method_name))
            setattr(cls, method_name, make_wrapper(method_name))

        return cls

    return decorate


def derivative(f, x0, dx, n):
    if n == 1:
        return (f(x0 + dx) - f(x0 - dx)) / (2 * dx)
    elif n == 2:
        return (f(x0 + dx) - 2 * f(x0) + f(x0 - dx)) / (dx**2)
    else:
        raise NotImplementedError
