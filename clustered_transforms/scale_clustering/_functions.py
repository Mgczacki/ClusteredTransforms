from typing import Union

import numpy as np


def logarithmic_interpolation(
    x: Union[float, np.array], x1: float, y1: float, x2: float, y2: float
) -> Union[float, np.array]:
    """Perform logarithmic interpolation between two points (x1, y1) and (x2, y2) for
    the value x.

    Parameters
    ----------
    x: float | np.Array
        The value(s) for which interpolation is required.
    x1: float
        The x-coordinate of the first known point.
    y1: float
        The y-coordinate of the first known point.
    x2: float
        The x-coordinate of the second known point.
    y2: float
        The y-coordinate of the second known point.
    """
    if x1 <= 0 or x2 <= 0 or x <= 0:
        raise ValueError("All values must be positive for logarithmic interpolation.")

    if x1 == x2:
        return (y1 + y2) / 2

    log_x = np.log10(x)
    log_x1 = np.log10(x1)
    log_x2 = np.log10(x2)

    y = y1 + (y2 - y1) * (log_x - log_x1) / (log_x2 - log_x1)

    return y


def inv_logarithmic_interpolation(
    y: Union[float, np.array], x1: float, y1: float, x2: float, y2: float
) -> Union[float, np.array]:
    """Invert the logarithmic interpolation between two points (x1, y1) and (x2, y2) for
    the value x.

    Parameters
    ----------
    y: float | np.Array
        The value(s) for which inverse interpolation is required.
    x1: float
        The x-coordinate of the first known point.
    y1: float
        The y-coordinate of the first known point.
    x2: float
        The x-coordinate of the second known point.
    y2: float
        The y-coordinate of the second known point.
    """
    if y < y1 or y > y2:
        raise ValueError("Interpolated y value is outside the range of interpolation.")

    if y1 == y2:
        return (x1 + x2) / 2

    log_x1 = np.log10(x1)
    log_x2 = np.log10(x2)

    log_x = log_x1 + (log_x2 - log_x1) * (y - y1) / (y2 - y1)

    x = 10**log_x

    return x


def scaled_logistic(
    x: Union[float, np.array],
    lower: float = 0,
    upper: float = 1,
    a: float = 1,
    x0: float = 0,
) -> Union[float, np.array]:
    """A scaled generalized logistic function.

    Parameters
    ----------
    x: float | np.Array
        Input value(s).
    lower: float, default = 0
        The lower image bound.
    upper: float, default = 0
        The higher image bound.
    a: float, deafult = 1
        The scale parameter.
    x0: float, default = 0
        The location parameter.
    """
    return lower + (upper - lower) / (1 + np.exp(-a * (x - x0)))


def inv_scaled_logistic(
    y: Union[float, np.array],
    lower: float = 0,
    upper: float = 1,
    a: float = 1,
    x0: float = 0,
) -> Union[float, np.array]:
    """Inverse of the scaled generalized logistic function.

    Parameters
    ----------
    y: float | np.Array
        Input value(s).
    lower: float, default = 0
        The lower image bound.
    upper: float, default = 0
        The higher image bound.
    a: float, deafult = 1
        The scale parameter.
    x0: float, default = 0
        The location parameter.
    """
    return x0 - (1 / a) * np.log((upper - lower) / (y - lower) - 1)
