# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Statistical scaling operators used by parameter aggregation workflows."""


from collections import Counter
from functools import reduce

import numpy as np


def multiply(x, y):
    """Multiply two numeric values.

    Parameters
    ----------
    x : float
        Left operand.
    y : float
        Right operand.

    Returns
    -------
    float
        Product of ``x`` and ``y``.
    """
    return x * y


class Scaling_operator:
    """Collection of static scaling operators for array-like data."""

    # TODO nonlinear scaling operator

    @staticmethod
    def Harmonic_mean(data):
        """Compute harmonic mean.

        Parameters
        ----------
        data : array-like
            Input numeric data.

        Returns
        -------
        float
            Harmonic mean value.
        """
        data = np.array(data)
        return len(data) / np.nansum(1 / data)

    @staticmethod
    def Arithmetic_mean(data):
        """Compute arithmetic mean.

        Parameters
        ----------
        data : array-like
            Input numeric data.

        Returns
        -------
        float
            Arithmetic mean value.
        """
        data = np.array(data)
        return np.nanmean(data)
    
    @staticmethod
    def Arithmetic_max(data):
        """Compute maximum value ignoring ``NaN``.

        Parameters
        ----------
        data : array-like
            Input numeric data.

        Returns
        -------
        float
            Maximum value.
        """
        data = np.array(data)
        return np.nanmax(data)

    @staticmethod
    def Arithmetic_min(data):
        """Compute minimum value ignoring ``NaN``.

        Parameters
        ----------
        data : array-like
            Input numeric data.

        Returns
        -------
        float
            Minimum value.
        """
        data = np.array(data)
        return np.nanmin(data)
    
    @staticmethod
    def Geometric_mean(data):
        """Compute geometric mean.

        Parameters
        ----------
        data : array-like
            Input numeric data.

        Returns
        -------
        float
            Geometric mean value.
        """
        data = np.array(data)
        return pow(reduce(multiply, data), 1 / len(data))

    @staticmethod
    def Maximum_difference(data):
        """Compute range as ``max(data) - min(data)``.

        Parameters
        ----------
        data : array-like
            Input numeric data.

        Returns
        -------
        float
            Difference between maximum and minimum values.
        """
        data = np.array(data)
        return np.nanmax(data) - np.nanmin(data)

    @staticmethod
    def Majority(data):
        """Compute the most frequent value.

        Parameters
        ----------
        data : array-like
            Input data.

        Returns
        -------
        float
            Most frequent value, or ``np.nan`` when valid data is empty.
        """
        data = np.array(data)
        data = data[~np.isnan(data)]
        if len(data) == 0:
            return np.nan
        counter = Counter(data)
        return max(counter.keys(), key=counter.get)


if __name__ == "__main__":
    x = np.array([2, 3])
    so = Scaling_operator()
    so.Arithmetic_mean(x)
    so.Geometric_mean(x)
    so.Harmonic_mean(x)
