# -*- coding: utf-8 -*-
"""Thresholding functions.
"""
import numpy as np
import numba as nb

from sigpy import backend, config, util


__all__ = ['soft_thresh', 'hard_thresh', 'l1_proj', 'l2_proj']


def soft_thresh(lamda, input):
    r"""Soft threshold.

    Performs:

    .. math::
        (| x | - \lambda)_+  \text{sgn}(x)

    Args:
        lamda (float, or array): Threshold parameter.
        input (array)

    Returns:
        array: soft-thresholded result.

    """
    device = backend.get_device(input)
    xp = device.xp

    lamda = xp.real(lamda)
    with device:
        if device == backend.cpu_device:
            output = _soft_thresh(lamda, input)
        else:  # pragma: no cover
            output = _soft_thresh_cuda(lamda, input)

        if np.issubdtype(input.dtype, np.floating):
            output = xp.real(output)

    return output


def hard_thresh(lamda, input):
    """Hard threshold.

    Args:
        lamda (float, or array): Threshold parameter.
        input (array)

    Returns:
        array: hard-thresholded result.

    """
    device = backend.get_device(input)

    if device == backend.cpu_device:
        return _hard_thresh(lamda, input)
    else:  # pragma: no cover
        with device:
            return _hard_thresh_cuda(lamda, input)


def l1_proj(eps, input):
    """Projection onto L1 ball.

    Args:
        eps (float, or array): L1 ball scaling.
        input (array)

    Returns:
        array: Result.

    References:
        J. Duchi, S. Shalev-Shwartz, and Y. Singer, "Efficient projections onto
        the l1-ball for learning in high dimensions" 2008.

    """
    device = backend.get_device(input)
    xp = device.xp

    with device:
        shape = input.shape
        input = input.ravel()

        if xp.linalg.norm(input, 1) < eps:
            return input
        else:
            shape = len(input)
            s = xp.sort(xp.abs(input))[::-1]
            st = (xp.cumsum(s) - eps) / (xp.arange(shape) + 1)
            idx = xp.flatnonzero((s - st) > 0).max()
            return soft_thresh(st[idx], input.reshape(shape))


def l2_proj(eps, input, axes=None):
    """Projection onto L2 ball.

    Args:
        eps (float, or array): L2 ball scaling.
        input (array)

    Returns:
        array: Result.

    """
    axes = util._normalize_axes(axes, input.ndim)

    device = backend.get_device(input)
    xp = device.xp
    with device:
        norm = xp.sum(xp.abs(input)**2, axis=axes, keepdims=True)**0.5
        mask = norm < eps
        output = mask * input + (1 - mask) * (eps * input / (norm + mask))

    return output


@nb.vectorize  # pragma: no cover
def _soft_thresh(lamda, input):
    abs_input = abs(input)
    if (abs_input == 0):
        sign = 0
    else:
        sign = input / abs_input

    mag = abs_input - lamda
    mag = (abs(mag) + mag) / 2

    return mag * sign


@nb.vectorize  # pragma: no cover
def _hard_thresh(lamda, input):
    abs_input = abs(input)
    if abs_input > lamda:
        return input
    else:
        return 0


if config.cupy_enabled:  # pragma: no cover
    import cupy as cp

    _soft_thresh_cuda = cp.ElementwiseKernel(
        'S lamda, T input',
        'T output',
        """
        S abs_input = abs(input);
        T sign;
        if (abs_input == 0)
            sign = 0;
        else
            sign = input / (T) abs_input;
        S mag = abs_input - lamda;
        mag = (abs(mag) + mag) / 2.;

        output = (T) mag * sign;
        """,
        name='soft_thresh')

    _hard_thresh_cuda = cp.ElementwiseKernel(
        'S lamda, T input',
        'T output',
        """
        S abs_input = abs(input);
        if (abs_input > lamda)
            output = input;
        else
            output = 0;
        """,
        name='hard_thresh')
