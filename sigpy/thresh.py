# -*- coding: utf-8 -*-
"""Thresholding functions.
"""
import numpy as np
import numba as nb

from sigpy import backend, config, util


__all__ = ['soft_thresh', 'hard_thresh', 'l1_proj',
           'l2_proj', 'linf_proj', 'psd_proj']


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
    if xp == np:
        return _soft_thresh(lamda, input)
    else:  # pragma: no cover
        if np.isscalar(lamda):
            lamda = backend.to_device(lamda, device)

        return _soft_thresh_cuda(lamda, input)


def hard_thresh(lamda, input):
    """Hard threshold.

    Args:
        lamda (float, or array): Threshold parameter.
        input (array)

    Returns:
        array: hard-thresholded result.

    """
    device = backend.get_device(input)
    xp = device.xp
    if xp == np:
        return _hard_thresh(lamda, input)
    else:  # pragma: no cover
        if np.isscalar(lamda):
            lamda = backend.to_device(lamda, device)

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
    xp = backend.get_array_module(input)
    shape = input.shape
    input = input.ravel()

    if xp.linalg.norm(input, 1) < eps:
        return input
    else:
        size = len(input)
        s = xp.sort(xp.abs(input))[::-1]
        st = (xp.cumsum(s) - eps) / (xp.arange(size) + 1)
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

    xp = backend.get_array_module(input)
    norm = xp.sum(xp.abs(input)**2, axis=axes, keepdims=True)**0.5
    mask = norm < eps
    output = mask * input + (1 - mask) * (eps * input / (norm + mask))

    return output


def linf_proj(eps, input, bias=None):
    """Projection onto L-infinity ball.

    Args:
        eps (float, or array): l-infinity ball scaling.
        input (array)

    Returns:
        array: Result.

    """
    if bias is not None:
        input = input - bias

    output = input - soft_thresh(eps, input)

    if bias is not None:
        output += bias

    return output


def psd_proj(input):
    """Projection onto postiive semi-definite matrices.

    Args:
        input (array): a two-dimensional matrix.

    Returns:
        array: Result.

    """
    xp = backend.get_array_module(input)
    w, v = xp.linalg.eig((input + xp.conj(input).T) / 2)
    w[w < 0] = 0
    return (v * w) @ v.conjugate().T


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
