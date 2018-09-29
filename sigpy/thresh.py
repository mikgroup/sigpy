# -*- coding: utf-8 -*-
"""Thresholding functions.
"""
import numpy as np
import numba as nb

from sigpy import backend, config, util

if config.cupy_enabled:
    import cupy as cp


def soft_thresh(lamda, input):
    r"""Soft threshold.

    Performs:

    .. math:: (| x | - \lambda)_+  \text{sgn}(x)

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
        else:
            output = _soft_thresh_cuda(lamda, input)

        if np.issubdtype(input.dtype, np.floating):
            output = xp.real(output)

    return output
    

def hard_thresh(lamda, input):
    """Hard threshold.

    Performs: 

    .. math:: 1\{|x| > \lambda\} x.

    Args:
        lamda (float, or array): Threshold parameter.
        input (array)

    Returns:
        array: hard-thresholded result.

    """
    device = backend.get_device(input)

    if device == backend.cpu_device:
        return _hard_thresh(lamda, input)
    else:
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

        tol = 1e-30
        output = mask * input + (1 - mask) * input / (norm + tol) * eps

    return output
def elitist_thresh(lamda, input, axes=None):
    """Elitist threshold.

    Args:
        lamda (float, or array): Threshold parameter.
        input (array): Input array.
        axes (None or tuple of ints): Axes to perform threshold.

    Returns:
        array: Result.

    References:
        Kowalski, M. 2009. Sparse regression using mixed norms.

    """
    shape = input.shape
    axes = util._normalize_axes(axes, input.ndim)
    remain_axes = tuple(set(range(input.ndim)) - set(axes))

    length = util.prod([shape[a] for a in axes])
    batch = input.size // length

    input = input.transpose(remain_axes + axes)
    input = input.reshape([batch, length])

    thresh = find_elitist_thresh(lamda, input)
    output = soft_thresh(thresh, input)

    output = output.reshape([shape[a] for a in remain_axes + axes])
    output = output.transpose(np.argsort(remain_axes + axes))

    return output


def find_elitist_thresh(lamda, input):
    device = backend.get_device(input)
    xp = device.xp
    batch = len(input)
    with device:
        sorted_input = xp.sort(xp.abs(input), axis=-1)[:, ::-1]
        thresh = xp.empty([batch, 1], dtype=sorted_input.dtype)

    if device == backend.cpu_device:
        _find_elitist_thresh(thresh, lamda, sorted_input)
    else:
        _find_elitist_thresh_cuda(thresh, lamda, sorted_input, size=batch)

    return thresh


@nb.vectorize
def _soft_thresh(lamda, input):
    abs_input = abs(input)
    if (abs_input == 0):
        sign = 0
    else:
        sign = input / abs_input

    mag = abs_input - lamda
    mag = (abs(mag) + mag) / 2

    return mag * sign


@nb.vectorize
def _hard_thresh(lamda, input):
    abs_input = abs(input)
    if abs_input > lamda:
        return input
    else:
        return 0


@nb.jit(nopython=True, cache=True)
def _find_elitist_thresh(thresh, lamda, input):
    batch, length = input.shape
    for i in range(batch):
        l1 = 0
        for j in range(length):
            l1 += input[i, j]
            t = l1 * lamda / (1 + lamda * (j + 1))

            if (j == length - 1):
                thresh[i, 0] = t
            elif (t > input[i, j + 1]):
                thresh[i, 0] = t
                break


if config.cupy_enabled:

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

    _find_elitist_thresh_cuda = cp.ElementwiseKernel(
        'raw S thresh, S lamda, raw T input',
        '',
        """
        const int length = input.shape()[1];
        S l1 = 0;
        for (int j = 0; j < length; j++) {
            const int idx[] = {i, j};
            l1 += input[idx];
            S t = l1 * lamda / ((S) 1. + lamda * (S) (j + 1.));
            
            const int thresh_idx[] = {i, 0};
            if (j == length - 1) {
                thresh[thresh_idx] = t;
                break;
            }
            const int next_idx[] = {i, j + 1};
            if (t > input[next_idx]) {
                thresh[thresh_idx] = t;
                break;
            }
        }
        """,
        name='find_elitist_thresh', reduce_dims=False)
