import numpy as np
import numba as nb

from sigpy import config, util

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

    device = util.get_device(input)
    if input.dtype == np.float32 or input.dtype == np.complex64:
        dtype = np.float32
    else:
        dtype = np.float64

    lamda = util.array(lamda, dtype=dtype, device=device)

    if device == util.cpu_device:
        return _soft_thresh(lamda, input)
    else:
        with device:
            return _soft_thresh_cuda(lamda, input)


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

    device = util.get_device(input)

    if device == util.cpu_device:
        return _hard_thresh(lamda, input)
    else:
        with device:
            return _hard_thresh_cuda(lamda, input)


def l0_proj(k, input, axes=None):
    """Projection onto L0 ball.

    Args:
        k (float, or array): Sparsity.
        input (array)

    Returns:
        array: Result.
    """

    device = util.get_device(input)
    xp = device.xp
    shape = input.shape
    axes = util._normalize_axes(axes, input.ndim)
    remain_axes = tuple(set(range(input.ndim)) - set(axes))
    batch = util.prod([shape[a] for a in remain_axes])
    length = util.prod([shape[a] for a in axes])

    with device:
        input = input.transpose(remain_axes + axes)
        input = input.reshape([batch, length])

        idx = xp.argpartition(xp.abs(input), -k, axis=-1)
        output = input
        output[xp.arange(batch), idx[0, :-k]] = 0

        output = output.reshape([shape[a] for a in remain_axes + axes])
        output = output.transpose(np.argsort(remain_axes + axes))

    return output


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
    device = util.get_device(input)
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

    device = util.get_device(input)
    xp = device.xp
    with device:

        norm = xp.sum(xp.abs(input)**2, axis=axes, keepdims=True)**0.5
        mask = norm < eps

        tol = 1e-30
        output = mask * input + (1 - mask) * input / (norm + tol) * eps

    return output


def find_elitist_thresh(lamda, input):

    device = util.get_device(input)
    xp = device.xp

    with device:
        sorted_input = xp.sort(xp.abs(input), axis=-1)[:, ::-1]

    batch = len(input)
    thresh = util.empty([batch, 1], dtype=sorted_input.dtype, device=device)
    lamda = util.move(lamda, device=device)
    if device == util.cpu_device:
        _find_elitist_thresh(thresh, lamda, sorted_input)
    else:
        _find_elitist_thresh_cuda(thresh, lamda, sorted_input, size=batch)

    return thresh


def elitist_thresh(lamda, input, axes=None):
    """Elitist threshold.

    Args:
        lamda (float, or array): Threshold parameter.
        input (array)
        axes (None or tuple of ints): Axes to perform threshold.

    Returns:
        array: Result.

    References:
        Kowalski, M. 2009. Sparse regression using mixed norms.
    """

    shape = input.shape
    axes = util._normalize_axes(axes, input.ndim)
    remain_axes = tuple(set(range(input.ndim)) - set(axes))

    device = util.get_device(input)
    xp = device.xp
    length = util.prod([shape[a] for a in axes])
    batch = input.size // length

    input = input.transpose(remain_axes + axes)
    input = input.reshape([batch, length])

    thresh = find_elitist_thresh(lamda, input)
    output = soft_thresh(thresh, input)

    output = output.reshape([shape[a] for a in remain_axes + axes])
    output = output.transpose(np.argsort(remain_axes + axes))

    return output


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
    for i in nb.prange(batch):
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
            sign = input / abs_input;
        S mag = abs_input - lamda;
        mag = (abs(mag) + mag) / 2.;

        output = mag * sign;
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
        'raw T thresh, T lamda, raw T input',
        '',
        """
        const int length = input.shape()[1];
        T l1 = 0;
        for (int j = 0; j < length; j++) {
            const int idx[] = {i, j};
            l1 += input[idx];
            T t = l1 * lamda / ((T) 1. + lamda * (T) (j + 1.));
            
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
