# -*- coding: utf-8 -*-
"""Utility functions.
"""
import numpy as np
import numba as nb

from sigpy import backend, config


__all__ = ['asscalar', 'prod', 'vec', 'split', 'rss', 'resize',
           'flip', 'circshift', 'downsample', 'upsample', 'dirac', 'randn',
           'triang', 'hanning', 'monte_carlo_sure', 'axpy', 'xpay',
           'ShuffledNumbers']


def _normalize_axes(axes, ndim):
    if axes is None:
        return tuple(range(ndim))
    else:
        return tuple(a % ndim for a in sorted(axes))


def _normalize_shape(shape):
    if isinstance(shape, int):
        return (shape, )
    else:
        return tuple(shape)


def _expand_shapes(*shapes):

    shapes = [list(shape) for shape in shapes]
    max_ndim = max(len(shape) for shape in shapes)
    shapes_exp = [[1] * (max_ndim - len(shape)) + shape
                  for shape in shapes]

    return tuple(shapes_exp)


def _check_same_dtype(*arrays):

    dtype = arrays[0].dtype
    for a in arrays:
        if a.dtype != dtype:
            raise TypeError(
                'inputs dtype mismatch, got {a_dtype}, and {dtype}.'.format(
                    a_dtype=a.dtype, dtype=dtype))


def asscalar(input):
    """Returns input array as scalar.

    Args:
        input (array): Input array

    Returns:
        scalar.

    """
    return np.asscalar(backend.to_device(input, backend.cpu_device))


def prod(shape):
    """Computes product of shape.

    Args:
        shape (tuple or list): shape.

    Returns:
        Product.

    """
    return np.prod(shape, dtype=np.long)


def vec(inputs):
    """Vectorize inputs.

    Args:
        shape (tuple or list): shape.

    Returns:
        array: Vectorized result.
    """
    device = backend.get_device(inputs[0])
    xp = device.xp

    with device:
        return xp.concatenate([i.ravel() for i in inputs])


def split(vec, oshapes):
    """Split input into specified output shapes.

    Args:
        oshapes (list of tuple of ints): Output shapes.

    Returns:
        list of arrays: Splitted outputs.
    """
    device = backend.get_device(vec)
    with device:
        outputs = []
        for oshape in oshapes:
            osize = prod(oshape)
            outputs.append(vec[:osize].reshape(oshape))
            vec = vec[osize:]

    return outputs


def rss(input, axes=(0, )):
    """Root sum of squares.

    Args:
        input (array): Input array.
        axes (None or tuple of ints): Axes to perform operation.

    Returns:
        array: Result.
    """

    device = backend.get_device(input)
    xp = device.xp

    with device:
        return xp.sum(xp.abs(input)**2, axis=axes)**0.5


def resize(input, oshape, ishift=None, oshift=None):
    """Resize with zero-padding or cropping.

    Args:
        input (array): Input array.
        oshape (tuple of ints): Output shape.
        ishift (None or tuple of ints): Input shift.
        oshift (None or tuple of ints): Output shift.

    Returns:
        array: Zero-padded or cropped result.
    """

    ishape_exp, oshape_exp = _expand_shapes(input.shape, oshape)

    if ishape_exp == oshape_exp:
        return input.reshape(oshape)

    if ishift is None:
        ishift = [max(i // 2 - o // 2, 0)
                  for i, o in zip(ishape_exp, oshape_exp)]

    if oshift is None:
        oshift = [max(o // 2 - i // 2, 0)
                  for i, o in zip(ishape_exp, oshape_exp)]

    copy_shape = [min(i - si, o - so) for i, si, o,
                  so in zip(ishape_exp, ishift, oshape_exp, oshift)]
    islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
    oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])

    device = backend.get_device(input)
    xp = device.xp
    with device:
        output = xp.zeros(oshape_exp, dtype=input.dtype)
        input = input.reshape(ishape_exp)
        output[oslice] = input[islice]

    return output.reshape(oshape)


def flip(input, axes=None):
    """Flip input.

    Args:
        input (array): Input array.
        axes (None or tuple of ints): Axes to perform operation.

    Returns:
        array: Flipped result.
    """

    if axes is None:
        axes = range(input.ndim)
    else:
        axes = _normalize_axes(axes, input.ndim)

    slc = []
    for d in range(input.ndim):
        if d in axes:
            slc.append(slice(None, None, -1))
        else:
            slc.append(slice(None))

    slc = tuple(slc)
    device = backend.get_device(input)
    with device:
        output = input[slc]

    return output


def circshift(input, shifts, axes=None):
    """Circular shift input.

    Args:
        input (array): Input array.
        shifts (tuple of ints): Shifts.
        axes (None or tuple of ints): Axes to perform operation.

    Returns:
        array: Result.
    """

    if axes is None:
        axes = range(input.ndim)

    assert(len(axes) == len(shifts))
    device = backend.get_device(input)
    xp = device.xp

    with device:
        for axis, shift in zip(axes, shifts):
            input = xp.roll(input, shift, axis=axis)

        return input


def downsample(input, factors, shift=None):
    """Downsample input.

    Args:
        input (array): Input array.
        factors (tuple of ints): Downsampling factors.
        shifts (None or tuple of ints): Shifts.

    Returns:
        array: Result.
    """

    if shift is None:
        shift = [0] * len(factors)

    slc = tuple(slice(s, None, f) for s, f in zip(shift, factors))

    device = backend.get_device(input)
    with device:
        return input[slc]


def upsample(input, oshape, factors, shift=None):
    """Upsample input.

    Args:
        input (array): Input array.
        factors (tuple of ints): Upsampling factors.
        shifts (None or tuple of ints): Shifts.

    Returns:
        array: Result.
    """

    if shift is None:
        shift = [0] * len(factors)

    slc = tuple(slice(s, None, f) for s, f in zip(shift, factors))

    device = backend.get_device(input)
    xp = device.xp
    with device:
        output = xp.zeros(oshape, dtype=input.dtype)
        output[slc] = input

    return output


def dirac(shape, dtype=np.float, device=backend.cpu_device):
    """Create Dirac delta.

    Args:
        shape (tuple of ints): Output shape.
        dtype (Dtype): Output data-type.
        device (Device): Output device.

    Returns:
        array: Dirac delta array.
    """

    device = backend.Device(device)
    xp = device.xp

    with device:
        return resize(xp.ones([1], dtype=dtype), shape)


def randn(shape, scale=1, dtype=np.float, device=backend.cpu_device):
    """Create random Gaussian array.

    Args:
        shape (tuple of ints): Output shape.
        scale (float): Standard deviation.
        dtype (Dtype): Output data-type.
        device (Device): Output device.

    Returns:
        array: Random Gaussian array.

    """
    device = backend.Device(device)
    xp = device.xp

    with device:
        if np.issubdtype(dtype, np.complexfloating):
            output = (xp.random.normal(
                size=shape, scale=scale / 2**0.5) * 1j).astype(dtype)
            output += xp.random.normal(size=shape, scale=scale / 2**0.5)
            return output
        else:
            return xp.random.normal(size=shape, scale=scale).astype(dtype)


def triang(shape, dtype=np.float, device=backend.cpu_device):
    """Create multi-dimensional triangular window.

    Args:
        shape (tuple of ints): Output shape.
        dtype (Dtype): Output data-type.
        device (Device): Output device.

    Returns:
        array: triangular filter.

    """
    device = backend.Device(device)
    xp = device.xp
    shape = _normalize_shape(shape)
    with device:
        window = xp.ones(shape, dtype=dtype)
        for n, i in enumerate(shape[::-1]):
            x = xp.arange(i, dtype=dtype)
            w = 1 - xp.abs(x - i // 2 + ((i + 1) % 2) / 2) / ((i + 1) // 2)
            window *= w.reshape([i] + [1] * n)

    return window


def hanning(shape, dtype=np.float, device=backend.cpu_device):
    """Create multi-dimensional hanning window.

    Args:
        shape (tuple of ints): Output shape.
        dtype (Dtype): Output data-type.
        device (Device): Output device.

    Returns:
        array: hanning filter.

    """
    device = backend.Device(device)
    xp = device.xp
    shape = _normalize_shape(shape)
    with device:
        window = xp.ones(shape, dtype=dtype)
        for n, i in enumerate(shape[::-1]):
            x = xp.arange(i, dtype=dtype)
            w = 0.5 - 0.5 * xp.cos(2 * np.pi * x / max(1, (i - (i % 2))))
            window *= w.reshape([i] + [1] * n)

    return window


def monte_carlo_sure(f, y, sigma, eps=1e-10):
    """Monte Carlo Stein Unbiased Risk Estimator (SURE).

    Monte carlo SURE assumes the observation y = x + e,
    where e is a white Gaussian array with standard deviation sigma.
    Monte carlo SURE provides an unbiased estimate of mean-squared error, ie:
    1 / n || f(y) - x ||_2^2

    Args:
        f (function): x -> f(x).
        y (array): observed measurement.
        sigma (float): noise standard deviation.

    Returns:
       float: SURE.

    References:
        Ramani, S., Blu, T. and Unser, M. 2008.
        Monte-Carlo Sure: A Black-Box Optimization of Regularization Parameters
        for General Denoising Algorithms. IEEE Transactions on Image Processing
        17, 9 (2008), 1540-1554.
    """
    device = backend.get_device(y)
    xp = device.xp

    n = y.size
    f_y = f(y)
    b = randn(y.shape, dtype=y.dtype, device=device)
    with device:
        divf_y = xp.real(xp.vdot(b, (f(y + eps * b) - f_y))) / eps
        sure = xp.mean(xp.abs(y - f_y)**2) - sigma**2 + \
            2 * sigma**2 * divf_y / n

    return sure


class ShuffledNumbers(object):
    """Produces shuffled numbers between given range.

    Args:
        Arguments to numpy.arange.

    """

    def __init__(self, *args):
        self.numbers = np.arange(*args)
        np.random.shuffle(self.numbers)
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        ret = self.numbers[self.i]

        self.i += 1
        if self.i == len(self.numbers):
            np.random.shuffle(self.numbers)
            self.i = 0

        return ret


def axpy(y, a, x):
    """Compute y = a * x + y.

    Args:
        y (array): Output array.
        a (scalar): Input scalar.
        x (array): Input array.

    """
    device = backend.get_device(x)
    x = backend.to_device(x, device)
    a = backend.to_device(a, device)

    with device:
        if device == backend.cpu_device:
            _axpy(y, a, x, out=y)
        else:
            _axpy_cuda(a, x, y)


def xpay(y, a, x):
    """Compute y = x + a * y.

    Args:
        y (array): Output array.
        a (scalar): Input scalar.
        x (array): Input array.
    """

    device = backend.get_device(y)
    x = backend.to_device(x, device)
    a = backend.to_device(a, device)

    with device:
        if device == backend.cpu_device:
            _xpay(y, a, x, out=y)
        else:
            _xpay_cuda(a, x, y)


@nb.vectorize(nopython=True, cache=True)  # pragma: no cover
def _axpy(y, a, x):
    return a * x + y


@nb.vectorize(nopython=True, cache=True)  # pragma: no cover
def _xpay(y, a, x):
    return x + a * y


if config.cupy_enabled:  # pragma: no cover
    import cupy as cp

    _axpy_cuda = cp.ElementwiseKernel(
        'S a, T x',
        'T y',
        """
        y += (T) a * x;
        """,
        name='axpy')

    _xpay_cuda = cp.ElementwiseKernel(
        'S a, T x',
        'T y',
        """
        y = x + (T) a * y;
        """,
        name='axpy')
