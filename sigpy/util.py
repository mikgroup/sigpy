# -*- coding: utf-8 -*-
"""Utility functions.
"""
import numpy as np

from sigpy import backend


__all__ = ['prod', 'vec', 'split', 'rss', 'resize',
           'flip', 'circshift', 'downsample', 'upsample', 'dirac', 'randn',
           'triang', 'hanning', 'monte_carlo_sure', 'axpy', 'xpay', 'leja']


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
    xp = backend.get_array_module(inputs[0])
    return xp.concatenate([i.ravel() for i in inputs])


def split(vec, oshapes):
    """Split input into specified output shapes.

    Args:
        oshapes (list of tuple of ints): Output shapes.

    Returns:
        list of arrays: Splitted outputs.
    """
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
    xp = backend.get_array_module(input)
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

    ishape1, oshape1 = _expand_shapes(input.shape, oshape)

    if ishape1 == oshape1:
        return input.reshape(oshape)

    if ishift is None:
        ishift = [max(i // 2 - o // 2, 0) for i, o in zip(ishape1, oshape1)]

    if oshift is None:
        oshift = [max(o // 2 - i // 2, 0) for i, o in zip(ishape1, oshape1)]

    copy_shape = [min(i - si, o - so)
                  for i, si, o, so in zip(ishape1, ishift, oshape1, oshift)]
    islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
    oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])

    xp = backend.get_array_module(input)
    output = xp.zeros(oshape1, dtype=input.dtype)
    input = input.reshape(ishape1)
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

    axes = _normalize_axes(axes, input.ndim)

    slc = []
    for d in range(input.ndim):
        if d in axes:
            slc.append(slice(None, None, -1))
        else:
            slc.append(slice(None))

    slc = tuple(slc)
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
    xp = backend.get_array_module(input)

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

    xp = backend.get_array_module(input)
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
            real_dtype = np.array([], dtype=dtype).real.dtype
            real_shape = tuple(shape) + (2, )
            output = xp.random.normal(size=real_shape, scale=scale / 2**0.5)
            output = output.astype(real_dtype)
            output = output.view(dtype=dtype).reshape(shape)
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
    divf_y = xp.real(xp.vdot(b, (f(y + eps * b) - f_y))) / eps
    sure = xp.mean(xp.abs(y - f_y)**2) - sigma**2 + 2 * sigma**2 * divf_y / n

    return sure


def leja(x):
    """ Perform leja ordering of roots of a polynomial.

    Orders roots in a way suitable to accurately compute polynomial
    coefficients.

    Args:
        x (array): roots to be ordered.

    Returns:
        array: ordered roots.

    References:
        Lang, M. and B. Frenzel. 1993.
        A New and Efficient Program for Finding All Polynomial Roots. Rice
        University ECE Technical Report, no. TR93-08, 1993.
    """

    n = np.size(x)
    # duplicate roots to n+1 rows
    a = np.tile(np.reshape(x, (1, n)), (n+1, 1))
    # take abs of first row
    a[0, :] = np.abs(a[0, :])

    tmp = np.zeros(n+1, dtype=complex)

    # find index of max abs value
    ind = np.argmax(a[0, :])
    if ind != 0:
        tmp[:] = a[:, 0]
        a[:, 0] = a[:, ind]
        a[:, ind] = tmp

    x_out = np.zeros(n, dtype=complex)
    x_out[0] = a[n-1, 0]  # first entry of last row
    a[1, 1:] = np.abs(a[1, 1:] - x_out[0])

    foo = a[0, 0:n]

    for l in range(1, n-1):
        foo = np.multiply(foo, a[l, :])
        ind = np.argmax(foo[l:])
        ind = ind + l
        if l != ind:
            tmp[:] = a[:, l]
            a[:, l] = a[:, ind]
            a[:, ind] = tmp
            # also swap inds in foo
            tmp[0] = foo[l]
            foo[l] = foo[ind]
            foo[ind] = tmp[0]
        x_out[l] = a[n-1, l]
        a[l+1, (l+1):n] = np.abs(a[l+1, (l+1):] - x_out[l])

    x_out = a[n, :]

    return x_out


def axpy(y, a, x):
    """Compute y = a * x + y.

    Args:
        y (array): Output array.
        a (scalar or array): Input scalar.
        x (array): Input array.

    """
    y += a * x


def xpay(y, a, x):
    """Compute y = x + a * y.

    Args:
        y (array): Output array.
        a (scalar or array): Input scalar.
        x (array): Input array.
    """
    y *= a
    y += x
