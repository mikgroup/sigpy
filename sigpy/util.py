"""
Convenient utilities.
"""
import math
import time
import logging
import numpy as np
import numba as nb

from sigpy import config
if config.cupy_enabled:
    import cupy as cp

if config.mpi4py_enabled:
    from mpi4py import MPI

    if config.nccl_enabled:
        from cupy.cuda import nccl


class Device(object):
    """Device class.

    This class extends from cupy.Device, with id = -1 representing CPU,
    and other ids representing the corresponding GPUs. 
    Similar to cupy.Device, the Device object can be used as a context. 

    Args:
        id_or_device (int or Device or cupy.Device): id = -1 represents CPU.
            and other ids represents corresponding GPUs.

    Attributes:
        id (int): id = -1 represents CPU, and other ids represents corresponding GPUs.

    """

    def __init__(self, id_or_device):

        if isinstance(id_or_device, int):
            id = id_or_device
        elif isinstance(id_or_device, Device):
            id = id_or_device.id
        elif config.cupy_enabled and isinstance(id_or_device, cp.cuda.device.Device):
            id = id_or_device.id
        else:
            raise ValueError('Accepts int, Device or cupy.Device, got {id_or_device}'.format(
                id_or_device=id_or_device))

        if id != -1:
            if config.cupy_enabled:
                self.device = cp.cuda.device.Device(id)
            else:
                raise ValueError(
                    'cupy not installed, but set device {id}'.format(id=id))

        self.id = id

    @property
    def xp(self):
        """module: numpy or cupy module for the device."""
        if self.id == -1:
            return np
        else:
            return cp

    def __eq__(self, other):

        if isinstance(other, int):
            return self.id == other
        elif isinstance(other, Device):
            return self.id == other.id
        elif config.cupy_enabled and isinstance(other, cp.cuda.device.Device):
            return self.id == other.id

    def __enter__(self):

        if self.id == -1:
            return None
        else:
            return self.device.__enter__()

    def __exit__(self, *args):

        if self.id == -1:
            pass
        else:
            self.device.__exit__()

    def __repr__(self):

        if self.id == -1:
            return '<cpu Device>'
        else:
            return '<gpu{id} Device>'.format(id=self.id)


cpu_device = Device(-1)


def _normalize_axes(axes, ndim):
    if axes is None:
        return tuple(range(ndim))
    else:
        return tuple(a % ndim for a in sorted(axes))


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
            raise TypeError('inputs dtype mismatch, got {a_dtype}, and {dtype}.'.format(
                a_dtype=a.dtype, dtype=dtype))


def get_xp(input):
    """Get numpy or cupy module from input array.

    Args:
        input (array): Input.
    
    Returns:
        module: numpy or cupy module.
    """
    if config.cupy_enabled:
        return cp.get_array_module(input)
    else:
        return np


def get_device(input):
    """Get Device from input array.

    Args:
        input (array): Input.
    
    Returns:
        Device.
    """
    if get_xp(input) == np:
        return cpu_device
    else:
        return Device(input.device)


def move(input, device=cpu_device):
    """Move input to device. Does not copy if same device.

    Args:
        input (array): Input.
        device (int or Device or cupy.Device): Output device.
    
    Returns:
        array: Output array placed in device.
    """
    device = Device(device)

    if get_device(input) == device:
        output = input

    elif device == cpu_device:
        with get_device(input):
            output = input.get()
    else:
        with device:
            output = cp.array(input)

    return output


def move_to(output, input):
    """Copy from input to output. Input/output can be in different device.

    Args:
        input (array): Input.
        output (array): Output.
    """
    if get_device(input) == get_device(output):
        with get_device(input):
            output[:] = input

    elif get_device(output) == cpu_device:
        with get_device(input):
            output[:] = input.get()

    elif get_device(input) == cpu_device:
        with get_device(output):
            output.set(input)

    else:
        with get_device(output):
            output[:] = cp.array(input)


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
    device = get_device(inputs[0])
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
    device = get_device(vec)
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

    device = get_device(input)
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

    copy_shape = [min(i - si, o - so) for i, si, o, so in zip(ishape_exp, ishift,
                                                              oshape_exp, oshift)]
    islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
    oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])

    device = get_device(input)
    output = zeros(oshape_exp, dtype=input.dtype, device=device)
    with device:
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
    device = get_device(input)
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
    device = get_device(input)
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

    slc = [slice(s, None, f) for s, f in zip(shift, factors)]

    device = get_device(input)
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

    slc = [slice(s, None, f) for s, f in zip(shift, factors)]

    device = get_device(input)
    output = zeros(oshape, dtype=input.dtype, device=device)
    with device:
        output[slc] = input

    return output


def dirac(shape, dtype=np.complex, device=cpu_device):
    """Create Dirac delta.

    Args:
        shape (tuple of ints): Output shape.
        dtype (Dtype): Output data-type.
        device (Device): Output device.

    Returns:
        array: Dirac delta array.
    """

    device = Device(device)
    xp = device.xp

    with device:
        return resize(xp.ones([1], dtype=dtype), shape)


def randn(shape, scale=1, dtype=np.complex, device=cpu_device):
    """Create random Gaussian array.

    Args:
        shape (tuple of ints): Output shape.
        scale (float): Standard deviation.
        dtype (Dtype): Output data-type.
        device (Device): Output device.

    Returns:
        array: Random Gaussian array.
    """

    device = Device(device)
    xp = device.xp

    with device:
        if np.issubdtype(dtype, np.complexfloating):
            output = (xp.random.normal(
                size=shape, scale=scale / 2**0.5) * 1j).astype(dtype)
            output += xp.random.normal(size=shape, scale=scale / 2**0.5)
            return output
        else:
            return xp.random.normal(size=shape, scale=scale).astype(dtype)


def randn_like(input, scale=1):
    """Create random Gaussian array with shape and dtype as input.

    Args:
        input (array): Input array as reference.
        scale (float): Standard deviation.

    Returns:
        array: Random Gaussian array.
    """

    return randn(input.shape, scale=scale, dtype=input.dtype, device=get_device(input))


def array(arr, dtype=np.complex, device=cpu_device):
    """Creates array on device.

    Args:
        arr (array): Input array.
        dtype (Dtype): Output data-type.
        device (Device): Output device.

    Returns:
        array: Array on device with dtype.
    """

    device = Device(device)
    xp = device.xp

    with device:
        return xp.array(arr, dtype=dtype)


def empty(shape, dtype=np.complex, device=cpu_device):
    """Create empty array.

    Args:
        shape (tuple of ints): Output shape.
        dtype (Dtype): Output data-type.
        device (Device): Output device.

    Returns:
        array: Empty array.
    """

    device = Device(device)
    xp = device.xp

    with device:
        return xp.empty(shape, dtype=dtype)


def empty_like(input):
    """Create empty array with shape and dtype as input.

    Args:
        input (array): Input array as reference.

    Returns:
        array: Empty array.
    """

    return empty(input.shape, dtype=input.dtype, device=get_device(input))


def zeros(shape, dtype=np.complex, device=cpu_device):
    """Create all-zeros array.

    Args:
        shape (tuple of ints): Output shape.
        dtype (Dtype): Output data-type.
        device (Device): Output device.

    Returns:
        array: All-zeros array.
    """

    device = Device(device)
    xp = device.xp

    with device:
        return xp.zeros(shape, dtype=dtype)


def zeros_like(input):
    """Create all-zeros array with shape and dtype as input.

    Args:
        input (array): Input array as reference.

    Returns:
        array: All-zeros array.
    """

    return zeros(input.shape, dtype=input.dtype, device=get_device(input))


def ones(shape, dtype=np.complex, device=cpu_device):
    """Create all-ones array.

    Args:
        shape (tuple of ints): Output shape.
        dtype (Dtype): Output data-type.
        device (Device): Output device.

    Returns:
        array: All-ones array.
    """

    device = Device(device)
    xp = device.xp

    with device:
        return xp.ones(shape, dtype=dtype)


def ones_like(input):
    """Create all-ones array with shape and dtype as input.

    Args:
        input (array): Input array as reference.

    Returns:
        array: All-ones array.
    """

    return ones(input.shape, dtype=input.dtype, device=get_device(input))


def dot(input1, input2):
    """Compute dot product.

    Args:
        input1 (array)
        input2 (array)

    Returns:
        float: Dot product between input1 and input2.
    """
    
    device = get_device(input1)
    xp = device.xp

    with device:
        return xp.real(xp.vdot(input1, input2))


def norm2(input):
    """Compute sum of squares.

    Args:
        input (array)

    Returns:
        float: Sum of squares of input.
    """
    
    device = get_device(input)
    xp = device.xp

    return dot(input, input)


def norm(input):
    """Compute L2 norm.

    Args:
        input (array)

    Returns:
        float: L2 norm of input.
    """
    
    device = get_device(input)
    xp = device.xp

    with device:
        return norm2(input)**0.5


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
        for General Denoising Algorithms. IEEE Transactions on Image Processing.
        17, 9 (2008), 1540-1554.
    """
    device = get_device(y)
    xp = device.xp

    n = y.size
    f_y = f(y)
    b = randn(y.shape, dtype=y.dtype, device=device)
    with device:
        divf_y = dot(b, (f(y + eps * b) - f_y)) / eps
        sure = xp.mean(xp.abs(y - f_y)**2) - sigma**2 + \
            2 * sigma**2 * divf_y / n

    return sure


def axpy(y, a, x):
    """Compute y = a * x + y.

    Args:
        y (array): Output array.
        a (scalar): Input scalar.
        x (array): Input array.
    """

    device = get_device(x)
    x = move(x, device)
    a = move(a, device)

    with device:
        if device == cpu_device:
            _axpy(y, a, x, out=y)
        else:
            _axpy_cuda(y, a, x)


def xpay(y, a, x):
    """Compute y = x + a * y.

    Args:
        y (array): Output array.
        a (scalar): Input scalar.
        x (array): Input array.
    """

    device = get_device(y)
    x = move(x, device)
    a = move(a, device)

    with device:
        if device == cpu_device:
            _xpay(y, a, x, out=y)
        else:
            _xpay_cuda(y, a, x)


@nb.vectorize(nopython=True, cache=True)
def _axpy(y, a, x):
    return a * x + y


@nb.vectorize(nopython=True, cache=True)
def _xpay(y, a, x):
    return x + a * y


if config.cupy_enabled:

    _axpy_cuda = cp.ElementwiseKernel(
        'T y, S a, T x',
        '',
        """
        y += (T) a * x;
        """,
        name='axpy')

    _xpay_cuda = cp.ElementwiseKernel(
        'T y, S a, T x',
        '',
        """
        y = x + (T) a * y;
        """,
        name='axpy')
