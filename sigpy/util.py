'''
Convenient utilities.
'''
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


__all__ = ['rss',
           'resize', 'flip', 'circshift',
           'randn', 'rand', 'mse']


class Device(object):

    def __init__(self, input):

        if isinstance(input, int):
            id = input
        elif isinstance(input, Device):
            id = input.id
        elif config.cupy_enabled and isinstance(input, cp.cuda.device.Device):
            id = input.id
        else:
            raise ValueError('Only accepts int, Device or cupy device as input, got {input}'.format(
                input=input))

        if id != -1:
            if config.cupy_enabled:
                self.device = cp.cuda.device.Device(id)
            else:
                raise ValueError(
                    'cupy not installed, but set device {id}'.format(id=id))

        self.id = id

    @property
    def xp(self):
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


def profile(func):
    def wrap(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info('%s takes %f secs', func.__qualname__,
                     end_time - start_time)
        return result

    return wrap


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
    if config.cupy_enabled:
        return cp.get_array_module(input)
    else:
        return np


def get_device(input):

    if get_xp(input) == np:
        return cpu_device
    else:
        return Device(input.device)


def move(input, device=cpu_device):

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


def prod(x):
    return np.prod(x, dtype=np.long)


def vec(inputs):
    device = get_device(inputs[0])
    xp = device.xp

    with device:
        return xp.concatenate([i.ravel() for i in inputs])


def split(vec, oshapes):
    device = get_device(vec)
    with device:
        outputs = []
        for oshape in oshapes:
            osize = prod(oshape)
            outputs.append(vec[:osize].reshape(oshape))
            vec = vec[osize:]

    return outputs


def mse(x, y):
    device = get_device(x)
    xp = device.xp

    with device:
        return xp.mean(xp.abs(x - y).ravel()**2)


def psnr(ref, rec):
    device = get_device(ref)
    xp = device.xp

    with device:
        return 10 * xp.log10(xp.abs(ref).max()**2 / mse(ref, rec))


def rss(input, axes=(0, )):

    device = get_device(input)
    xp = device.xp

    with device:
        return xp.sum(xp.abs(input)**2, axis=axes)**0.5


def resize(input, oshape, ishift=None, oshift=None):

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

    if shift is None:
        shift = [0] * len(factors)

    slc = [slice(s, None, f) for s, f in zip(shift, factors)]

    device = get_device(input)
    with device:
        return input[slc]


def upsample(input, oshape, factors, shift=None):

    if shift is None:
        shift = [0] * len(factors)

    slc = [slice(s, None, f) for s, f in zip(shift, factors)]

    device = get_device(input)
    output = zeros(oshape, dtype=input.dtype, device=device)
    with device:
        output[slc] = input

    return output


def dirac(shape, dtype=np.complex, device=cpu_device):

    device = Device(device)
    xp = device.xp

    with device:
        return resize(xp.ones([1], dtype=dtype), shape)


def randn(shape, scale=1, dtype=np.complex, device=cpu_device):

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

    return randn(input.shape, scale=scale, dtype=input.dtype, device=get_device(input))


def array(arr, dtype=np.complex, device=cpu_device):

    device = Device(device)
    xp = device.xp

    with device:
        return xp.array(arr, dtype=dtype)


def empty(shape, dtype=np.complex, device=cpu_device):

    device = Device(device)
    xp = device.xp

    with device:
        return xp.empty(shape, dtype=dtype)


def empty_like(input):

    return empty(input.shape, dtype=input.dtype, device=get_device(input))


def zeros(shape, dtype=np.complex, device=cpu_device):

    device = Device(device)
    xp = device.xp

    with device:
        return xp.zeros(shape, dtype=dtype)


def zeros_like(input):

    return zeros(input.shape, dtype=input.dtype, device=get_device(input))


def ones(shape, dtype=np.complex, device=cpu_device):

    device = Device(device)
    xp = device.xp

    with device:
        return xp.ones(shape, dtype=dtype)


def ones_like(input):

    return ones(input.shape, dtype=input.dtype, device=get_device(input))


def dot(input1, input2):
    device = get_device(input1)
    xp = device.xp

    with device:
        return xp.real(xp.vdot(input1, input2))


def norm2(input):
    device = get_device(input)
    xp = device.xp

    return dot(input, input)


def norm(input):
    device = get_device(input)
    xp = device.xp

    with device:
        return norm2(input)**0.5


def monte_carlo_sure(f, y, sigma, eps=1e-10):
    '''
    Monte Carlo SURE.

    f - function x -> f(x)
    y - observed measurement
    sigma - noise standard deviation
    '''
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


def get_ugly_number(n):
    if n <= 1:
        return n

    ugly_nums = [1]
    i2, i3, i5 = 0, 0, 0
    while(True):

        ugly_num = min(ugly_nums[i2] * 2,
                       ugly_nums[i3] * 3,
                       ugly_nums[i5] * 5)

        if ugly_num >= n:
            return ugly_num

        ugly_nums.append(ugly_num)
        if ugly_num == ugly_nums[i2] * 2:
            i2 += 1
        elif ugly_num == ugly_nums[i3] * 3:
            i3 += 1
        elif ugly_num == ugly_nums[i5] * 5:
            i5 += 1


def axpy(y, a, x):

    device = get_device(x)
    x = move(x, device)
    a = move(a, device)

    with device:
        if device == cpu_device:
            _axpy(y, a, x, out=y)
        else:
            _axpy_cuda(y, a, x)


def xpay(y, a, x):

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
        '''
        y += (T) a * x;
        ''',
        name='axpy')

    _xpay_cuda = cp.ElementwiseKernel(
        'T y, S a, T x',
        '',
        '''
        y = x + (T) a * y;
        ''',
        name='axpy')
