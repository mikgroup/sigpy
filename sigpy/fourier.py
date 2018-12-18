# -*- coding: utf-8 -*-
"""FFT functions.

This module contains FFT functions that support centered operation.

"""
import numpy as np

from sigpy import backend, config, interp, util
if config.cupy_enabled:
    import cupy as cp


__all__ = ['fft', 'ifft', 'nufft', 'nufft_adjoint', 'estimate_shape']


def fft(input, oshape=None, axes=None, center=True, norm='ortho'):
    """FFT function that supports centering.

    Args:
        input (array): input array.
        oshape (None or array of ints): output shape.
        axes (None or array of ints): Axes over which to compute the FFT.
        norm (Nonr or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        array: FFT result of dimension oshape.

    See Also:
        :func:`numpy.fft.fftn`

    """
    device = backend.get_device(input)
    xp = device.xp

    with device:
        if not np.issubdtype(input.dtype, np.complexfloating):
            input = input.astype(np.complex)

        if center:
            output = _fftc(input, oshape=oshape, axes=axes, norm=norm)
        else:
            output = xp.fft.fftn(input, s=oshape, axes=axes, norm=norm)

        if np.issubdtype(input.dtype, np.complexfloating) and input.dtype != output.dtype:
            output = output.astype(input.dtype)

        return output


def ifft(input, oshape=None, axes=None, center=True, norm='ortho'):
    """IFFT function that supports centering.

    Args:
        input (array): input array.
        oshape (None or array of ints): output shape.
        axes (None or array of ints): Axes over which to compute the inverse FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        array of dimension oshape.

    See Also:
        :func:`numpy.fft.ifftn`

    """
    device = backend.get_device(input)
    xp = device.xp

    with device:
        if not np.issubdtype(input.dtype, np.complexfloating):
            input = input.astype(np.complex)

        if center:
            output = _ifftc(input, oshape=oshape, axes=axes, norm=norm)
        else:
            output = xp.fft.ifftn(input, s=oshape, axes=axes, norm=norm)

        if np.issubdtype(input.dtype, np.complexfloating) and input.dtype != output.dtype:
            output = output.astype(input.dtype)

        return output


def nufft(input, coord, oversamp=1.25, width=4.0, n=128):
    """Non-uniform Fast Fourier Transform.

    Args:
        input (array): input array.
        coord (array): coordinate array of shape (..., ndim). 
            ndim determines the number of dimension to apply nufft.
        oversamp (float): oversampling factor.
        width (float): interpolation kernel full-width in terms of oversampled grid.
        n (int): number of sampling points of interpolation kernel.

    Returns:
        array: Fourier domain points of shape input.shape[:-ndim] + coord.shape[:-1]

    References:
        Fessler, J. A., & Sutton, B. P. (2003). 
        Nonuniform fast Fourier transforms using min-max interpolation. 
        IEEE Transactions on Signal Processing, 51(2), 560-574.

        Beatty, P. J., Nishimura, D. G., & Pauly, J. M. (2005). 
        Rapid gridding reconstruction with a minimal oversampling ratio. 
        IEEE transactions on medical imaging, 24(6), 799-808.

    """
    device = backend.get_device(input)
    xp = device.xp
    ndim = coord.shape[-1]
    beta = np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5

    with device:
        output = input.copy()
        os_shape = list(input.shape)

        for a in range(-ndim, 0):
            i = input.shape[a]
            os_i = _get_ugly_number(oversamp * i)
            os_shape[a] = os_i
            idx = xp.arange(i, dtype=input.dtype)

            # Calculate apodization
            apod = (beta**2 - (np.pi * width * (idx - i // 2) / os_i)**2)**0.5
            apod /= xp.sinh(apod)

            # Swap axes
            output = output.swapaxes(a, -1)
            os_shape[a], os_shape[-1] = os_shape[-1], os_shape[a]

            # Apodize
            output *= apod

            # Oversampled FFT
            output = util.resize(output, os_shape)
            output = fft(output, axes=[-1], norm=None)
            output /= i**0.5

            # Swap back
            output = output.swapaxes(a, -1)
            os_shape[a], os_shape[-1] = os_shape[-1], os_shape[a]

        coord = _scale_coord(backend.to_device(coord, device), input.shape, oversamp)
        kernel = backend.to_device(
            _kb(np.arange(n, dtype=coord.dtype) / n, width, beta, coord.dtype), device)

        output = interp.interpolate(output, width, kernel, coord)

        return output


def estimate_shape(coord):
    """Estimate array shape from coordinates.

    Shape is estimated by the different between maximum and minimum of
    coordinates in each axis.

    Args:
        coord (array): Coordinates.
    """
    ndim = coord.shape[-1]
    with backend.get_device(coord):
        shape = [int(coord[..., i].max() - coord[..., i].min()) for i in range(ndim)]

    return shape


def nufft_adjoint(input, coord, oshape=None, oversamp=1.25, width=4.0, n=128):
    """Adjoint non-uniform Fast Fourier Transform.

    Args:
        input (array): Input Fourier domain array.
        coord (array): coordinate array of shape (..., ndim). 
            ndim determines the number of dimension to apply nufft adjoint.
        oshape (tuple of ints): output shape.
        oversamp (float): oversampling factor.
        width (float): interpolation kernel full-width in terms of oversampled grid.
        n (int): number of sampling points of interpolation kernel.

    Returns:
        array: Transformed array.

    See Also:
        :func:`sigpy.nufft.nufft`

    """
    device = backend.get_device(input)
    xp = device.xp
    ndim = coord.shape[-1]
    beta = np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5
    if oshape is None:
        oshape = list(input.shape[:-coord.ndim + 1]) + estimate_shape(coord)
    else:
        oshape = list(oshape)

    with device:
        coord = _scale_coord(backend.to_device(coord, device), oshape, oversamp)
        kernel = backend.to_device(
            _kb(np.arange(n, dtype=coord.dtype) / n, width, beta, coord.dtype), device)
        os_shape = oshape[:-ndim] + [_get_ugly_number(oversamp * i) for i in oshape[-ndim:]]
        output = interp.gridding(input, os_shape, width, kernel, coord)

        for a in range(-ndim, 0):
            i = oshape[a]
            os_i = os_shape[a]
            idx = xp.arange(i, dtype=input.dtype)
            os_shape[a] = i

            # Swap axes
            output = output.swapaxes(a, -1)
            os_shape[a], os_shape[-1] = os_shape[-1], os_shape[a]

            # Oversampled IFFT
            output = ifft(output, axes=[-1], norm=None)
            output *= os_i / i**0.5
            output = util.resize(output, os_shape)

            # Calculate apodization
            apod = (beta**2 - (np.pi * width * (idx - i // 2) / os_i)**2)**0.5
            apod /= xp.sinh(apod)

            # Apodize
            output *= apod

            # Swap back
            output = output.swapaxes(a, -1)
            os_shape[a], os_shape[-1] = os_shape[-1], os_shape[a]

        return output


def _fftc(input, oshape=None, axes=None, norm='ortho'):

    ndim = input.ndim
    axes = util._normalize_axes(axes, ndim)
    device = backend.get_device(input)
    xp = device.xp

    if oshape is None:
        oshape = input.shape
        
    with device:
        tmp = util.resize(input, oshape)
        tmp = xp.fft.ifftshift(tmp, axes=axes)
        tmp = xp.fft.fftn(tmp, axes=axes, norm=norm)
        output = xp.fft.fftshift(tmp, axes=axes)
        return output


def _ifftc(input, oshape=None, axes=None, norm='ortho'):
    ndim = input.ndim
    axes = util._normalize_axes(axes, ndim)
    device = backend.get_device(input)
    xp = device.xp

    if oshape is None:
        oshape = input.shape

    with device:
        tmp = util.resize(input, oshape)
        tmp = xp.fft.ifftshift(tmp, axes=axes)
        tmp = xp.fft.ifftn(tmp, axes=axes, norm=norm)
        output = xp.fft.fftshift(tmp, axes=axes)
        return output


def _kb(x, width, beta, dtype):
    return 1 / width * np.i0(beta * (1 - x**2)**0.5).astype(dtype)


def _scale_coord(coord, shape, oversamp):
    ndim = coord.shape[-1]
    device = backend.get_device(coord)
    scale = backend.to_device([_get_ugly_number(oversamp * i) / i for i in shape[-ndim:]], device)
    shift = backend.to_device([_get_ugly_number(oversamp * i) // 2 for i in shape[-ndim:]], device)

    with device:
        coord = scale * coord + shift

    return coord


def _get_ugly_number(n):
    """Get closest ugly number greater than n.

    An ugly number is defined as a positive integer that is a multiple of 2, 3, and 5.
    
    Args:
        n (int): Base number.

    """
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
