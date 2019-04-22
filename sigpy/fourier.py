# -*- coding: utf-8 -*-
"""FFT and non-uniform FFT (NUFFT) functions.

"""
import numpy as np

from sigpy import backend, interp, util


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

        if np.issubdtype(input.dtype,
                         np.complexfloating) and input.dtype != output.dtype:
            output = output.astype(input.dtype, copy=False)

        return output


def ifft(input, oshape=None, axes=None, center=True, norm='ortho'):
    """IFFT function that supports centering.

    Args:
        input (array): input array.
        oshape (None or array of ints): output shape.
        axes (None or array of ints): Axes over which to compute
            the inverse FFT.
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

        if np.issubdtype(input.dtype,
                         np.complexfloating) and input.dtype != output.dtype:
            output = output.astype(input.dtype)

        return output


def nufft(input, coord, oversamp=1.25, width=4.0, n=128):
    """Non-uniform Fast Fourier Transform.

    Args:
        input (array): input signal domain array of shape
            (..., n_{ndim - 1}, ..., n_1, n_0),
            where ndim is specified by coord.shape[-1]. The nufft
            is applied on the last ndim axes, and looped over
            the remaining axes.
        coord (array): Fourier domain coordinate array of shape (..., ndim).
            ndim determines the number of dimensions to apply the nufft.
            coord[..., i] should be scaled to have its range between
            -n_i // 2, and n_i // 2.
        oversamp (float): oversampling factor.
        width (float): interpolation kernel full-width in terms of
            oversampled grid.
        n (int): number of sampling points of the interpolation kernel.

    Returns:
        array: Fourier domain data of shape
            input.shape[:-ndim] + coord.shape[:-1].

    References:
        Fessler, J. A., & Sutton, B. P. (2003).
        Nonuniform fast Fourier transforms using min-max interpolation
        IEEE Transactions on Signal Processing, 51(2), 560-574.
        Beatty, P. J., Nishimura, D. G., & Pauly, J. M. (2005).
        Rapid gridding reconstruction with a minimal oversampling ratio.
        IEEE transactions on medical imaging, 24(6), 799-808.

    """
    device = backend.get_device(input)
    ndim = coord.shape[-1]
    beta = np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5
    os_shape = _get_oversamp_shape(input.shape, ndim, oversamp)

    with device:
        output = input.copy()

        # Apodize
        _apodize(output, ndim, oversamp, width, beta)

        # Zero-pad
        output /= util.prod(input.shape[-ndim:])**0.5
        output = util.resize(output, os_shape)

        # FFT
        output = fft(output, axes=range(-ndim, 0), norm=None)

        # Interpolate
        coord = _scale_coord(backend.to_device(
            coord, device), input.shape, oversamp)
        kernel = _get_kaiser_bessel_kernel(n, width, beta, coord.dtype, device)
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
        shape = [int(coord[..., i].max() - coord[..., i].min())
                 for i in range(ndim)]

    return shape


def nufft_adjoint(input, coord, oshape=None, oversamp=1.25, width=4.0, n=128):
    """Adjoint non-uniform Fast Fourier Transform.

    Args:
        input (array): input Fourier domain array of shape
            (...) + coord.shape[:-1]. That is, the last dimensions
            of input must match the first dimensions of coord.
            The nufft_adjoint is applied on the last coord.ndim - 1 axes,
            and looped over the remaining axes.
        coord (array): Fourier domain coordinate array of shape (..., ndim).
            ndim determines the number of dimension to apply nufft adjoint.
            coord[..., i] should be scaled to have its range between
            -n_i // 2, and n_i // 2.
        oshape (tuple of ints): output shape of the form
            (..., n_{ndim - 1}, ..., n_1, n_0).
        oversamp (float): oversampling factor.
        width (float): interpolation kernel full-width in terms of
            oversampled grid.
        n (int): number of sampling points of the interpolation kernel.

    Returns:
        array: signal domain array with shape specified by oshape.

    See Also:
        :func:`sigpy.nufft.nufft`

    """
    device = backend.get_device(input)
    ndim = coord.shape[-1]
    beta = np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5
    if oshape is None:
        oshape = list(input.shape[:-coord.ndim + 1]) + estimate_shape(coord)
    else:
        oshape = list(oshape)

    os_shape = _get_oversamp_shape(oshape, ndim, oversamp)

    with device:
        # Gridding
        coord = _scale_coord(backend.to_device(
            coord, device), oshape, oversamp)
        kernel = _get_kaiser_bessel_kernel(n, width, beta, coord.dtype, device)
        output = interp.gridding(input, os_shape, width, kernel, coord)

        # IFFT
        output = ifft(output, axes=range(-ndim, 0), norm=None)

        # Crop
        output = util.resize(output, oshape)
        output *= util.prod(os_shape[-ndim:]) / util.prod(oshape[-ndim:])**0.5

        # Apodize
        _apodize(output, ndim, oversamp, width, beta)

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


def _get_kaiser_bessel_kernel(n, width, beta, dtype, device):
    """Precompute Kaiser Bessel kernel.

    Precomputes Kaiser-Bessel kernel with n points.

    Args:
        n (int): number of sampling points.
        width (float): kernel width.
        beta (float): kaiser bessel parameter.
        dtype (dtype): output data type.
        device (Device): output device.

    Returns:
        array: Kaiser-Bessel kernel table.

    """
    device = backend.Device(device)
    xp = device.xp
    with device:
        x = xp.arange(n, dtype=dtype) / n
        kernel = 1 / width * xp.i0(beta * (1 - x**2)**0.5).astype(dtype)
        return kernel


def _scale_coord(coord, shape, oversamp):
    ndim = coord.shape[-1]
    device = backend.get_device(coord)
    scale = backend.to_device(
        [_get_ugly_number(oversamp * i) / i for i in shape[-ndim:]], device)
    shift = backend.to_device(
        [_get_ugly_number(oversamp * i) // 2 for i in shape[-ndim:]], device)

    with device:
        coord = scale * coord + shift

    return coord


def _get_ugly_number(n):
    """Get closest ugly number greater than n.

    An ugly number is defined as a positive integer that is a
    multiple of 2 3 and 5.

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


def _get_oversamp_shape(shape, ndim, oversamp):
    return list(shape)[:-ndim] + [_get_ugly_number(oversamp * i)
                                  for i in shape[-ndim:]]


def _apodize(input, ndim, oversamp, width, beta):
    device = backend.get_device(input)
    xp = device.xp

    output = input
    with device:
        for a in range(-ndim, 0):
            i = output.shape[a]
            os_i = _get_ugly_number(oversamp * i)
            idx = xp.arange(i, dtype=output.dtype)

            # Calculate apodization
            apod = (beta**2 - (np.pi * width * (idx - i // 2) / os_i)**2)**0.5
            apod /= xp.sinh(apod)
            output *= apod.reshape([i] + [1] * (-a - 1))

        return output
