# -*- coding: utf-8 -*-
"""Wavelet transform functions.
"""
import numpy as np
import pywt
from sigpy import backend

__all__ = ['fwt', 'iwt']


def get_wavelet_shape(shape, wave_name, axes, level):
    input = np.zeros(shape)
    tmp = fwt(input, wave_name=wave_name, axes=axes, level=level)

    return tmp.shape


def apply_dec_along_axis(input, axes, dec_lo, dec_hi, level, apply_zpad):
    """Apply wavelet decomposition along axes.

    Helper function to recursively apply decomposition wavelet filters
    along axes.

    Args:
        input (array): Input array.
        axes (tuple of int): Axes to perform wavelet transform.
        dec_lo (array): Wavelet coefficients for approximation coefficients.
        dec_hi (array): Wavelet coefficients for decimation coefficients.
        level (int): Level to determine amount of zero-padding.
        apply_zpad (bool): Set to true to apply z-pad.
    """
    assert type(axes) == tuple
    assert dec_lo.shape == dec_hi.shape

    if (len(axes) == 0):
        return input

    # Loading sigpy.
    device = backend.get_device(input)
    xp = device.xp

    axis = axes[0]

    # Zero padding.
    x = input
    if (apply_zpad):
        pad_size = (1 + (dec_hi.size * level + x.shape[axis])//(2**level)) * \
                    2 ** level - x.shape[axis]
        pad_array = [(0, pad_size) if k == axis else (0, 0)
                     for k in range(len(x.shape))]
        x = xp.pad(x, pad_array, 'constant', constant_values=(0, 0))

    # Fourier space.
    X = xp.fft.fftn(x, axes=(axis,))

    lo = xp.zeros((x.shape[axis],)).astype(xp.complex64)
    lo[:dec_lo.size] = dec_lo
    lo = xp.reshape(xp.fft.fftn(xp.roll(lo, -(dec_lo.size//2)), axes=(0,)),
                    [lo.size if k == axis else 1 for k in range(len(x.shape))])

    hi = xp.zeros((x.shape[axis],)).astype(xp.complex64)
    hi[:dec_hi.size] = dec_hi
    hi = xp.reshape(xp.fft.fftn(xp.roll(hi, -(dec_hi.size//2)), axes=(0,)),
                    [hi.size if k == axis else 1 for k in range(len(x.shape))])

    # Apply convolutions.
    y_lo = xp.fft.ifftn(X * lo, axes=(axis,))
    y_hi = xp.fft.ifftn(X * hi, axes=(axis,))

    # Sub-sampling
    y_lo = xp.take(y_lo, [t * 2 for t in range(0, y_lo.shape[axis]//2)],
                   axis=axis)
    y_hi = xp.take(y_hi, [t * 2 for t in range(0, y_hi.shape[axis]//2)],
                   axis=axis)

    # Apply recursion to other axis and concatenate.
    return xp.concatenate((apply_dec_along_axis(y_lo, axes[1:], dec_lo,
                           dec_hi, level, apply_zpad),
                           apply_dec_along_axis(y_hi, axes[1:], dec_lo,
                           dec_hi, level, apply_zpad)), axis=axis)


def apply_rec_along_axis(input, axes, rec_lo, rec_hi):
    """Apply wavelet recomposition along axes.

    Helper function to recursively apply decomposition wavelet filters
    along axes. Assumes input has been appropriately zero-padded by
    apply_dec_along_axis (used by fwt).

    Args:
        input (array): Input array.
        axes (tuple of int): Axes to perform wavelet transform.
        rec_lo (array): Wavelet coefficients for approximation coefficients.
        rec_hi (array): Wavelet coefficients for decimation coefficients.
    """
    assert type(axes) == tuple
    assert rec_lo.shape == rec_hi.shape

    if (len(axes) == 0):
        return input

    # Load sigpy.
    device = backend.get_device(input)
    xp = device.xp

    axis = axes[0]

    # Preparing filters.
    lo = xp.zeros((input.shape[axis],)).astype(xp.complex64)
    lo[:rec_lo.size] = rec_lo
    lo = xp.reshape(xp.fft.fftn(xp.roll(lo, 1-(rec_lo.size//2)), axes=(0,)),
                    [lo.size if k == axis else 1
                     for k in range(len(input.shape))])

    hi = xp.zeros((input.shape[axis],)).astype(xp.complex64)
    hi[:rec_hi.size] = rec_hi
    hi = xp.reshape(xp.fft.fftn(xp.roll(hi, 1-(rec_hi.size//2)), axes=(0,)),
                    [hi.size if k == axis else 1
                     for k in range(len(input.shape))])

    # Coefficient indices.
    lo_coeffs = tuple([slice(0, input.shape[k]//2)
                       if k == axis else slice(0, None)
                       for k in range(len(input.shape))])
    hi_coeffs = tuple([slice(input.shape[k]//2, None)
                       if k == axis else slice(0, None)
                       for k in range(len(input.shape))])

    # Extracting coefficients.
    x_lo = xp.zeros(input.shape).astype(xp.complex64)
    x_hi = xp.zeros(input.shape).astype(xp.complex64)

    sample_idx = tuple([slice(0, None, 2)
                        if k == axis else slice(0, None)
                        for k in range(len(input.shape))])
    x_lo[sample_idx] = input[lo_coeffs]
    x_hi[sample_idx] = input[hi_coeffs]

    # Apply convolutions.
    X_lo = xp.fft.fftn(x_lo, axes=(axis,))
    X_hi = xp.fft.fftn(x_hi, axes=(axis,))
    y_lo = xp.fft.ifftn(X_lo * lo, axes=(axis,))
    y_hi = xp.fft.ifftn(X_hi * hi, axes=(axis,))

    # Apply recursion to other axis and concatenate.
    return apply_rec_along_axis(y_lo + y_hi, axes[1:], rec_lo, rec_hi)


def fwt(input, wave_name='db4', axes=None, level=None, apply_zpad=True):
    """Forward wavelet transform.

    Args:
        input (array): Input array.
        wave_name (str): Wavelet name.
        axes (None or tuple of int): Axes to perform wavelet transform.
        level (None or int): Number of wavelet levels.
        apply_zpad (bool): If true, zero-pad for linear convolution.
    """
    device = backend.get_device(input)
    xp = device.xp

    if axes is None:
        axes = tuple([k for k in range(len(input.shape))
                      if input.shape[k] > 1])

    if (type(axes) == int):
        axes = (axes,)

    wavdct = pywt.Wavelet(wave_name)
    dec_lo = xp.array(wavdct.dec_lo)
    dec_hi = xp.array(wavdct.dec_hi)

    if level is None:
        level = pywt.dwt_max_level(
                    xp.min(xp.array([input.shape[ax] for ax in axes])),
                    dec_lo.size)

    if level <= 0:
        return input

    assert level > 0

    y = apply_dec_along_axis(input, axes, dec_lo, dec_hi, level, apply_zpad)
    approx_idx = tuple([slice(0, y.shape[k]//2)
                        if k in axes else slice(0, None)
                        for k in range(len(input.shape))])
    y[approx_idx] = fwt(y[approx_idx], wave_name=wave_name,
                        axes=axes, level=level-1, apply_zpad=False)

    return y


def iwt(input, oshape, wave_name='db4', axes=None, level=None, inplace=False):
    """Inverse wavelet transform.

    Args:
        input (array): Input array.
        oshape (tuple): Output shape.
        wave_name (str): Wavelet name.
        axes (None or tuple of int): Axes to perform wavelet transform.
        level (None or int): Number of wavelet levels.
        inplace (bool): Modify input array in place.
    """
    device = backend.get_device(input)
    xp = device.xp

    if axes is None:
        axes = tuple([k for k in range(len(input.shape))
                      if input.shape[k] > 1])

    if (type(axes) == int):
        axes = (axes,)

    wavdct = pywt.Wavelet(wave_name)
    rec_lo = xp.array(wavdct.rec_lo)
    rec_hi = xp.array(wavdct.rec_hi)

    if level is None:
        level = pywt.dwt_max_level(
                    xp.min(xp.array([input.shape[ax] for ax in axes])),
                    rec_lo.size)

    if level <= 0:
        return input

    assert level > 0
    for ax in axes:
        assert input.shape[ax] % 2 == 0

    x = input if inplace else input.astype(xp.complex64).copy()

    approx_idx = tuple([slice(0, input.shape[k]//2)
                        if k in axes else slice(0, None)
                        for k in range(len(input.shape))])
    x[approx_idx] = iwt(x[approx_idx], input[approx_idx].shape,
                        wave_name=wave_name, axes=axes, level=level-1,
                        inplace=True)

    y = apply_rec_along_axis(x, axes, rec_lo, rec_hi)
    crop_idx = tuple([slice(0, oshape[k])
                      if k in axes else slice(0, None)
                      for k in range(len(input.shape))])

    return y[crop_idx]
