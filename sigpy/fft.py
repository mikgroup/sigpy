import numpy as np

from sigpy import config, util
if config.cupy_enabled:
    import cupy as cp


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
    device = util.get_device(input)
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
    device = util.get_device(input)
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


def _fftc(input, oshape=None, axes=None, norm='ortho'):

    ndim = input.ndim
    axes = util._normalize_axes(axes, ndim)
    device = util.get_device(input)
    xp = device.xp

    if oshape is None:
        oshape = input.shape

    with device:
        tmp = input
        tshape = list(input.shape)
        for a in axes:
            i = oshape[a]
            tshape[a] = i
            idx = xp.arange(i, dtype=input.dtype)

            tmp = tmp.swapaxes(a, -1)
            tshape[a], tshape[-1] = tshape[-1], tshape[a]

            tmp = util.resize(tmp, tshape)
            tmp = xp.fft.ifftshift(tmp, axes=-1)
            tmp = xp.fft.fft(tmp, axis=-1, norm=norm)
            tmp = xp.fft.fftshift(tmp, axes=-1)

            tmp = tmp.swapaxes(a, -1)
            tshape[a], tshape[-1] = tshape[-1], tshape[a]

        output = tmp

    return output


def _ifftc(input, oshape=None, axes=None, norm='ortho'):

    ndim = input.ndim
    axes = util._normalize_axes(axes, ndim)
    device = util.get_device(input)
    xp = device.xp

    if oshape is None:
        oshape = input.shape

    with device:
        tmp = input
        tshape = list(input.shape)
        for a in axes:

            i = oshape[a]
            tshape[a] = i
            idx = xp.arange(i, dtype=input.dtype)

            tmp = tmp.swapaxes(a, -1)
            tshape[a], tshape[-1] = tshape[-1], tshape[a]

            tmp = util.resize(tmp, tshape)
            tmp = xp.fft.ifftshift(tmp, axes=-1)
            tmp = xp.fft.ifft(tmp, axis=-1, norm=norm)
            tmp = xp.fft.fftshift(tmp, axes=-1)

            tmp = tmp.swapaxes(a, -1)
            tshape[a], tshape[-1] = tshape[-1], tshape[a]

        output = tmp

    return output
