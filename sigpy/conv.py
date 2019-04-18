# -*- coding: utf-8 -*-
"""Convolution functions.

This module contains convolution functions that support multi-dimension,
and multi-channel.

"""
import numpy as np
from sigpy import backend, fourier, util, config


__all__ = ['convolve', 'convolve_adjoint_input', 'convolve_adjoint_filter']


def convolve(input, filt, input_multi_channel=False,
             output_multi_channel=False, mode='full',
             strides=None):
    """Convolution that supports multi-dimensional and multi-channel inputs.

    This convolution follows the signal processing convention.

    Args:
        input (array): input array with shape batch_shape + ishape
            if input_multi_channel=False.
            Otherwise with shape batch_shape + [input_channel] + ishape.
        filt (array): filter array of shape fshape,
            [input_channel] + fshape, [output_channel] + fshape,
            or [output_channel, input_channel] + fshape.
        input_multi_channel (bool): Specify if input has multiple channels.
        output_multi_channel (bool): Specify if output has multiple channels.
        mode (str): {'full', 'valid'}.

    Returns:
        array: output array with shape batch_shape + oshape,
            if output_multi_channel=False.
            Otherwise with shape batch_shape + [output_channel] + oshape
            if output_multi_channel=True.
            oshape = ishape + fshape - 1 if mode='full',
            oshape = ishape - fshape + 1 if mode='valid'.

    """
    ndim, ishape, fshape, batch_shape, batch_size, \
        input_channel, output_channel = _get_convolve_params(
            input, filt, input_multi_channel, output_multi_channel)

    device = backend.get_device(input)
    filt = backend.to_device(filt, device)
    with device:
        input = input.reshape((batch_size, input_channel) + ishape)
        filt = filt.reshape((output_channel, input_channel) + fshape)
        filt = filt.astype(input.dtype, copy=False)

    if (device != backend.cpu_device
        and config.cudnn_enabled):  # pragma: no cover
        output = _cudnn_convolve(input, filt, mode=mode, strides=strides)
    else:
        output = _fft_convolve(input, filt, mode=mode, strides=strides)

    oshape = output.shape[-ndim:]
    with device:
        if output_multi_channel:
            return output.reshape(
                batch_shape + (output_channel, ) + oshape)
        else:
            return output.reshape(batch_shape + oshape)


def convolve_adjoint_input(filt, output, input_multi_channel=False,
                           output_multi_channel=False, mode='full'):
    """Convolution adjoint.

    Args:
        filt (array): filter array of shape fshape,
            [input_channel] + fshape, [output_channel] + fshape,
            or [output_channel, input_channel] + fshape.
        y (array): output array with shape batch_shape + oshape
            if output_multi_channel=False.
            Otherwise with shape batch_shape + [output_channel] + ishape.
        input_multi_channel (bool): Specify if input has multiple channels.
        output_multi_channel (bool): Specify if output has multiple channels.
        mode (str): {'full', 'valid'}.

    Returns:
        array: output array with shape batch_shape + oshape,
            if output_multi_channel=False.
            Otherwise with shape batch_shape + [output_channel] + oshape
            if output_multi_channel=True.
            oshape = ishape + fshape - 1 if mode='full',
            oshape = ishape - fshape + 1 if mode='valid'.

    """
    ndim, oshape, fshape, batch_shape, batch_size, \
        input_channel, output_channel = _get_convolve_adjoint_input_params(
            filt, output, input_multi_channel, output_multi_channel)

    device = backend.get_device(filt)
    output = backend.to_device(output, device)
    with device:
        filt = filt.reshape((output_channel, input_channel) + fshape)
        output = output.reshape((batch_size, output_channel) + oshape)
        output = output.astype(filt.dtype, copy=False)

    if (device != backend.cpu_device
        and config.cudnn_enabled):  # pragma: no cover
        input = _cudnn_convolve_adjoint_input(filt, output, mode=mode)
    else:
        input = _fft_convolve_adjoint_input(filt, output, mode=mode)

    ishape = input.shape[-ndim:]
    with device:
        if input_multi_channel:
            return input.reshape(
                batch_shape + (input_channel, ) + ishape)
        else:
            return input.reshape(batch_shape + ishape)


def convolve_adjoint_filter(input, output, ndim, input_multi_channel=False,
                            output_multi_channel=False, mode='full'):
    """Convolution adjoint.

    Args:
        input (array): input array with shape batch_shape + ishape
            if input_multi_channel=False.
            Otherwise with shape batch_shape + [input_channel] + ishape.
        y (array): output array with shape batch_shape + oshape
            if output_multi_channel=False.
            Otherwise with shape batch_shape + [output_channel] + ishape.
        ndim (int): number of dimensions.
        input_multi_channel (bool): Specify if input has multiple channels.
        output_multi_channel (bool): Specify if output has multiple channels.
        mode (str): {'full', 'valid'}.

    Returns:
        array: filter array of shape fshape,
            [input_channel] + fshape, [output_channel] + fshape,
            or [output_channel, input_channel] + fshape.

    """
    ishape, oshape, batch_shape, batch_size, \
        input_channel, output_channel = _get_convolve_adjoint_filter_params(
            input, output, ndim, input_multi_channel, output_multi_channel)

    device = backend.get_device(input)
    output = backend.to_device(output, device)
    with device:
        input = input.reshape((batch_size, input_channel) + ishape)
        output = output.reshape((batch_size, output_channel) + oshape)
        output = output.astype(input.dtype, copy=False)

    if (device != backend.cpu_device
        and config.cudnn_enabled):  # pragma: no cover
        filt = _cudnn_convolve_adjoint_filter(input, output, mode=mode)
    else:
        filt = _fft_convolve_adjoint_filter(input, output, mode=mode)

    with device:
        filt_shape = filt.shape[-ndim:]
        if input_multi_channel:
            filt_shape = (input_channel, ) + filt_shape

        if output_multi_channel:
            filt_shape = (output_channel, ) + filt_shape

        return filt.reshape(filt_shape)


def _get_convolve_params(input, filt,
                         input_multi_channel, output_multi_channel):
    ndim = filt.ndim - input_multi_channel - output_multi_channel
    ishape = input.shape[-ndim:]
    fshape = filt.shape[-ndim:]
    batch_shape = input.shape[:-ndim - input_multi_channel]
    batch_size = util.prod(batch_shape)

    if input_multi_channel:
        input_channel = input.shape[-ndim - 1]
    else:
        input_channel = 1

    if output_multi_channel:
        output_channel = filt.shape[
            -ndim - input_multi_channel - output_multi_channel]
    else:
        output_channel = 1

    return ndim, ishape, fshape, batch_shape, \
        batch_size, input_channel, output_channel


def _get_convolve_adjoint_input_params(
        filt, output, input_multi_channel, output_multi_channel):
    ndim = filt.ndim - input_multi_channel - output_multi_channel
    oshape = output.shape[-ndim:]
    fshape = filt.shape[-ndim:]
    batch_shape = output.shape[:-ndim - output_multi_channel]
    batch_size = util.prod(batch_shape)

    if input_multi_channel:
        input_channel = filt.shape[-ndim - 1]
    else:
        input_channel = 1

    if output_multi_channel:
        output_channel = output.shape[-ndim - 1]
    else:
        output_channel = 1

    return ndim, oshape, fshape, batch_shape, \
        batch_size, input_channel, output_channel


def _get_convolve_adjoint_filter_params(
        input, output, ndim, input_multi_channel, output_multi_channel):
    oshape = output.shape[-ndim:]
    ishape = input.shape[-ndim:]
    batch_shape = output.shape[:-ndim - output_multi_channel]
    batch_size = util.prod(batch_shape)

    if input_multi_channel:
        input_channel = input.shape[-ndim - 1]
    else:
        input_channel = 1

    if output_multi_channel:
        output_channel = output.shape[-ndim - 1]
    else:
        output_channel = 1

    return ishape, oshape, batch_shape, \
        batch_size, input_channel, output_channel


def _fft_convolve(input, filt, mode='full', strides=None):
    ndim = input.ndim - 2
    batch_size = len(input)
    output_channel, input_channel = filt.shape[:2]
    ishape = input.shape[-ndim:]
    fshape = filt.shape[-ndim:]
    if mode == 'full':
        oshape = tuple(
            m + n - 1 for m, n in zip(ishape, fshape))
        pad_shape = oshape
    elif mode == 'valid':
        oshape = tuple(
            m - n + 1 for m, n in zip(ishape, fshape))
        pad_shape = ishape

    dtype = input.dtype
    device = backend.get_device(input)
    xp = device.xp
    with device:
        input = input.reshape((batch_size, 1, input_channel) + ishape)
        input_pad = util.resize(
            input, (batch_size, 1, input_channel) + pad_shape,
            oshift=[0] * input.ndim)
        filt_pad = util.resize(
            filt, (output_channel, input_channel) + pad_shape,
            oshift=[0] * filt.ndim)

        if np.issubdtype(dtype, np.floating):
            input_fft = xp.fft.rfftn(
                input_pad, axes=range(-ndim, 0), norm='ortho')
            filt_fft = xp.fft.rfftn(
                filt_pad, axes=range(-ndim, 0), norm='ortho')
            y_fft = xp.sum(input_fft * filt_fft, axis=-ndim - 1)
            output = xp.fft.irfftn(
                y_fft, pad_shape,
                axes=range(-ndim, 0), norm='ortho').astype(dtype)
        else:
            input_fft = fourier.fft(
                input_pad, axes=range(-ndim, 0), center=False)
            filt_fft = fourier.fft(
                filt_pad, axes=range(-ndim, 0), center=False)
            y_fft = xp.sum(input_fft * filt_fft, axis=-ndim - 1)
            output = fourier.ifft(y_fft, axes=range(-ndim, 0), center=False)

        if mode == 'full':
            shift = [0] * output.ndim
        elif mode == 'valid':
            shift = [0, 0] + [n - 1 for n in fshape]

        output = util.resize(
            output, (batch_size, output_channel) + oshape,
            ishift=shift)
        output *= util.prod(pad_shape)**0.5

        if strides is not None:
            slc = (slice(None), slice(None))
            slc += tuple(slice(None, None, s) for s in strides)
            output = output[slc]

        return output


def _fft_convolve_adjoint_input(filt, output, mode='full'):
    ndim = output.ndim - 2
    batch_size = len(output)
    output_channel, input_channel = filt.shape[:2]
    oshape = output.shape[-ndim:]
    fshape = filt.shape[-ndim:]
    if mode == 'full':
        ishape = tuple(
            p - n + 1 for p, n in zip(oshape, fshape))
        pad_shape = oshape
    elif mode == 'valid':
        ishape = tuple(
            p + n - 1 for p, n in zip(oshape, fshape))
        pad_shape = ishape

    dtype = output.dtype
    device = backend.get_device(filt)
    xp = device.xp
    with device:
        filt = xp.conj(util.flip(filt, axes=range(-ndim, 0)))
        output = output.reshape((batch_size, output_channel, 1) + oshape)

        y_pad = util.resize(
            output, (batch_size, output_channel, 1) + pad_shape,
            oshift=[0] * output.ndim)
        filt_pad = util.resize(
            filt, (output_channel, input_channel) + pad_shape,
            oshift=[0] * filt.ndim)

        if np.issubdtype(dtype, np.floating):
            y_fft = xp.fft.rfftn(y_pad, axes=range(-ndim, 0), norm='ortho')
            filt_fft = xp.fft.rfftn(
                filt_pad, axes=range(-ndim, 0), norm='ortho')
            input_fft = xp.sum(y_fft * filt_fft, axis=-ndim - 2)
            input = xp.fft.irfftn(
                input_fft, pad_shape,
                axes=range(-ndim, 0), norm='ortho').astype(dtype)
        else:
            y_fft = fourier.fft(y_pad, axes=range(-ndim, 0), center=False)
            filt_fft = fourier.fft(
                filt_pad, axes=range(-ndim, 0), center=False)
            input_fft = xp.sum(y_fft * filt_fft, axis=-ndim - 2)
            input = fourier.ifft(
                input_fft, axes=range(-ndim, 0), center=False)

        if mode == 'full':
            shift = [0, 0] + [n - 1 for n in fshape]
        elif mode == 'valid':
            shift = [0] * input.ndim

        input = util.resize(
            input, (batch_size, input_channel) + ishape, ishift=shift)
        input *= util.prod(pad_shape)**0.5
        return input


def _fft_convolve_adjoint_filter(input, output, mode='full'):
    ndim = input.ndim - 2
    batch_size = len(input)
    output_channel = output.shape[1]
    input_channel = input.shape[1]
    oshape = output.shape[-ndim:]
    ishape = input.shape[-ndim:]
    if mode == 'full':
        fshape = tuple(
            p - m + 1 for m, p in zip(ishape, oshape))
        pad_shape = oshape
    elif mode == 'valid':
        fshape = tuple(
            m - p + 1 for m, p in zip(ishape, oshape))
        pad_shape = ishape

    dtype = input.dtype
    device = backend.get_device(input)
    xp = device.xp
    with device:
        input = xp.conj(util.flip(input, axes=range(-ndim, 0)))
        input = input.reshape((batch_size, 1, input_channel) + ishape)
        output = output.reshape((batch_size, output_channel, 1) + oshape)

        input_pad = util.resize(
            input, (batch_size, 1, input_channel) + pad_shape,
            oshift=[0] * input.ndim)
        y_pad = util.resize(
            output, (batch_size, output_channel, 1) + pad_shape,
            oshift=[0] * output.ndim)

        if np.issubdtype(dtype, np.floating):
            input_fft = xp.fft.rfftn(
                input_pad, axes=range(-ndim, 0), norm='ortho')
            y_fft = xp.fft.rfftn(y_pad, axes=range(-ndim, 0), norm='ortho')
            filt_fft = xp.sum(input_fft * y_fft, axis=0)
            filt = xp.fft.irfftn(
                filt_fft, pad_shape,
                axes=range(-ndim, 0), norm='ortho').astype(dtype)
        else:
            input_fft = fourier.fft(
                input_pad, axes=range(-ndim, 0), center=False)
            y_fft = fourier.fft(y_pad, axes=range(-ndim, 0), center=False)
            filt_fft = xp.sum(input_fft * y_fft, axis=0)
            filt = fourier.ifft(filt_fft, axes=range(-ndim, 0), center=False)

        if mode == 'full':
            shift = [0, 0] + [m - 1 for m in ishape]
        elif mode == 'valid':
            shift = [0, 0] + [p - 1 for p in oshape]

        filt = util.resize(
            filt, (output_channel, input_channel) + fshape, ishift=shift)
        filt *= util.prod(pad_shape)**0.5
        return filt


if config.cudnn_enabled:  # pragma: no cover
    from cupy import cudnn

    def _cudnn_convolve(input, filt, mode='full', strides=None):
        dtype = input.dtype
        device = backend.get_device(input)
        xp = device.xp
        if np.issubdtype(dtype, np.complexfloating):
            with device:
                inputr = xp.real(input)
                inputi = xp.imag(input)
                filtr = xp.real(filt)
                filti = xp.imag(filt)

                # Concatenate real and imaginary to input/output channels
                input = xp.concatenate([inputr, inputi], axis=1)
                filt = xp.concatenate(
                    [xp.concatenate([filtr, -filti], axis=1),
                     xp.concatenate([filti, filtr], axis=1)], axis=0)

                output = _cudnn_convolve(
                    input, filt, mode=mode, strides=strides)

                # Convert back to complex
                output_channel = output.shape[1] // 2
                output = output[:, :output_channel] + 1j * output[
                    :, output_channel:]
                output = output.astype(dtype)

                return output

        ndim = input.ndim - 2
        batch_size = len(input)
        output_channel = filt.shape[0]
        ishape = input.shape[-ndim:]
        fshape = filt.shape[-ndim:]
        if strides is None:
            strides = (1, ) * ndim

        dilations = (1, ) * ndim
        groups = 1
        auto_tune = True
        tensor_core = 'auto'
        if mode == 'full':
            oshape = tuple(
                (m + n - 1 + s - 1) // s
                for m, n, s in zip(ishape, fshape, strides))
            pads = tuple(n - 1 for n in filt.shape[2:])
        elif mode == 'valid':
            oshape = tuple(
                (m - n + 1 + s - 1) // s
                for m, n, s in zip(ishape, fshape, strides))
            pads = (0, ) * ndim

        with device:
            output = xp.empty((batch_size, output_channel) + oshape,
                              dtype=dtype)
            filt = util.flip(filt, axes=range(-ndim, 0))
            cudnn.convolution_forward(input, filt, None, output,
                                      pads, strides, dilations, groups,
                                      auto_tune=auto_tune,
                                      tensor_core=tensor_core)

        return output

    def _cudnn_convolve_adjoint_input(
            filt, output, mode='full', strides=None):
        dtype = output.dtype
        device = backend.get_device(output)
        xp = device.xp
        if np.issubdtype(dtype, np.complexfloating):
            with device:
                filtr = xp.real(filt)
                filti = xp.imag(filt)
                outputr = xp.real(output)
                outputi = xp.imag(output)

                # Concatenate real and imaginary to input/output channels
                output = xp.concatenate([outputr, outputi], axis=1)
                filt = xp.concatenate(
                    [xp.concatenate([filtr, -filti], axis=1),
                     xp.concatenate([filti, filtr], axis=1)], axis=0)

                input = _cudnn_convolve_adjoint_input(filt, output, mode=mode)

                # Convert back to complex
                input_channel = input.shape[1] // 2
                input = input[:, :input_channel] + 1j * input[
                    :, input_channel:]
                input = input.astype(dtype)

                return input

        ndim = output.ndim - 2
        batch_size = len(output)
        input_channel = filt.shape[1]
        oshape = output.shape[-ndim:]
        fshape = filt.shape[-ndim:]
        if strides is None:
            strides = (1, ) * ndim

        dilations = (1, ) * ndim
        groups = 1
        auto_tune = True
        tensor_core = 'auto'
        deterministic = False
        if mode == 'full':
            ishape = tuple(
                p - n + 1 for p, n in zip(oshape, fshape))
            pads = tuple(n - 1 for n in filt.shape[2:])
        elif mode == 'valid':
            ishape = tuple(
                p + n - 1 for p, n in zip(oshape, fshape))
            pads = (0, ) * ndim

        with device:
            input = xp.empty((batch_size, input_channel) + ishape,
                             dtype=dtype)
            filt = util.flip(filt, axes=range(-ndim, 0))
            cudnn.convolution_backward_data(filt, output, None, input,
                                            pads, strides, dilations, groups,
                                            deterministic=deterministic,
                                            auto_tune=auto_tune,
                                            tensor_core=tensor_core)

        return input

    def _cudnn_convolve_adjoint_filter(input, output,
                                       strides=None, mode='full'):
        dtype = output.dtype
        device = backend.get_device(output)
        xp = device.xp
        if np.issubdtype(dtype, np.complexfloating):
            with device:
                inputr = xp.real(input)
                inputi = xp.imag(input)
                outputr = xp.real(output)
                outputi = xp.imag(output)

                # Concatenate real and imaginary to input/output channels
                input = xp.concatenate([inputr, inputi], axis=1)
                output = xp.concatenate([outputr, outputi], axis=1)

                filt = _cudnn_convolve_adjoint_filter(
                    input, output, mode=mode)

                # Convert back to complex
                filtr = filt[:filt.shape[0] // 2, :filt.shape[1] // 2]
                filtr += filt[filt.shape[0] // 2:, filt.shape[1] // 2:]
                filti = filt[filt.shape[0] // 2:, :filt.shape[1] // 2]
                filti -= filt[:filt.shape[0] // 2, filt.shape[1] // 2:]
                return (filtr + 1j * filti).astype(dtype)

        ndim = output.ndim - 2
        input_channel = input.shape[1]
        output_channel = output.shape[1]
        ishape = input.shape[-ndim:]
        oshape = output.shape[-ndim:]
        if strides is None:
            strides = (1, ) * ndim

        dilations = (1, ) * ndim
        groups = 1
        auto_tune = True
        tensor_core = 'auto'
        deterministic = False
        if mode == 'full':
            fshape = tuple(
                p - m + 1 for m, p in zip(ishape, oshape))
            pads = tuple(n - 1 for n in fshape)
        elif mode == 'valid':
            fshape = tuple(
                m - p + 1 for m, p in zip(ishape, oshape))
            pads = (0, ) * ndim

        with device:
            filt = xp.empty(
                (output_channel, input_channel) + fshape, dtype=dtype)
            cudnn.convolution_backward_filter(
                input, output, filt,
                pads, strides, dilations, groups,
                deterministic=deterministic,
                auto_tune=auto_tune,
                tensor_core=tensor_core)
            filt = util.flip(filt, axes=range(-ndim, 0))

        return filt
