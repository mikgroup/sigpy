# -*- coding: utf-8 -*-
"""Convolution functions.

This module contains convolution functions that support multi-dimension, and multi-channel.

"""
import numpy as np
from sigpy import backend, fourier, util, config

if config.cudnn_enabled:
    from cupy import cudnn


__all__ = ['convolve', 'convolve_adjoint_input', 'convolve_adjoint_filter']


def convolve(x, W, input_multi_channel=False, output_multi_channel=False, mode='full'):
    """Convolution that supports multi-dimensional and multi-channel inputs.

    Args:
        x (array): input array with shape batch_shape + input_shape 
            if input_multi_channel=False.
            Otherwise with shape batch_shape + [input_channel] + input_shape.
        W (array): filter array of shape filter_shape, 
            [input_channel] + filter_shape, [output_channel] + filter_shape,
            or [output_channel, input_channel] + filter_shape.
        input_multi_channel (bool): Specify whether input has multiple channels.
        output_multi_channel (bool): Specify whether output has multiple channels.
        mode (str): {'full', 'valid'}.

    Returns:
        array: output array with shape batch_shape + output_shape, 
            if output_multi_channel=False.
            Otherwise with shape batch_shape + [output_channel] + output_shape 
            if output_multi_channel=True.
            output_shape = input_shape + filter_shape - 1 if mode='full',
            output_shape = input_shape - filter_shape + 1 if mode='valid'.

    """
    ndim, input_shape, filter_shape, batch_shape, batch_size, \
        input_channel, output_channel = _get_convolve_params(
        x, W, input_multi_channel, output_multi_channel)

    device = backend.get_device(x)
    with device:
        x = x.reshape((batch_size, input_channel) + input_shape)
        W = W.reshape((output_channel, input_channel) + filter_shape)            
        
    if device != backend.cpu_device and config.cudnn_enabled:
        y = _cudnn_convolve(x, W, mode=mode)
    else:
        y = _fft_convolve(x, W, mode=mode)

    output_shape = y.shape[-ndim:]
    with device:
        if output_multi_channel:
            return y.reshape(batch_shape + (output_channel, ) + output_shape)
        else:
            return y.reshape(batch_shape + output_shape)


def convolve_adjoint_input(W, y, input_multi_channel=False,
                          output_multi_channel=False, mode='full'):
    """Convolution adjoint.

    Args:
        W (array): filter array of shape filter_shape, 
            [input_channel] + filter_shape, [output_channel] + filter_shape,
            or [output_channel, input_channel] + filter_shape.
        y (array): output array with shape batch_shape + output_shape 
            if output_multi_channel=False.
            Otherwise with shape batch_shape + [output_channel] + input_shape.
        input_multi_channel (bool): Specify whether input has multiple channels.
        output_multi_channel (bool): Specify whether output has multiple channels.
        mode (str): {'full', 'valid'}.

    Returns:
        array: output array with shape batch_shape + output_shape, 
            if output_multi_channel=False.
            Otherwise with shape batch_shape + [output_channel] + output_shape 
            if output_multi_channel=True.
            output_shape = input_shape + filter_shape - 1 if mode='full',
            output_shape = input_shape - filter_shape + 1 if mode='valid'.

    """
    ndim, output_shape, filter_shape, batch_shape, batch_size, \
        input_channel, output_channel = _get_convolve_adjoint_input_params(
            W, y, input_multi_channel, output_multi_channel)

    device = backend.get_device(y)
    with device:
        y = y.reshape((batch_size, output_channel) + output_shape)
        W = W.reshape((output_channel, input_channel) + filter_shape)            

    if device != backend.cpu_device and config.cudnn_enabled:
        x = _cudnn_convolve_adjoint_input(W, y, mode=mode)
    else:
        x = _fft_convolve_adjoint_input(W, y, mode=mode)

    input_shape = x.shape[-ndim:]
    with device:
        if input_multi_channel:
            return x.reshape(batch_shape + (input_channel, ) + input_shape)
        else:
            return x.reshape(batch_shape + input_shape)


def convolve_adjoint_filter(x, y, ndim, input_multi_channel=False,
                            output_multi_channel=False, mode='full'):
    """Convolution adjoint.

    Args:
        x (array): input array with shape batch_shape + input_shape 
            if input_multi_channel=False.
            Otherwise with shape batch_shape + [input_channel] + input_shape.
        y (array): output array with shape batch_shape + output_shape 
            if output_multi_channel=False.
            Otherwise with shape batch_shape + [output_channel] + input_shape.
        ndim (int): number of dimensions.
        input_multi_channel (bool): Specify whether input has multiple channels.
        output_multi_channel (bool): Specify whether output has multiple channels.
        mode (str): {'full', 'valid'}.

    Returns:
        array: filter array of shape filter_shape, 
            [input_channel] + filter_shape, [output_channel] + filter_shape,
            or [output_channel, input_channel] + filter_shape.

    """
    input_shape, output_shape, batch_shape, batch_size, \
        input_channel, output_channel = _get_convolve_adjoint_filter_params(
            x, y, ndim, input_multi_channel, output_multi_channel)

    device = backend.get_device(x)
    with device:
        x = x.reshape((batch_size, input_channel) + input_shape)
        y = y.reshape((batch_size, output_channel) + output_shape)          

    if device != backend.cpu_device and config.cudnn_enabled:
        W = _cudnn_convolve_adjoint_filter(x, y, mode=mode)
    else:
        W = _fft_convolve_adjoint_filter(x, y, mode=mode)

    with device:
        W_shape = W.shape[-ndim:]
        if input_multi_channel:
            W_shape = (input_channel, ) + W_shape

        if output_multi_channel:
            W_shape = (output_channel, ) + W_shape
            
        return W.reshape(W_shape)


def _get_convolve_params(x, W, input_multi_channel, output_multi_channel):
    ndim = W.ndim - input_multi_channel - output_multi_channel
    input_shape = x.shape[-ndim:]
    filter_shape = W.shape[-ndim:]
    batch_shape = x.shape[:-ndim - input_multi_channel]
    batch_size = util.prod(batch_shape)

    if input_multi_channel:
        input_channel = x.shape[-ndim - 1]
    else:
        input_channel = 1

    if output_multi_channel:
        output_channel = W.shape[-ndim - input_multi_channel - output_multi_channel]
    else:
        output_channel = 1

    return ndim, input_shape, filter_shape, batch_shape, \
        batch_size, input_channel, output_channel


def _get_convolve_adjoint_input_params(W, y, input_multi_channel, output_multi_channel):
    ndim = W.ndim - input_multi_channel - output_multi_channel
    output_shape = y.shape[-ndim:]
    filter_shape = W.shape[-ndim:]
    batch_shape = y.shape[:-ndim - output_multi_channel]
    batch_size = util.prod(batch_shape)

    if input_multi_channel:
        input_channel = W.shape[-ndim - 1]
    else:
        input_channel = 1

    if output_multi_channel:
        output_channel = y.shape[-ndim - 1]
    else:
        output_channel = 1

    return ndim, output_shape, filter_shape, batch_shape, \
        batch_size, input_channel, output_channel


def _get_convolve_adjoint_filter_params(x, y, ndim,
                                        input_multi_channel, output_multi_channel):
    output_shape = y.shape[-ndim:]
    input_shape = x.shape[-ndim:]
    batch_shape = y.shape[:-ndim - output_multi_channel]
    batch_size = util.prod(batch_shape)

    if input_multi_channel:
        input_channel = x.shape[-ndim - 1]
    else:
        input_channel = 1

    if output_multi_channel:
        output_channel = y.shape[-ndim - 1]
    else:
        output_channel = 1

    return input_shape, output_shape, batch_shape, \
        batch_size, input_channel, output_channel


def _fft_convolve(x, W, mode='full'):
    ndim = x.ndim - 2
    batch_size = len(x)
    output_channel, input_channel = W.shape[:2]
    input_shape = x.shape[-ndim:]
    filter_shape = W.shape[-ndim:]
    if mode == 'full':
        output_shape = tuple(m + n - 1 for m, n in zip(input_shape, filter_shape))
        pad_shape = output_shape
    elif mode == 'valid':
        output_shape = tuple(m - n + 1 for m, n in zip(input_shape, filter_shape))
        pad_shape = input_shape

    dtype = x.dtype
    device = backend.get_device(x)
    xp = device.xp
    with device:
        x = x.reshape((batch_size, 1, input_channel) + input_shape)
        x_pad = util.resize(x, (batch_size, 1, input_channel) + pad_shape,
                            oshift=[0] * x.ndim)
        W_pad = util.resize(W, (output_channel, input_channel) + pad_shape,
                            oshift=[0] * W.ndim)

        if np.issubdtype(dtype, np.floating):
            x_fft = xp.fft.rfftn(x_pad, axes=range(-ndim, 0), norm='ortho')
            W_fft = xp.fft.rfftn(W_pad, axes=range(-ndim, 0), norm='ortho')
            y_fft = xp.sum(x_fft * W_fft, axis=-ndim - 1)
            y = xp.fft.irfftn(y_fft, pad_shape,
                              axes=range(-ndim, 0), norm='ortho').astype(dtype)
        else:
            x_fft = fourier.fft(x_pad, axes=range(-ndim, 0), center=False)
            W_fft = fourier.fft(W_pad, axes=range(-ndim, 0), center=False)
            y_fft = xp.sum(x_fft * W_fft, axis=-ndim - 1)
            y = fourier.ifft(y_fft, axes=range(-ndim, 0), center=False)

        if mode == 'full':
            shift = [0] * y.ndim
        elif mode == 'valid':
            shift = [0, 0] + [n - 1 for n in filter_shape]

        y = util.resize(y, (batch_size, output_channel) + output_shape, ishift=shift)
        y *= util.prod(pad_shape)**0.5
        return y


def _fft_convolve_adjoint_input(W, y, mode='full'):
    ndim = y.ndim - 2
    batch_size = len(y)
    output_channel, input_channel = W.shape[:2]
    output_shape = y.shape[-ndim:]
    filter_shape = W.shape[-ndim:]
    if mode == 'full':
        input_shape = tuple(p - n + 1 for p, n in zip(output_shape, filter_shape))
        pad_shape = output_shape
    elif mode == 'valid':
        input_shape = tuple(p + n - 1 for p, n in zip(output_shape, filter_shape))
        pad_shape = input_shape

    dtype = y.dtype
    device = backend.get_device(y)
    xp = device.xp
    with device:
        y = y.reshape((batch_size, output_channel, 1) + output_shape)
        W = xp.conj(util.flip(W, axes=range(-ndim, 0)))
        
        y_pad = util.resize(y, (batch_size, output_channel, 1) + pad_shape,
                            oshift=[0] * y.ndim)
        W_pad = util.resize(W, (output_channel, input_channel) + pad_shape,
                            oshift=[0] * W.ndim)

        if np.issubdtype(dtype, np.floating):
            y_fft = xp.fft.rfftn(y_pad, axes=range(-ndim, 0), norm='ortho')
            W_fft = xp.fft.rfftn(W_pad, axes=range(-ndim, 0), norm='ortho')
            x_fft = xp.sum(y_fft * W_fft, axis=-ndim - 2)
            x = xp.fft.irfftn(x_fft, pad_shape, axes=range(-ndim, 0), norm='ortho').astype(dtype)
        else:
            y_fft = fourier.fft(y_pad, axes=range(-ndim, 0), center=False)
            W_fft = fourier.fft(W_pad, axes=range(-ndim, 0), center=False)
            x_fft = xp.sum(y_fft * W_fft, axis=-ndim - 2)
            x = fourier.ifft(x_fft, axes=range(-ndim, 0), center=False)

        if mode == 'full':
            shift = [0, 0] + [n - 1 for n in filter_shape]
        elif mode == 'valid':
            shift = [0] * x.ndim

        x = util.resize(x, (batch_size, input_channel) + input_shape, ishift=shift)
        x *= util.prod(pad_shape)**0.5
        return x


def _fft_convolve_adjoint_filter(x, y, mode='full'):
    ndim = x.ndim - 2
    batch_size = len(x)
    output_channel = y.shape[1]
    input_channel = x.shape[1]
    output_shape = y.shape[-ndim:]
    input_shape = x.shape[-ndim:]
    if mode == 'full':
        filter_shape = tuple(p - m + 1 for m, p in zip(input_shape, output_shape))
        pad_shape = output_shape
    elif mode == 'valid':
        filter_shape = tuple(m - p + 1 for m, p in zip(input_shape, output_shape))
        pad_shape = input_shape

    dtype = x.dtype
    device = backend.get_device(x)
    xp = device.xp
    with device:
        x = xp.conj(util.flip(x, axes=range(-ndim, 0)))
        x = x.reshape((batch_size, 1, input_channel) + input_shape)
        y = y.reshape((batch_size, output_channel, 1) + output_shape)
        
        x_pad = util.resize(x, (batch_size, 1, input_channel) + pad_shape,
                            oshift=[0] * x.ndim)
        y_pad = util.resize(y, (batch_size, output_channel, 1) + pad_shape,
                            oshift=[0] * y.ndim)

        if np.issubdtype(dtype, np.floating):
            x_fft = xp.fft.rfftn(x_pad, axes=range(-ndim, 0), norm='ortho')
            y_fft = xp.fft.rfftn(y_pad, axes=range(-ndim, 0), norm='ortho')
            W_fft = xp.sum(x_fft * y_fft, axis=0)
            W = xp.fft.irfftn(W_fft, pad_shape,
                              axes=range(-ndim, 0), norm='ortho').astype(dtype)
        else:
            x_fft = fourier.fft(x_pad, axes=range(-ndim, 0), center=False)
            y_fft = fourier.fft(y_pad, axes=range(-ndim, 0), center=False)
            W_fft = xp.sum(x_fft * y_fft, axis=0)
            W = fourier.ifft(W_fft, axes=range(-ndim, 0), center=False)

        if mode == 'full':
            shift = [0, 0] + [m - 1 for m in input_shape]
        elif mode == 'valid':
            shift = [0, 0] + [p - 1 for p in output_shape]

        W = util.resize(W, (output_channel, input_channel) + filter_shape, ishift=shift)
        W *= util.prod(pad_shape)**0.5
        return W
    

def _cudnn_convolve(x, W, mode='full'):
    dtype = x.dtype
    device = backend.get_device(x)
    xp = device.xp
    if np.issubdtype(dtype, np.complexfloating):
        with device:
            xr = xp.real(x)
            xi = xp.imag(x)
            Wr = xp.real(W)
            Wi = xp.imag(W)

            # Concatenate real and imaginary to input/output channels
            x = xp.concatenate([xr, xi], axis=1)
            W = xp.concatenate([xp.concatenate([Wr, -Wi], axis=1),
                                xp.concatenate([Wi, Wr], axis=1)], axis=0)

            y = _cudnn_convolve(x, W, mode=mode)

            # Convert back to complex
            return (y[:, :y.shape[1] // 2] + 1j * y[:, y.shape[1] // 2:]).astype(dtype)

    ndim = x.ndim - 2
    batch_size = len(x)
    output_channel = W.shape[0]
    input_shape = x.shape[-ndim:]
    filter_shape = W.shape[-ndim:]
    strides = (1, ) * ndim
    dilations = (1, ) * ndim
    groups = 1
    auto_tune = True
    tensor_core = 'auto'
    if mode == 'full':
        output_shape = tuple(m + n - 1 for m, n in zip(input_shape, filter_shape))
        pads = tuple(n - 1 for n in W.shape[2:])
    elif mode == 'valid':
        output_shape = tuple(m - n + 1 for m, n in zip(input_shape, filter_shape))
        pads = (0, ) * ndim

    with device:
        y = xp.empty((batch_size, output_channel) + output_shape, dtype=dtype)
        W = util.flip(W, axes=range(-ndim, 0))
        cudnn.convolution_forward(x, W, None, y,
                                  pads, strides, dilations, groups,
                                  auto_tune=auto_tune, tensor_core=tensor_core)

    return y
    

def _cudnn_convolve_adjoint_input(W, y, mode='full'):
    dtype = y.dtype
    device = backend.get_device(y)
    xp = device.xp
    if np.issubdtype(dtype, np.complexfloating):
        with device:
            Wr = xp.real(W)
            Wi = xp.imag(W)
            yr = xp.real(y)
            yi = xp.imag(y)

            # Concatenate real and imaginary to input/output channels
            y = xp.concatenate([yr, yi], axis=1)
            W = xp.concatenate([xp.concatenate([Wr, -Wi], axis=1),
                                xp.concatenate([Wi, Wr], axis=1)], axis=0)

            x = _cudnn_convolve_adjoint_input(W, y, mode=mode)

            # Convert back to complex
            return (x[:, :x.shape[1] // 2] + 1j * x[:, x.shape[1] // 2:]).astype(dtype)

    ndim = y.ndim - 2
    batch_size = len(y)
    input_channel = W.shape[1]
    output_shape = y.shape[-ndim:]
    filter_shape = W.shape[-ndim:]
    strides = (1, ) * ndim
    dilations = (1, ) * ndim
    groups = 1
    auto_tune = True
    tensor_core = 'auto'
    deterministic = False
    if mode == 'full':
        input_shape = tuple(p - n + 1 for p, n in zip(output_shape, filter_shape))
        pads = tuple(n - 1 for n in W.shape[2:])
    elif mode == 'valid':
        input_shape = tuple(p + n - 1 for p, n in zip(output_shape, filter_shape))
        pads = (0, ) * ndim

    with device:
        x = xp.empty((batch_size, input_channel) + input_shape, dtype=dtype)
        W = util.flip(W, axes=range(-ndim, 0))
        cudnn.convolution_backward_data(W, y, None, x,
                                        pads, strides, dilations, groups,
                                        deterministic=deterministic,
                                        auto_tune=auto_tune,
                                        tensor_core=tensor_core)

    return x
    

def _cudnn_convolve_adjoint_filter(x, y, mode='full'):
    dtype = y.dtype
    device = backend.get_device(y)
    xp = device.xp
    if np.issubdtype(dtype, np.complexfloating):
        with device:
            xr = xp.real(x)
            xi = xp.imag(x)
            yr = xp.real(y)
            yi = xp.imag(y)

            # Concatenate real and imaginary to input/output channels
            x = xp.concatenate([xr, xi], axis=1)
            y = xp.concatenate([yr, yi], axis=1)

            W = _cudnn_convolve_adjoint_filter(x, y, mode=mode)
    
            # Convert back to complex
            Wr = W[:W.shape[0] // 2, :W.shape[1] // 2]
            Wr += W[W.shape[0] // 2:, W.shape[1] // 2:]
            Wi = W[W.shape[0] // 2:, :W.shape[1] // 2]
            Wi -= W[:W.shape[0] // 2, W.shape[1] // 2:]
            return (Wr + 1j * Wi).astype(dtype)

    ndim = y.ndim - 2
    batch_size = len(y)
    input_channel = x.shape[1]
    output_channel = y.shape[1]
    input_shape = x.shape[-ndim:]
    output_shape = y.shape[-ndim:]
    strides = (1, ) * ndim
    dilations = (1, ) * ndim
    groups = 1
    auto_tune = True
    tensor_core = 'auto'
    deterministic = False
    if mode == 'full':
        filter_shape = tuple(p - m + 1 for m, p in zip(input_shape, output_shape))
        pads = tuple(n - 1 for n in filter_shape)
    elif mode == 'valid':
        filter_shape = tuple(m - p + 1 for m, p in zip(input_shape, output_shape))
        pads = (0, ) * ndim

    with device:
        W = xp.empty((output_channel, input_channel) + filter_shape, dtype=dtype)
        cudnn.convolution_backward_filter(x, y, W,
                                          pads, strides, dilations, groups,
                                          deterministic=deterministic,
                                          auto_tune=auto_tune,
                                          tensor_core=tensor_core)
        W = util.flip(W, axes=range(-ndim, 0))

    return W
