# -*- coding: utf-8 -*-
"""Convolution functions with  multi-dimension, and multi-channel support.

"""
import numpy as np
import scipy.signal as signal
from sigpy import backend, util, config


__all__ = ['convolve', 'convolve_data_adjoint', 'convolve_filter_adjoint']


def convolve(data, filt,
             mode='full', strides=None,
             multi_channel=False):
    r"""Convolution that supports multi-dimensional and multi-channel inputs.

    This function follows the signal processing definition of convolution.

    Args:
        data (array): data array of shape:
            :math:`[..., m_1, ..., m_D]` if multi_channel is False,
            :math:`[..., c_i, m_1, ..., m_D]` otherwise.
        filt (array): filter array of shape:
            :math:`[n_1, ..., n_D]` if multi_channel is False
            :math:`[c_o, c_i, n_1, ..., n_D]` otherwise.
        mode (str): {'full', 'valid'}.
        strides (None or tuple of ints): convolution strides of length D.
        multi_channel (bool): specify if input/output has multiple channels.

    Returns:
        array: output array of shape:
            :math:`[..., p_1, ..., p_D]` if multi_channel is False,
            :math:`[..., c_o, p_1, ..., p_D]` otherwise.

    """
    device = backend.get_device(data)
    filt = backend.to_device(filt, device)
    with device:
        filt = filt.astype(data.dtype, copy=False)

    if device == backend.cpu_device:
        output = _convolve(data, filt, mode=mode, strides=strides,
                           multi_channel=multi_channel)
    else:  # pragma: no cover
        if config.cudnn_enabled:
            if np.issubdtype(data.dtype, np.floating):
                output = _convolve_cuda(data, filt,
                                        mode=mode, strides=strides,
                                        multi_channel=multi_channel)
            else:
                output = _complex(_convolve_cuda, data, filt,
                                  mode=mode, strides=strides,
                                  multi_channel=multi_channel)
        else:
            data = backend.to_device(data)
            filt = backend.to_device(filt)
            output = _convolve_data_adjoint(data, output,
                                            mode=mode, strides=strides,
                                            multi_channel=multi_channel)
            output = backend.to_device(output, device)

    return output


def convolve_data_adjoint(output, filt, data_shape,
                          mode='full', strides=None,
                          multi_channel=False):
    """Adjoint convolution operation with respect to data.

    Args:
        output (array): output array of shape
            :math:`[..., p_1, ..., p_D]` if multi_channel is False,
            :math:`[..., c_o, p_1, ..., p_D]` otherwise.
        filt (array): filter array of shape
            :math:`[n_1, ..., n_D]` if multi_channel is False
            :math:`[c_o, c_i, n_1, ..., n_D]` otherwise.
        mode (str): {'full', 'valid'}.
        strides (None or tuple of ints): convolution strides of length D.
        multi_channel (bool): specify if input/output has multiple channels.
        multi_channel (bool): specify if data/output has multiple channels.
        mode (str): {'full', 'valid'}.

    Returns:
        array: data array of shape
            :math:`[..., m_1, ..., m_D]` if multi_channel is False,
            :math:`[..., c_i, m_1, ..., m_D]` otherwise.

    """
    device = backend.get_device(output)
    data_shape = tuple(data_shape)
    filt = backend.to_device(filt, device)
    with device:
        filt = filt.astype(output.dtype, copy=False)

    if device == backend.cpu_device:
        data = _convolve_data_adjoint(output, filt, data_shape,
                                      mode=mode, strides=strides,
                                      multi_channel=multi_channel)
    else:  # pragma: no cover
        if config.cudnn_enabled:
            if np.issubdtype(output.dtype, np.floating):
                data = _convolve_data_adjoint_cuda(
                    output, filt, data_shape,
                    mode=mode, strides=strides,
                    multi_channel=multi_channel)
            else:
                data = _complex(_convolve_data_adjoint_cuda,
                                output, filt.conj(), data_shape,
                                mode=mode, strides=strides,
                                multi_channel=multi_channel)
        else:
            filt = backend.to_device(filt)
            output = backend.to_device(output)
            data = _convolve_data_adjoint(output, filt, data_shape,
                                          mode=mode, strides=strides,
                                          multi_channel=multi_channel)
            data = backend.to_device(output, device)

    return data


def convolve_filter_adjoint(output, data, filt_shape,
                            mode='full', strides=None,
                            multi_channel=False):
    """Adjoint convolution operation with respect to filter.

    Args:
        output (array): output array of shape:
            :math:`[..., p_1, ..., p_D]` if multi_channel is False,
            :math:`[..., c_o, p_1, ..., p_D]` otherwise.
        data (array): data array of shape:
            :math:`[..., m_1, ..., m_D]` if multi_channel is False,
            :math:`[..., c_i, m_1, ..., m_D]` otherwise.
        mode (str): {'full', 'valid'}.
        strides (None or tuple of ints): convolution strides of length D.
        multi_channel (bool): specify if input/output has multiple channels.

    Returns:
        array: filter array of shape:
            :math:`[n_1, ..., n_D]` if multi_channel is False
            :math:`[c_o, c_i, n_1, ..., n_D]` otherwise.

    """
    device = backend.get_device(output)
    filt_shape = tuple(filt_shape)
    data = backend.to_device(data, device)
    with device:
        data = data.astype(output.dtype, copy=False)

    if device == backend.cpu_device:
        filt = _convolve_filter_adjoint(output, data, filt_shape,
                                        mode=mode, strides=strides,
                                        multi_channel=multi_channel)
    else:  # pragma: no cover
        if config.cudnn_enabled:
            if np.issubdtype(output.dtype, np.floating):
                filt = _convolve_filter_adjoint_cuda(
                    output, data, filt_shape,
                    mode=mode, strides=strides,
                    multi_channel=multi_channel)
            else:
                filt = _complex(_convolve_filter_adjoint_cuda,
                                output, data.conj(), filt_shape,
                                mode=mode, strides=strides,
                                multi_channel=multi_channel)
        else:
            data = backend.to_device(data)
            output = backend.to_device(output)
            filt = _convolve_filter_adjoint(output, data, filt_shape,
                                            mode=mode, strides=strides,
                                            multi_channel=multi_channel)
            filt = backend.to_device(filt, device)

    return filt


def _get_convolve_params(data_shape, filt_shape,
                         mode, strides, multi_channel):
    D = len(filt_shape) - 2 * multi_channel
    m = tuple(data_shape[-D:])
    n = tuple(filt_shape[-D:])
    b = tuple(data_shape[:-D - multi_channel])
    B = util.prod(b)

    if multi_channel:
        if filt_shape[-D - 1] != data_shape[-D - 1]:
            raise ValueError('Data channel mismatch, '
                             'got {} from data and {} from filt.'.format(
                                 data_shape[-D - 1], filt_shape[-D - 1]))

        c_i = filt_shape[-D - 1]
        c_o = filt_shape[-D - 2]
    else:
        c_i = 1
        c_o = 1

    if strides is None:
        s = (1, ) * D
    else:
        if len(strides) != D:
            raise ValueError('Strides must have length {}.'.format(D))

        s = tuple(strides)

    if mode == 'full':
        p = tuple((m_d + n_d - 1 + s_d - 1) // s_d
                  for m_d, n_d, s_d in zip(m, n, s))
    elif mode == 'valid':
        if (any(m_d >= n_d for m_d, n_d in zip(m, n)) and
            any(m_d < n_d for m_d, n_d in zip(m, n))):
            raise ValueError('In valid mode, either data or filter must be '
                             'at least as large as the other in every axis.')

        p = tuple((m_d - n_d + 1 + s_d - 1) // s_d
                  for m_d, n_d, s_d in zip(m, n, s))
    else:
        raise ValueError('Invalid mode, got {}'.format(mode))

    return D, b, B, m, n, s, c_i, c_o, p


def _convolve(data, filt,
              mode='full', strides=None,
              multi_channel=False):
    D, b, B, m, n, s, c_i, c_o, p = _get_convolve_params(
        data.shape, filt.shape,
        mode, strides, multi_channel)

    # Normalize shapes.
    data = data.reshape((B, c_i) + m)
    filt = filt.reshape((c_o, c_i) + n)
    output = np.zeros((B, c_o) + p, dtype=data.dtype)
    slc = tuple(slice(None, None, s_d) for s_d in s)

    for k in range(B):
        for j in range(c_o):
            for i in range(c_i):
                output[k, j] += signal.convolve(
                    data[k, i], filt[j, i], mode=mode)[slc]

    # Reshape.
    if multi_channel:
        output = output.reshape(b + (c_o, ) + p)
    else:
        output = output.reshape(b + p)

    return output


def _convolve_data_adjoint(output, filt, data_shape,
                           mode='full', strides=None,
                           multi_channel=False):
    D, b, B, m, n, s, c_i, c_o, p = _get_convolve_params(
        data_shape, filt.shape,
        mode, strides, multi_channel)

    # Normalize shapes.
    output = output.reshape((B, c_o) + p)
    filt = filt.reshape((c_o, c_i) + n)
    data = np.zeros((B, c_i) + m, dtype=output.dtype)
    slc = tuple(slice(None, None, s_d) for s_d in s)
    if mode == 'full':
        output_kj = np.zeros([m_d + n_d - 1 for m_d, n_d in zip(m, n)],
                             dtype=output.dtype)
        adjoint_mode = 'valid'
    elif mode == 'valid':
        output_kj = np.zeros([max(m_d, n_d) - min(m_d, n_d) + 1
                              for m_d, n_d in zip(m, n)],
                             dtype=output.dtype)
        if all(m_d >= n_d for m_d, n_d in zip(m, n)):
            adjoint_mode = 'full'
        else:
            adjoint_mode = 'valid'

    for k in range(B):
        for j in range(c_o):
            for i in range(c_i):
                output_kj[slc] = output[k, j]
                data[k, i] += signal.correlate(
                    output_kj, filt[j, i],
                    mode=adjoint_mode)

    # Reshape.
    data = data.reshape(data_shape)
    return data


def _convolve_filter_adjoint(output, data, filt_shape,
                             mode='full', strides=None,
                             multi_channel=False):
    D, b, B, m, n, s, c_i, c_o, p = _get_convolve_params(
        data.shape, filt_shape,
        mode, strides, multi_channel)

    # Normalize shapes.
    data = data.reshape((B, c_i) + m)
    output = output.reshape((B, c_o) + p)
    slc = tuple(slice(None, None, s_d) for s_d in s)
    if mode == 'full':
        output_kj = np.zeros([m_d + n_d - 1 for m_d, n_d in zip(m, n)],
                             dtype=output.dtype)
        adjoint_mode = 'valid'
    elif mode == 'valid':
        output_kj = np.zeros([max(m_d, n_d) - min(m_d, n_d) + 1
                              for m_d, n_d in zip(m, n)],
                             dtype=output.dtype)
        if all(m_d >= n_d for m_d, n_d in zip(m, n)):
            adjoint_mode = 'valid'
        else:
            adjoint_mode = 'full'

    filt = np.zeros((c_o, c_i) + n, dtype=output.dtype)
    for k in range(B):
        for j in range(c_o):
            for i in range(c_i):
                output_kj[slc] = output[k, j]
                filt[j, i] += signal.correlate(
                    output_kj, data[k, i], mode=adjoint_mode)

    # Reshape.
    filt = filt.reshape(filt_shape)
    return filt


if config.cudnn_enabled:  # pragma: no cover
    from cupy import cudnn

    def _complex(func, data1, data2, *kargs, **kwargs):
        """Helper function to convert func to support complex floats.
        """
        device = backend.get_device(data1)
        xp = device.xp
        with device:
            data1r = xp.real(data1)
            data1i = xp.imag(data1)
            data2r = xp.real(data2)
            data2i = xp.imag(data2)

            outputr = func(data1r, data2r, *kargs, **kwargs)
            outputr -= func(data1i, data2i, *kargs, **kwargs)
            outputi = func(data1i, data2r, *kargs, **kwargs)
            outputi += func(data1r, data2i, *kargs, **kwargs)

            output = outputr + 1j * outputi
            output = output.astype(data1.dtype, copy=False)
            return output

    def _convolve_cuda(data, filt,
                       mode='full', strides=None,
                       multi_channel=False):
        device = backend.get_device(data)
        xp = device.xp

        D, b, B, m, n, s, c_i, c_o, p = _get_convolve_params(
            data.shape, filt.shape,
            mode, strides, multi_channel)
        dilations = (1, ) * D
        groups = 1
        auto_tune = True
        tensor_core = 'auto'
        if mode == 'full':
            pads = tuple(n_d - 1 for n_d in n)
        elif mode == 'valid':
            pads = (0, ) * D

        with device:
            data = data.reshape((B, c_i) + m)
            filt = filt.reshape((c_o, c_i) + n)
            output = xp.empty((B, c_o) + p, dtype=data.dtype)
            filt = util.flip(filt, axes=range(-D, 0))
            cudnn.convolution_forward(data, filt, None, output,
                                      pads, s, dilations, groups,
                                      auto_tune=auto_tune,
                                      tensor_core=tensor_core)

            # Reshape.
            if multi_channel:
                output = output.reshape(b + (c_o, ) + p)
            else:
                output = output.reshape(b + p)

        return output

    def _convolve_data_adjoint_cuda(output, filt, data_shape,
                                    mode='full', strides=None,
                                    multi_channel=False):
        device = backend.get_device(output)
        xp = device.xp

        D, b, B, m, n, s, c_i, c_o, p = _get_convolve_params(
            data_shape, filt.shape,
            mode, strides, multi_channel)
        dilations = (1, ) * D
        groups = 1
        auto_tune = True
        tensor_core = 'auto'
        deterministic = False
        if mode == 'full':
            pads = tuple(n_d - 1 for n_d in n)
        elif mode == 'valid':
            pads = (0, ) * D

        with device:
            output = output.reshape((B, c_o) + p)
            filt = filt.reshape((c_o, c_i) + n)
            data = xp.empty((B, c_i) + m, dtype=output.dtype)
            filt = util.flip(filt, axes=range(-D, 0))
            cudnn.convolution_backward_data(filt, output, None, data,
                                            pads, s, dilations, groups,
                                            deterministic=deterministic,
                                            auto_tune=auto_tune,
                                            tensor_core=tensor_core)

            # Reshape.
            data = data.reshape(data_shape)

        return data

    def _convolve_filter_adjoint_cuda(output, data, filt_shape,
                                      mode='full', strides=None,
                                      multi_channel=False):
        device = backend.get_device(data)
        xp = device.xp

        D, b, B, m, n, s, c_i, c_o, p = _get_convolve_params(
            data.shape, filt_shape,
            mode, strides, multi_channel)
        dilations = (1, ) * D
        groups = 1
        auto_tune = True
        tensor_core = 'auto'
        deterministic = False
        if mode == 'full':
            pads = tuple(n_d - 1 for n_d in n)
        elif mode == 'valid':
            pads = (0, ) * D

        with device:
            data = data.reshape((B, c_i) + m)
            output = output.reshape((B, c_o) + p)
            filt = xp.empty((c_o, c_i) + n, dtype=output.dtype)
            cudnn.convolution_backward_filter(data, output, filt,
                                              pads, s, dilations, groups,
                                              deterministic=deterministic,
                                              auto_tune=auto_tune,
                                              tensor_core=tensor_core)
            filt = util.flip(filt, axes=range(-D, 0))
            filt = filt.reshape(filt_shape)

        return filt
