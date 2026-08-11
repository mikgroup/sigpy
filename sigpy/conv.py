# -*- coding: utf-8 -*-
"""Convolution functions with  multi-dimension, and multi-channel support.

"""
import numpy as np
import scipy.signal as signal

from sigpy import backend, config, util

__all__ = ["convolve", "convolve_data_adjoint", "convolve_filter_adjoint"]


def convolve(data, filt, mode="full", strides=None, multi_channel=False):
    r"""Convolution that supports multi-dimensional and multi-channel inputs.

    This function follows the signal processing definition of convolution.
    Note that cuDNN only supports inputs with D=1, 2 or 3.
    If data is a cupy array, and if you have cupy<14, it uses cupy.cudnn;
        if cupy>=14 and cudnn frontend is installed, it uses cudnn frontend;
        otherwise it uses fft based convolution.

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
    xp = backend.get_array_module(data)
    if xp == np:
        output = _convolve(
            data, filt, mode=mode, strides=strides, multi_channel=multi_channel
        )
    elif config.cupycudnn_enabled:  # pragma: no cover
        if np.issubdtype(data.dtype, np.floating):
            output = _convolve_cuda(
                data,
                filt,
                mode=mode,
                strides=strides,
                multi_channel=multi_channel,
            )
        else:
            output = _complex(
                _convolve_cuda,
                data,
                filt,
                mode=mode,
                strides=strides,
                multi_channel=multi_channel,
            )
    elif config.cudnn_frontend_enabled and _cudnn_fe_supported(
        filt.shape, multi_channel
    ):  # pragma: no cover
        if np.issubdtype(data.dtype, np.floating):
            output = _convolve_cudnn_fe(
                data,
                filt,
                mode=mode,
                strides=strides,
                multi_channel=multi_channel,
            )
        else:
            output = _complex(
                _convolve_cudnn_fe,
                data,
                filt,
                mode=mode,
                strides=strides,
                multi_channel=multi_channel,
            )
    else:  # pragma: no cover
        output = _convolve_fft(
            data, filt, mode=mode, strides=strides, multi_channel=multi_channel
        )

    return output


def convolve_data_adjoint(
    output, filt, data_shape, mode="full", strides=None, multi_channel=False
):
    """Adjoint convolution operation with respect to data.

    Note that cuDNN only supports inputs with D=1, 2 or 3.
    If data is a cupy array, and if you have cupy<14, it uses cupy.cudnn;
        if cupy>=14 and cudnn frontend is installed, it uses cudnn frontend;
        otherwise it uses fft based convolution.

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
    data_shape = tuple(data_shape)

    xp = backend.get_array_module(output)
    if xp == np:
        data = _convolve_data_adjoint(
            output,
            filt,
            data_shape,
            mode=mode,
            strides=strides,
            multi_channel=multi_channel,
        )
    elif config.cupycudnn_enabled:  # pragma: no cover
        if np.issubdtype(output.dtype, np.floating):
            data = _convolve_data_adjoint_cuda(
                output,
                filt,
                data_shape,
                mode=mode,
                strides=strides,
                multi_channel=multi_channel,
            )
        else:
            data = _complex(
                _convolve_data_adjoint_cuda,
                output,
                filt.conj(),
                data_shape,
                mode=mode,
                strides=strides,
                multi_channel=multi_channel,
            )
    elif config.cudnn_frontend_enabled and _cudnn_fe_supported(
        filt.shape, multi_channel
    ):  # pragma: no cover
        if np.issubdtype(output.dtype, np.floating):
            data = _convolve_data_adjoint_cudnn_fe(
                output,
                filt,
                data_shape,
                mode=mode,
                strides=strides,
                multi_channel=multi_channel,
            )
        else:
            data = _complex(
                _convolve_data_adjoint_cudnn_fe,
                output,
                filt.conj(),
                data_shape,
                mode=mode,
                strides=strides,
                multi_channel=multi_channel,
            )
    else:  # pragma: no cover
        data = _convolve_data_adjoint_fft(
            output,
            filt,
            data_shape,
            mode=mode,
            strides=strides,
            multi_channel=multi_channel,
        )

    return data


def convolve_filter_adjoint(
    output, data, filt_shape, mode="full", strides=None, multi_channel=False
):
    """Adjoint convolution operation with respect to filter.

    Note that cuDNN only supports inputs with D=1, 2 or 3.
    If data is a cupy array, and if you have cupy<14, it uses cupy.cudnn;
        if cupy>=14 and cudnn frontend is installed, it uses cudnn frontend;
        otherwise it uses fft based convolution.

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
    filt_shape = tuple(filt_shape)
    xp = backend.get_array_module(data)
    if xp == np:
        filt = _convolve_filter_adjoint(
            output,
            data,
            filt_shape,
            mode=mode,
            strides=strides,
            multi_channel=multi_channel,
        )
    elif config.cupycudnn_enabled:  # pragma: no cover
        if np.issubdtype(output.dtype, np.floating):
            filt = _convolve_filter_adjoint_cuda(
                output,
                data,
                filt_shape,
                mode=mode,
                strides=strides,
                multi_channel=multi_channel,
            )
        else:
            filt = _complex(
                _convolve_filter_adjoint_cuda,
                output,
                data.conj(),
                filt_shape,
                mode=mode,
                strides=strides,
                multi_channel=multi_channel,
            )
    elif config.cudnn_frontend_enabled and _cudnn_fe_supported(
        filt_shape, multi_channel
    ):  # pragma: no cover
        if np.issubdtype(output.dtype, np.floating):
            filt = _convolve_filter_adjoint_cudnn_fe(
                output,
                data,
                filt_shape,
                mode=mode,
                strides=strides,
                multi_channel=multi_channel,
            )
        else:
            filt = _complex(
                _convolve_filter_adjoint_cudnn_fe,
                output,
                data.conj(),
                filt_shape,
                mode=mode,
                strides=strides,
                multi_channel=multi_channel,
            )
    else:  # pragma: no cover
        filt = _convolve_filter_adjoint_fft(
            output,
            data,
            filt_shape,
            mode=mode,
            strides=strides,
            multi_channel=multi_channel,
        )

    return filt


def _get_convolve_params(data_shape, filt_shape, mode, strides, multi_channel):
    D = len(filt_shape) - 2 * multi_channel
    m = tuple(data_shape[-D:])
    n = tuple(filt_shape[-D:])
    b = tuple(data_shape[: -D - multi_channel])
    B = util.prod(b)

    if multi_channel:
        if filt_shape[-D - 1] != data_shape[-D - 1]:
            raise ValueError(
                "Data channel mismatch, "
                "got {} from data and {} from filt.".format(
                    data_shape[-D - 1], filt_shape[-D - 1]
                )
            )

        c_i = filt_shape[-D - 1]
        c_o = filt_shape[-D - 2]
    else:
        c_i = 1
        c_o = 1

    if strides is None:
        s = (1,) * D
    else:
        if len(strides) != D:
            raise ValueError("Strides must have length {}.".format(D))

        s = tuple(strides)

    if mode == "full":
        p = tuple(
            (m_d + n_d - 1 + s_d - 1) // s_d for m_d, n_d, s_d in zip(m, n, s)
        )
    elif mode == "valid":
        if any(m_d >= n_d for m_d, n_d in zip(m, n)) and any(
            m_d < n_d for m_d, n_d in zip(m, n)
        ):
            raise ValueError(
                "In valid mode, either data or filter must be "
                "at least as large as the other in every axis."
            )

        p = tuple(
            (m_d - n_d + 1 + s_d - 1) // s_d for m_d, n_d, s_d in zip(m, n, s)
        )
    else:
        raise ValueError("Invalid mode, got {}".format(mode))

    return D, b, B, m, n, s, c_i, c_o, p


def _valid_idx(len1, len2):
    """Selecting the valid region of the full linear convolution.


    """
    return slice(min(len1, len2) - 1, max(len1, len2))


def _fft_convolve_sum(x, y, fshape):
    """Convolve two stacks over their trailing axes and contract the rest.

    Args:
        x (array): array of shape :math:`[P, Q, \\ldots]`.
        y (array): array of shape :math:`[R, Q, \\ldots]`.
        fshape (tuple of ints): linear convolution shape, of length D.

    Returns:
        array: array of shape :math:`[P, R] + fshape`, whose :math:`[a, b]`
        entry is the sum over :math:`q` of the linear convolutions of
        :math:`x[a, q]` and :math:`y[b, q]`.

    Doing the whole :math:`[P, R, Q]` contraction as one batched matrix
    multiply in the frequency domain is what keeps this from degenerating
    into a Python loop over channels.

    """
    xp = backend.get_array_module(x)
    D = len(fshape)
    axes = tuple(range(-D, 0))
    real = not (
        xp.iscomplexobj(x) or xp.iscomplexobj(y)
    )  # rfftn saves half the work and returns real without a cast.
    fftn = xp.fft.rfftn if real else xp.fft.fftn
    ifftn = xp.fft.irfftn if real else xp.fft.ifftn

    x_fft = fftn(x, s=fshape, axes=axes)
    y_fft = fftn(y, s=fshape, axes=axes)

    P, Q = x.shape[:2]
    R = y.shape[0]
    grid = x_fft.shape[-D:]
    G = util.prod(grid)

    # Batched matmul over the transformed grid: [G, P, Q] @ [G, Q, R].
    x_fft = x_fft.reshape((P, Q, G)).transpose(2, 0, 1)
    y_fft = y_fft.reshape((R, Q, G)).transpose(2, 1, 0)
    out_fft = xp.matmul(x_fft, y_fft)
    out_fft = out_fft.transpose(1, 2, 0).reshape((P, R) + grid)

    return ifftn(out_fft, s=fshape, axes=axes)


def _convolve_fft(data, filt, mode="full", strides=None, multi_channel=False):
    D, b, B, m, n, s, c_i, c_o, p = _get_convolve_params(
        data.shape, filt.shape, mode, strides, multi_channel
    )

    # Normalize shapes.
    data = data.reshape((B, c_i) + m)
    filt = filt.reshape((c_o, c_i) + n)

    fshape = tuple(m_d + n_d - 1 for m_d, n_d in zip(m, n))
    output = _fft_convolve_sum(data, filt, fshape)

    if mode == "full":
        slc = tuple(slice(None, None, s_d) for s_d in s)
    else:
        slc = tuple(
            slice(_valid_idx(m_d, n_d).start, m_d, s_d)
            for m_d, n_d, s_d in zip(m, n, s)
        )

    output = output[(slice(None), slice(None)) + slc]
    output = output.astype(data.dtype, copy=False)

    # Reshape.
    if multi_channel:
        output = output.reshape(b + (c_o,) + p)
    else:
        output = output.reshape(b + p)

    return output


def _adjoint_output(output, m, n, p, s, mode, B, c_o):
    """Zero-fill a strided output back onto the unstrided adjoint grid."""
    xp = backend.get_array_module(output)
    if mode == "full":
        u = tuple(m_d + n_d - 1 for m_d, n_d in zip(m, n))
    else:
        u = tuple(max(m_d, n_d) - min(m_d, n_d) + 1 for m_d, n_d in zip(m, n))

    output_kj = xp.zeros((B, c_o) + u, dtype=output.dtype)
    slc = tuple(slice(None, None, s_d) for s_d in s)
    output_kj[(slice(None), slice(None)) + slc] = output.reshape((B, c_o) + p)
    return output_kj, u


def _convolve_data_adjoint_fft(
    output, filt, data_shape, mode="full", strides=None, multi_channel=False
):
    D, b, B, m, n, s, c_i, c_o, p = _get_convolve_params(
        data_shape, filt.shape, mode, strides, multi_channel
    )

    filt = filt.reshape((c_o, c_i) + n)
    output_kj, u = _adjoint_output(output, m, n, p, s, mode, B, c_o)

    # correlate(x, y) == convolve(x, conj(flip(y))), and the sum runs over
    # the output channels, so the filter is transposed to [c_i, c_o, ...].
    filt = util.flip(filt, axes=range(-D, 0)).conj().swapaxes(0, 1)

    fshape = tuple(u_d + n_d - 1 for u_d, n_d in zip(u, n))
    data = _fft_convolve_sum(output_kj, filt, fshape)

    if mode == "full" or any(m_d < n_d for m_d, n_d in zip(m, n)):
        slc = tuple(_valid_idx(u_d, n_d) for u_d, n_d in zip(u, n))
        data = data[(slice(None), slice(None)) + slc]

    data = data.astype(output.dtype, copy=False)
    return data.reshape(data_shape)


def _convolve_filter_adjoint_fft(
    output, data, filt_shape, mode="full", strides=None, multi_channel=False
):
    D, b, B, m, n, s, c_i, c_o, p = _get_convolve_params(
        data.shape, filt_shape, mode, strides, multi_channel
    )

    data = data.reshape((B, c_i) + m)
    output_kj, u = _adjoint_output(output, m, n, p, s, mode, B, c_o)

    # Here the sum runs over the batch, so both stacks are transposed to put
    # the batch axis second.
    data = util.flip(data, axes=range(-D, 0)).conj().swapaxes(0, 1)
    output_kj = output_kj.swapaxes(0, 1)

    fshape = tuple(u_d + m_d - 1 for u_d, m_d in zip(u, m))
    filt = _fft_convolve_sum(output_kj, data, fshape)

    if mode == "full" or all(m_d >= n_d for m_d, n_d in zip(m, n)):
        slc = tuple(_valid_idx(u_d, m_d) for u_d, m_d in zip(u, m))
        filt = filt[(slice(None), slice(None)) + slc]

    filt = filt.astype(output.dtype, copy=False)
    return filt.reshape(filt_shape)


def _convolve(data, filt, mode="full", strides=None, multi_channel=False):
    D, b, B, m, n, s, c_i, c_o, p = _get_convolve_params(
        data.shape, filt.shape, mode, strides, multi_channel
    )

    # Normalize shapes.
    data = data.reshape((B, c_i) + m)
    filt = filt.reshape((c_o, c_i) + n)
    output = np.zeros((B, c_o) + p, dtype=data.dtype)
    slc = tuple(slice(None, None, s_d) for s_d in s)

    for k in range(B):
        for j in range(c_o):
            for i in range(c_i):
                output[k, j] += signal.convolve(
                    data[k, i], filt[j, i], mode=mode
                )[slc]

    # Reshape.
    if multi_channel:
        output = output.reshape(b + (c_o,) + p)
    else:
        output = output.reshape(b + p)

    return output


def _convolve_data_adjoint(
    output, filt, data_shape, mode="full", strides=None, multi_channel=False
):
    D, b, B, m, n, s, c_i, c_o, p = _get_convolve_params(
        data_shape, filt.shape, mode, strides, multi_channel
    )

    # Normalize shapes.
    output = output.reshape((B, c_o) + p)
    filt = filt.reshape((c_o, c_i) + n)
    data = np.zeros((B, c_i) + m, dtype=output.dtype)
    slc = tuple(slice(None, None, s_d) for s_d in s)
    if mode == "full":
        output_kj = np.zeros(
            [m_d + n_d - 1 for m_d, n_d in zip(m, n)], dtype=output.dtype
        )
        adjoint_mode = "valid"
    elif mode == "valid":
        output_kj = np.zeros(
            [max(m_d, n_d) - min(m_d, n_d) + 1 for m_d, n_d in zip(m, n)],
            dtype=output.dtype,
        )
        if all(m_d >= n_d for m_d, n_d in zip(m, n)):
            adjoint_mode = "full"
        else:
            adjoint_mode = "valid"

    for k in range(B):
        for j in range(c_o):
            for i in range(c_i):
                output_kj[slc] = output[k, j]
                data[k, i] += signal.correlate(
                    output_kj, filt[j, i], mode=adjoint_mode
                )

    # Reshape.
    data = data.reshape(data_shape)
    return data


def _convolve_filter_adjoint(
    output, data, filt_shape, mode="full", strides=None, multi_channel=False
):
    D, b, B, m, n, s, c_i, c_o, p = _get_convolve_params(
        data.shape, filt_shape, mode, strides, multi_channel
    )

    # Normalize shapes.
    data = data.reshape((B, c_i) + m)
    output = output.reshape((B, c_o) + p)
    slc = tuple(slice(None, None, s_d) for s_d in s)
    if mode == "full":
        output_kj = np.zeros(
            [m_d + n_d - 1 for m_d, n_d in zip(m, n)], dtype=output.dtype
        )
        adjoint_mode = "valid"
    elif mode == "valid":
        output_kj = np.zeros(
            [max(m_d, n_d) - min(m_d, n_d) + 1 for m_d, n_d in zip(m, n)],
            dtype=output.dtype,
        )
        if all(m_d >= n_d for m_d, n_d in zip(m, n)):
            adjoint_mode = "valid"
        else:
            adjoint_mode = "full"

    filt = np.zeros((c_o, c_i) + n, dtype=output.dtype)
    for k in range(B):
        for j in range(c_o):
            for i in range(c_i):
                output_kj[slc] = output[k, j]
                filt[j, i] += signal.correlate(
                    output_kj, data[k, i], mode=adjoint_mode
                )

    # Reshape.
    filt = filt.reshape(filt_shape)
    return filt


def _complex(func, data1, data2, *kargs, **kwargs):
    """Helper function to convert func to support complex floats."""
    xp = backend.get_array_module(data1)
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


if config.cupycudnn_enabled:  # pragma: no cover
    from cupy import cudnn

    def _convolve_cuda(
        data, filt, mode="full", strides=None, multi_channel=False
    ):
        xp = backend.get_array_module(data)

        D, b, B, m, n, s, c_i, c_o, p = _get_convolve_params(
            data.shape, filt.shape, mode, strides, multi_channel
        )

        if D == 1:
            return _convolve_cuda(
                xp.expand_dims(data, -1),
                xp.expand_dims(filt, -1),
                mode=mode,
                strides=list(strides) + [1] if strides is not None else None,
                multi_channel=multi_channel,
            ).squeeze(-1)
        elif D > 3:
            raise ValueError(
                f"cuDNN convolution only supports 1, 2, or 3D, got {D}."
            )

        dilations = (1,) * D
        groups = 1
        auto_tune = True
        tensor_core = "auto"
        if mode == "full":
            pads = tuple(n_d - 1 for n_d in n)
        elif mode == "valid":
            pads = (0,) * D

        data = data.reshape((B, c_i) + m)
        filt = filt.reshape((c_o, c_i) + n)
        output = xp.empty((B, c_o) + p, dtype=data.dtype)
        filt = util.flip(filt, axes=range(-D, 0))
        cudnn.convolution_forward(
            data,
            filt,
            None,
            output,
            pads,
            s,
            dilations,
            groups,
            auto_tune=auto_tune,
            tensor_core=tensor_core,
        )

        # Reshape.
        if multi_channel:
            output = output.reshape(b + (c_o,) + p)
        else:
            output = output.reshape(b + p)

        return output

    def _convolve_data_adjoint_cuda(
        output,
        filt,
        data_shape,
        mode="full",
        strides=None,
        multi_channel=False,
    ):
        xp = backend.get_array_module(output)

        D, b, B, m, n, s, c_i, c_o, p = _get_convolve_params(
            data_shape, filt.shape, mode, strides, multi_channel
        )

        if D == 1:
            return _convolve_data_adjoint_cuda(
                xp.expand_dims(output, -1),
                xp.expand_dims(filt, -1),
                list(data_shape) + [1],
                mode=mode,
                strides=list(strides) + [1] if strides is not None else None,
                multi_channel=multi_channel,
            ).squeeze(-1)
        elif D > 3:
            raise ValueError(
                f"cuDNN convolution only supports 1, 2 or 3D, got {D}."
            )

        dilations = (1,) * D
        groups = 1
        auto_tune = True
        tensor_core = "auto"
        deterministic = False
        if mode == "full":
            pads = tuple(n_d - 1 for n_d in n)
        elif mode == "valid":
            pads = (0,) * D

        output = output.reshape((B, c_o) + p)
        filt = filt.reshape((c_o, c_i) + n)
        data = xp.empty((B, c_i) + m, dtype=output.dtype)
        filt = util.flip(filt, axes=range(-D, 0))
        cudnn.convolution_backward_data(
            filt,
            output,
            None,
            data,
            pads,
            s,
            dilations,
            groups,
            deterministic=deterministic,
            auto_tune=auto_tune,
            tensor_core=tensor_core,
        )

        # Reshape.
        data = data.reshape(data_shape)

        return data

    def _convolve_filter_adjoint_cuda(
        output,
        data,
        filt_shape,
        mode="full",
        strides=None,
        multi_channel=False,
    ):
        xp = backend.get_array_module(data)

        D, b, B, m, n, s, c_i, c_o, p = _get_convolve_params(
            data.shape, filt_shape, mode, strides, multi_channel
        )
        if D == 1:
            return _convolve_filter_adjoint_cuda(
                xp.expand_dims(output, -1),
                xp.expand_dims(data, -1),
                list(filt_shape) + [1],
                mode=mode,
                strides=list(strides) + [1] if strides is not None else None,
                multi_channel=multi_channel,
            ).squeeze(-1)
        elif D > 3:
            raise ValueError(
                f"cuDNN convolution only supports 1, 2 or 3D, got {D}."
            )

        dilations = (1,) * D
        groups = 1
        auto_tune = True
        tensor_core = "auto"
        deterministic = False
        if mode == "full":
            pads = tuple(n_d - 1 for n_d in n)
        elif mode == "valid":
            pads = (0,) * D

        data = data.reshape((B, c_i) + m)
        output = output.reshape((B, c_o) + p)
        filt = xp.empty((c_o, c_i) + n, dtype=output.dtype)
        cudnn.convolution_backward_filter(
            data,
            output,
            filt,
            pads,
            s,
            dilations,
            groups,
            deterministic=deterministic,
            auto_tune=auto_tune,
            tensor_core=tensor_core,
        )
        filt = util.flip(filt, axes=range(-D, 0))
        filt = filt.reshape(filt_shape)

        return filt


if config.cudnn_frontend_enabled:  # pragma: no cover
    import cudnn as _cudnn

    _cudnn_handle = None
    _cudnn_plans = {}

    _cudnn_dtypes = {
        np.dtype(np.float32): _cudnn.data_type.FLOAT,
        np.dtype(np.float64): _cudnn.data_type.DOUBLE,
    }

    def _cudnn_fe_supported(filt_shape, multi_channel):
        """cuDNN convolution only covers D=1, 2 or 3."""
        return len(filt_shape) - 2 * multi_channel <= 3

    def _get_cudnn_handle():
        global _cudnn_handle
        if _cudnn_handle is None:
            _cudnn_handle = _cudnn.create_handle()

        return _cudnn_handle

    # cuDNN reads operands with vectorized loads and rejects pointers that
    # are not 16 byte aligned.
    _cudnn_alignment = 16

    def _cudnn_operand(a):
        """Return a densely packed, aligned version of a.

        The contiguity flag cannot be trusted here: every axis of
        ``xp.imag(x)`` has size one for the scalar-ish shapes convolve is
        called with in tests, which marks the view contiguous even though it
        is strided over the complex array and starts one float in. That
        offset is what cuDNN rejects, and ascontiguousarray will not fix it
        because it believes the array is already contiguous.

        """
        xp = backend.get_array_module(a)
        packed = a.strides == tuple(
            i * a.itemsize for i in _contiguous_stride(a.shape)
        )
        if packed and a.data.ptr % _cudnn_alignment == 0:
            return a

        out = xp.empty(a.shape, dtype=a.dtype)
        out[...] = a
        return out

    def _contiguous_stride(shape):
        """Element strides of a C-contiguous array of the given shape."""
        stride = []
        acc = 1
        for d in reversed(shape):
            stride.append(acc)
            acc *= d

        return list(reversed(stride))

    def _build_cudnn_graph(
        kind,
        a_shape,
        b_shape,
        c_shape,
        pads,
        strides,
        dilations,
        dtype,
        tensor_core,
    ):
        data_type = _cudnn_dtypes[np.dtype(dtype)]
        # Double precision convolution has engines only when the compute
        # type is double as well.
        compute_type = (
            _cudnn.data_type.DOUBLE
            if data_type == _cudnn.data_type.DOUBLE
            else _cudnn.data_type.FLOAT
        )
        graph = _cudnn.pygraph(
            handle=_get_cudnn_handle(),
            intermediate_data_type=compute_type,
            compute_data_type=compute_type,
        )
        a = graph.tensor(
            dim=list(a_shape),
            stride=_contiguous_stride(a_shape),
            data_type=data_type,
        )
        b = graph.tensor(
            dim=list(b_shape),
            stride=_contiguous_stride(b_shape),
            data_type=data_type,
        )
        conv = {
            "fprop": graph.conv_fprop,
            "dgrad": graph.conv_dgrad,
            "wgrad": graph.conv_wgrad,
        }[kind]
        c = conv(
            a,
            b,
            padding=list(pads),
            stride=list(strides),
            dilation=list(dilations),
        )
        c.set_output(True).set_data_type(data_type)
        c.set_dim(list(c_shape)).set_stride(_contiguous_stride(c_shape))

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans(
            [_cudnn.heur_mode.A, _cudnn.heur_mode.FALLBACK]
        )
        if not tensor_core:
            # Tensor core engines run 3D single precision convolution at
            # TF32, which costs about 3e-4 relative error where the rest of
            # the pipeline holds 1e-7. Deselecting them measured free.
            graph.deselect_numeric_notes([_cudnn.numerical_note.TENSOR_CORE])

        graph.check_support()
        graph.build_plans()
        return graph, a, b, c

    def _cudnn_execute(kind, a, b, c_shape, pads, strides, dilations):
        """Run one cuDNN convolution, reusing the plan across calls.

        Building a graph costs far more than executing one, and sigpy
        convolves the same shapes repeatedly inside iterative algorithms, so
        plans are cached on everything that defines them.

        """
        xp = backend.get_array_module(a)
        a = _cudnn_operand(a)
        b = _cudnn_operand(b)
        key = (
            kind,
            a.shape,
            b.shape,
            tuple(c_shape),
            tuple(pads),
            tuple(strides),
            tuple(dilations),
            a.dtype.str,
        )
        if key not in _cudnn_plans:
            try:
                _cudnn_plans[key] = _build_cudnn_graph(
                    kind,
                    a.shape,
                    b.shape,
                    c_shape,
                    pads,
                    strides,
                    dilations,
                    a.dtype,
                    tensor_core=False,
                )
            except Exception:
                _cudnn_plans[key] = _build_cudnn_graph(
                    kind,
                    a.shape,
                    b.shape,
                    c_shape,
                    pads,
                    strides,
                    dilations,
                    a.dtype,
                    tensor_core=True,
                )

        graph, a_t, b_t, c_t = _cudnn_plans[key]
        c = xp.empty(tuple(c_shape), dtype=a.dtype)
        workspace = xp.empty(
            max(graph.get_workspace_size(), 1), dtype=xp.uint8
        )
        _cudnn.set_stream(
            handle=_get_cudnn_handle(),
            stream=xp.cuda.get_current_stream().ptr,
        )
        graph.execute(
            {a_t: a.data.ptr, b_t: b.data.ptr, c_t: c.data.ptr},
            workspace.data.ptr,
            handle=_get_cudnn_handle(),
        )
        return c

    def _cudnn_pads(mode, n):
        return tuple(n_d - 1 for n_d in n) if mode == "full" else (0,) * len(n)

    def _convolve_cudnn_fe(
        data, filt, mode="full", strides=None, multi_channel=False
    ):
        xp = backend.get_array_module(data)
        D, b, B, m, n, s, c_i, c_o, p = _get_convolve_params(
            data.shape, filt.shape, mode, strides, multi_channel
        )
        if D == 1:
            return _convolve_cudnn_fe(
                xp.expand_dims(data, -1),
                xp.expand_dims(filt, -1),
                mode=mode,
                strides=list(strides) + [1] if strides is not None else None,
                multi_channel=multi_channel,
            ).squeeze(-1)

        data = data.reshape((B, c_i) + m)
        filt = util.flip(filt.reshape((c_o, c_i) + n), axes=range(-D, 0))
        output = _cudnn_execute(
            "fprop",
            data,
            filt,
            (B, c_o) + p,
            _cudnn_pads(mode, n),
            s,
            (1,) * D,
        )

        # Reshape.
        if multi_channel:
            output = output.reshape(b + (c_o,) + p)
        else:
            output = output.reshape(b + p)

        return output

    def _convolve_data_adjoint_cudnn_fe(
        output,
        filt,
        data_shape,
        mode="full",
        strides=None,
        multi_channel=False,
    ):
        xp = backend.get_array_module(output)
        D, b, B, m, n, s, c_i, c_o, p = _get_convolve_params(
            data_shape, filt.shape, mode, strides, multi_channel
        )
        if D == 1:
            return _convolve_data_adjoint_cudnn_fe(
                xp.expand_dims(output, -1),
                xp.expand_dims(filt, -1),
                list(data_shape) + [1],
                mode=mode,
                strides=list(strides) + [1] if strides is not None else None,
                multi_channel=multi_channel,
            ).squeeze(-1)

        output = output.reshape((B, c_o) + p)
        filt = util.flip(filt.reshape((c_o, c_i) + n), axes=range(-D, 0))
        data = _cudnn_execute(
            "dgrad",
            output,
            filt,
            (B, c_i) + m,
            _cudnn_pads(mode, n),
            s,
            (1,) * D,
        )

        # Reshape.
        return data.reshape(data_shape)

    def _convolve_filter_adjoint_cudnn_fe(
        output,
        data,
        filt_shape,
        mode="full",
        strides=None,
        multi_channel=False,
    ):
        xp = backend.get_array_module(data)
        D, b, B, m, n, s, c_i, c_o, p = _get_convolve_params(
            data.shape, filt_shape, mode, strides, multi_channel
        )
        if D == 1:
            return _convolve_filter_adjoint_cudnn_fe(
                xp.expand_dims(output, -1),
                xp.expand_dims(data, -1),
                list(filt_shape) + [1],
                mode=mode,
                strides=list(strides) + [1] if strides is not None else None,
                multi_channel=multi_channel,
            ).squeeze(-1)

        data = data.reshape((B, c_i) + m)
        output = output.reshape((B, c_o) + p)
        filt = _cudnn_execute(
            "wgrad",
            data,
            output,
            (c_o, c_i) + n,
            _cudnn_pads(mode, n),
            s,
            (1,) * D,
        )
        filt = util.flip(filt, axes=range(-D, 0))

        # Reshape.
        return filt.reshape(filt_shape)
