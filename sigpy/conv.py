import numpy as np
from sigpy import fft, util, config


__all__ = ['convolve', 'correlate',
           'cudnn_convolve', 'cudnn_convolve_backward_filter',
           'cudnn_convolve_backward_data']


def convolve(input1, input2, axes=None, mode='full'):
    """Multi-dimensional convolution.

    Args:
        input1 (array): Input 1.
        input2 (array): Input 2.
        axes (None or array of ints): Axes to perform convolution.
        mode (str): {'full', 'valid'}

    Returns:
        array: Convolved result of shape input1.shape + input2.shape - 1
            if mode='full', and input1.shape - input2.shape + 1 otherwise.

    """

    util._check_same_dtype(input1, input2)
    axes = util._normalize_axes(axes, max(input1.ndim, input2.ndim))
    tshape1, tshape2, toshape, oshape, shift, scale = _get_convolve_shapes(input1.shape,
                                                                           input2.shape,
                                                                           axes, mode)
    dtype = input1.dtype
    max_ndim = len(oshape)
    device = util.get_device(input1)
    xp = device.xp

    with device:
        input1_pad = util.resize(input1, tshape1, oshift=[0] * max_ndim)
        input2_pad = util.resize(input2, tshape2, oshift=[0] * max_ndim)

        if np.issubdtype(dtype, np.floating):
            input1_fft = xp.fft.rfftn(input1_pad, axes=axes, norm='ortho')
            input2_fft = xp.fft.rfftn(input2_pad, axes=axes, norm='ortho')
            output = xp.fft.irfftn(input1_fft * input2_fft, [toshape[a] for a in axes],
                                   axes=axes, norm='ortho').astype(dtype)
        else:
            input1_fft = fft.fft(input1_pad, axes=axes, center=False)
            input2_fft = fft.fft(input2_pad, axes=axes, center=False)
            output = fft.ifft(input1_fft * input2_fft, axes=axes, center=False)

        if mode == 'valid':
            output = util.resize(output, oshape, ishift=shift)
        output *= scale

        return output


def correlate(input1, input2, axes=None, mode='full'):
    """Multi-dimensional cross-correlation.

    Args:
        input1 (array): Input 1.
        input2 (array): Input 2.
        axes (None or array of ints): Axes to perform cross-correlation.
        mode (str): {'full', 'valid'}

    Returns:
        array: Correlated result of shape input1.shape + input2.shape - 1
            if mode='full', and input1.shape - input2.shape + 1 otherwise.

    """

    device = util.get_device(input1)
    xp = device.xp

    with device:
        _, ishape2_exp = util._expand_shapes(input1.shape, input2.shape)
        input2_conj = xp.conj(
            util.flip(input2.reshape(ishape2_exp), axes=axes))
        output = convolve(input1, input2_conj, axes=axes, mode=mode)
        return output


if config.cudnn_enabled:
    import cupy as cp
    from cupy import cudnn
    from cupy.cuda import cudnn as libcudnn

    def _get_cudnn_convolve_y_shape(x_shape, W_shape, mode):
        c_O, c_I = W_shape[:2]
        b = x_shape[0]

        if mode == 'full':
            return (b, c_O) + tuple(m + n - 1 for m, n in zip(x_shape[2:], W_shape[2:]))
        else:
            return (b, c_O) + tuple(m - n + 1 for m, n in zip(x_shape[2:], W_shape[2:]))

    def _get_cudnn_convolve_x_shape(W_shape, y_shape, mode):
        c_O, c_I = W_shape[:2]
        b = y_shape[0]

        if mode == 'full':
            return (b, c_I) + tuple(p - n + 1 for p, n in zip(y_shape[2:], W_shape[2:]))
        else:
            return (b, c_I) + tuple(p + n - 1 for p, n in zip(y_shape[2:], W_shape[2:]))

    def _get_cudnn_convolve_W_shape(x_shape, y_shape, mode):
        c_I = x_shape[1]
        c_O = y_shape[1]
        b = y_shape[0]

        if mode == 'full':
            return (c_O, c_I) + tuple(p - m + 1 for p, m in zip(y_shape[2:], x_shape[2:]))
        else:
            return (c_O, c_I) + tuple(m - p + 1 for p, m in zip(y_shape[2:], x_shape[2:]))

    def cudnn_convolve(
            x, W, mode='full', strides=None,
            workspace_size=1024 * 1024 * 1024,
            conv_mode=libcudnn.CUDNN_CONVOLUTION,
            pref=libcudnn.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
    ):
        """
        x - (b, c_I, m_1, m_2, ..., m_N)
        W - (c_O, c_I, n_1, n_2, ..., n_N)
        y - (b, c_O, p_1, p_2, ..., p_N)
        """
        assert x.shape[1] == W.shape[1]
        assert x.ndim == W.ndim
        util._check_same_dtype(W, x)

        ndim = W.ndim - 2
        dtype = x.dtype
        device = util.get_device(x)
        xp = device.xp

        if np.issubdtype(dtype, np.complexfloating):
            with device:
                xr = xp.real(x)
                xi = xp.imag(x)
                Wr = xp.real(W)
                Wi = xp.imag(W)

                yr = cudnn_convolve(xr, Wr, mode=mode,
                                    strides=strides, conv_mode=conv_mode)
                yr -= cudnn_convolve(xi, Wi, mode=mode,
                                     strides=strides, conv_mode=conv_mode)
                yi = cudnn_convolve(xr, Wi, mode=mode,
                                    strides=strides, conv_mode=conv_mode)
                yi += cudnn_convolve(xi, Wr, mode=mode,
                                     strides=strides, conv_mode=conv_mode)

                return (yr + 1j * yi).astype(dtype)

        if strides is None:
            strides = (1, ) * ndim

        if mode == 'full':
            pads = tuple(n - 1 for n in W.shape[2:])
        else:
            pads = (0, ) * ndim

        y_shape = _get_cudnn_convolve_y_shape(x.shape, W.shape, mode)

        with device:
            x = cp.ascontiguousarray(x)
            W = cp.ascontiguousarray(W)
            y = util.empty(y_shape, dtype=dtype, device=device)

            handle = cudnn.get_handle()
            x_desc = cudnn.create_tensor_descriptor(x)
            y_desc = cudnn.create_tensor_descriptor(y)
            W_desc = cudnn.create_filter_descriptor(W)

            conv_desc = cudnn.create_convolution_descriptor(
                pads, strides, dtype, mode=conv_mode)

            workspace = util.empty(workspace_size, dtype='b', device=device)
            algo = libcudnn.getConvolutionForwardAlgorithm(
                handle, x_desc.value, W_desc.value,
                conv_desc.value, y_desc.value, pref,
                workspace_size)

            oz_dtype = 'd' if x.dtype == 'd' else 'f'
            one = np.array(1, dtype=oz_dtype).ctypes
            zero = np.array(0, dtype=oz_dtype).ctypes
            libcudnn.convolutionForward(
                handle, one.data, x_desc.value, x.data.ptr,
                W_desc.value, W.data.ptr, conv_desc.value,
                algo, workspace.data.ptr, workspace_size, zero.data,
                y_desc.value, y.data.ptr)

            return y

    def cudnn_convolve_backward_filter(
            x, y, mode='full', strides=None,
            workspace_size=1024 * 1024 * 1024,
            conv_mode=libcudnn.CUDNN_CONVOLUTION,
            pref=libcudnn.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT
    ):
        """
        x - (b, c_I, m_1, m_2, ..., m_N)
        y - (b, c_O, p_1, p_2, ..., p_N)
        W - (c_O, c_I, n_1, n_2, ..., n_N)
        """
        assert x.shape[0] == y.shape[0]
        assert x.ndim == y.ndim
        util._check_same_dtype(x, y)

        ndim = x.ndim - 2
        dtype = x.dtype
        device = util.get_device(x)
        xp = device.xp

        if np.issubdtype(dtype, np.complexfloating):
            with device:
                xr = xp.real(x)
                xi = -xp.imag(x)
                yr = xp.real(y)
                yi = xp.imag(y)

                Wr = cudnn_convolve_backward_filter(xr, yr, mode=mode, strides=strides,
                                                    conv_mode=conv_mode, pref=pref)
                Wr -= cudnn_convolve_backward_filter(xi, yi, mode=mode, strides=strides,
                                                     conv_mode=conv_mode, pref=pref)

                Wi = cudnn_convolve_backward_filter(xr, yi, mode=mode, strides=strides,
                                                    conv_mode=conv_mode, pref=pref)

                Wi += cudnn_convolve_backward_filter(xi, yr, mode=mode, strides=strides,
                                                     conv_mode=conv_mode, pref=pref)

                return (Wr + 1j * Wi).astype(dtype)

        W_shape = _get_cudnn_convolve_W_shape(x.shape, y.shape, mode)
        if strides is None:
            strides = (1, ) * ndim

        if mode == 'full':
            pads = tuple(n - 1 for n in W_shape[2:])
        else:
            pads = (0, ) * ndim

        with device:
            x = cp.ascontiguousarray(x)
            W = util.empty(W_shape, dtype=dtype, device=device)
            y = cp.ascontiguousarray(y)

            handle = cudnn.get_handle()
            x_desc = cudnn.create_tensor_descriptor(x)
            y_desc = cudnn.create_tensor_descriptor(y)
            W_desc = cudnn.create_filter_descriptor(W)

            conv_desc = cudnn.create_convolution_descriptor(
                pads, strides, dtype, mode=conv_mode)

            workspace = util.empty(workspace_size, dtype='b', device=device)
            algo = libcudnn.getConvolutionBackwardFilterAlgorithm(
                handle, x_desc.value, y_desc.value,
                conv_desc.value, W_desc.value, pref,
                workspace_size)

            oz_dtype = 'd' if x.dtype == 'd' else 'f'
            one = np.array(1, dtype=oz_dtype).ctypes
            zero = np.array(0, dtype=oz_dtype).ctypes
            libcudnn.convolutionBackwardFilter_v3(
                handle, one.data, x_desc.value, x.data.ptr,
                y_desc.value, y.data.ptr, conv_desc.value,
                algo, workspace.data.ptr, workspace_size, zero.data,
                W_desc.value, W.data.ptr)

            return W

    def cudnn_convolve_backward_data(W, y,
                                     mode='full', strides=None,
                                     workspace_size=1024 * 1024 * 1024,
                                     conv_mode=libcudnn.CUDNN_CONVOLUTION,
                                     pref=libcudnn.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT):
        """
        x - (b, c_I, m_1, m_2, ..., m_N)
        W - (c_O, c_I, n_1, n_2, ..., n_N)
        y - (b, c_O, p_1, p_2, ..., p_N)
        """
        assert W.shape[0] == y.shape[1]
        assert y.ndim == W.ndim
        util._check_same_dtype(W, y)

        ndim = W.ndim - 2
        dtype = W.dtype
        device = util.get_device(W)
        xp = device.xp

        if np.issubdtype(dtype, np.complexfloating):
            with device:
                Wr = xp.real(W)
                Wi = -xp.imag(W)
                yr = xp.real(y)
                yi = xp.imag(y)

                xr = cudnn_convolve_backward_data(Wr, yr, mode=mode, strides=strides,
                                                  conv_mode=conv_mode, pref=pref)
                xr -= cudnn_convolve_backward_data(Wi, yi, mode=mode, strides=strides,
                                                   conv_mode=conv_mode, pref=pref)

                xi = cudnn_convolve_backward_data(Wr, yi, mode=mode, strides=strides,
                                                  conv_mode=conv_mode, pref=pref)

                xi += cudnn_convolve_backward_data(Wi, yr, mode=mode, strides=strides,
                                                   conv_mode=conv_mode, pref=pref)

                return (xr + 1j * xi).astype(dtype)

        x_shape = _get_cudnn_convolve_x_shape(W.shape, y.shape, mode)
        if strides is None:
            strides = (1, ) * ndim
        if mode == 'full':
            pads = tuple(n - 1 for n in W.shape[2:])
        else:
            pads = (0, ) * ndim

        with device:
            x = util.empty(x_shape, dtype=dtype, device=device)
            W = cp.ascontiguousarray(W)
            y = cp.ascontiguousarray(y)

            handle = cudnn.get_handle()
            x_desc = cudnn.create_tensor_descriptor(x)
            y_desc = cudnn.create_tensor_descriptor(y)
            W_desc = cudnn.create_filter_descriptor(W)

            conv_desc = cudnn.create_convolution_descriptor(
                pads, strides, dtype, mode=conv_mode)

            workspace = util.empty(workspace_size, dtype='b', device=device)
            algo = libcudnn.getConvolutionBackwardDataAlgorithm(
                handle, W_desc.value, y_desc.value,
                conv_desc.value, x_desc.value, pref,
                workspace_size)

            oz_dtype = 'd' if x.dtype == 'd' else 'f'
            one = np.array(1, dtype=oz_dtype).ctypes
            zero = np.array(0, dtype=oz_dtype).ctypes
            libcudnn.convolutionBackwardData_v3(
                handle, one.data, W_desc.value, W.data.ptr,
                y_desc.value, y.data.ptr, conv_desc.value,
                algo, workspace.data.ptr, workspace_size, zero.data,
                x_desc.value, x.data.ptr)

            return x


def _get_convolve_shapes(ishape1, ishape2, axes, mode):
    ishape1 = list(ishape1)
    ishape2 = list(ishape2)

    max_ndim = max(len(ishape1), len(ishape2))
    axes = util._normalize_axes(axes, max_ndim)

    tshape1 = [1] * (max_ndim - len(ishape1)) + ishape1
    tshape2 = [1] * (max_ndim - len(ishape2)) + ishape2
    toshape = [max(t1, t2) for t1, t2 in zip(tshape1, tshape2)]
    oshape = [max(t1, t2) for t1, t2 in zip(tshape1, tshape2)]
    shift = [0] * max_ndim

    scale = 1
    for a in axes:
        if mode == 'full':
            i = tshape1[a] + tshape2[a] - 1
            oshape[a] = i
        else:
            i = max(tshape1[a], tshape2[a])
            oshape[a] = max(tshape1[a], tshape2[a]) - \
                min(tshape1[a], tshape2[a]) + 1

        tshape1[a] = i
        tshape2[a] = i
        toshape[a] = i
        shift[a] = i - oshape[a]
        scale *= i**0.5

    return tshape1, tshape2, toshape, oshape, shift, scale
