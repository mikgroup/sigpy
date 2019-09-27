# -*- coding: utf-8 -*-
"""Functions for interoperability between sigpy and pytorch.

"""
import numpy as np
from sigpy import backend, config

__all__ = ['to_pytorch', 'from_pytorch', 'to_pytorch_function']


def to_pytorch(array, requires_grad=True):  # pragma: no cover
    """Zero-copy conversion from numpy/cupy array to pytorch tensor.

    For complex array input, returns a tensor with shape + [2],
    where tensor[..., 0] and tensor[..., 1] represent the real
    and imaginary.

    Args:
        array (numpy/cupy array): input.

    Returns:
        PyTorch tensor.

    """
    import torch
    from torch.utils.dlpack import from_dlpack

    device = backend.get_device(array)
    if not np.issubdtype(array.dtype, np.floating):
        with device:
            shape = array.shape
            array = array.view(dtype=array.real.dtype)
            array = array.reshape(shape + (2, ))

    if device == backend.cpu_device:
        tensor = torch.from_numpy(array)
    else:
        tensor = from_dlpack(array.toDlpack())

    tensor.requires_grad = requires_grad
    return tensor


def from_pytorch(tensor, iscomplex=False):  # pragma: no cover
    """Zero-copy conversion from pytorch tensor to numpy/cupy array.

    If iscomplex, then tensor must have the last dimension as 2,
    and the output will be viewed as a complex valued array.

    Args:
        tensor (PyTorch tensor): input.
        iscomplex (bool): whether input represents complex valued tensor.

    Returns:
        Numpy/cupy array.

    """
    from torch.utils.dlpack import to_dlpack

    device = tensor.device
    if device.type == 'cpu':
        output = tensor.detach().numpy()
    else:
        if config.cupy_enabled:
            import cupy as cp
            output = cp.fromDlpack(to_dlpack(tensor))
        else:
            raise TypeError('CuPy not installed, '
                            'but trying to convert GPU PyTorch Tensor.')

    if iscomplex:
        if output.shape[-1] != 2:
            raise ValueError('shape[-1] must be 2 when iscomplex is '
                             'specified, but got {}'.format(output.shape))

        with backend.get_device(output):
            if output.dtype == np.float32:
                output = output.view(np.complex64)
            elif output.dtype == np.float64:
                output = output.view(np.complex128)

            output = output.reshape(output.shape[:-1])

    return output


def to_pytorch_function(linop,
                        input_iscomplex=False,
                        output_iscomplex=False):  # pragma: no cover
    """Convert SigPy Linop to PyTorch Function.

    The returned function can be treated as a native
    pytorch function performing the linop operator.
    The function can be backpropagated, applied on GPU arrays,
    and has minimal overhead as the underlying arrays
    are shared without copying.
    For complex valued input/output, the appropriate options
    should be set when calling the function.

    Args:
        linop (Linop): linear operator to be converted.
        input_iscomplex (bool): whether the PyTorch input
            represents complex tensor.
        output_iscomplex (bool): whether the PyTorch output
            represents complex tensor.

    Returns:
        torch.autograd.Function: equivalent PyTorch Function.

    """
    import torch

    class LinopFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            return to_pytorch(linop(from_pytorch(
                input, iscomplex=input_iscomplex)))

        @staticmethod
        def backward(ctx, grad_output):
            return to_pytorch(linop.H(from_pytorch(
                grad_output, iscomplex=output_iscomplex)))

    return LinopFunction
