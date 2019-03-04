# -*- coding: utf-8 -*-
"""This module provides functions for interoperability
between sigpy and pytorch.
"""
import numpy as np
from sigpy import backend, config

__all__ = []
if config.pytorch_enabled:
    __all__ += ['to_pytorch', 'from_pytorch']

    def to_pytorch(array):
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
            return torch.from_numpy(array)
        else:
            return from_dlpack(array.toDlpack())

    def from_pytorch(tensor, iscomplex=False):
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
            output = tensor.numpy()
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

        return output
