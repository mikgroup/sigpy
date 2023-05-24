# -*- coding: utf-8 -*-
"""Configuration.

This module contains flags to turn on and off optional modules.

"""
import warnings
from importlib import util
import torch

cuda_avail = torch.cuda.is_available()
cupy_enabled = (util.find_spec("cupy") is not None) and cuda_avail
if cupy_enabled:
    try:
        import cupy  # noqa
    except ImportError as e:
        warnings.warn(
            f"Importing cupy failed. "
            f"For more details, see the error stack below:\n{e}"
        )
        cupy_enabled = False

if cupy_enabled:  # pragma: no cover
    try:
        cudnn_enabled = util.find_spec("cupy.cuda.cudnn") is not None
        if cudnn_enabled:
            from cupy import cudnn  # noqa: F401
    except ImportError as e:
        warnings.warn(
            f"Importing cupy.cuda.cudnn failed. "
            f"For more details, see the error stack below:\n{e}"
        )
        cudnn_enabled = False
    try:
        nccl_enabled = util.find_spec("cupy.cuda.nccl") is not None
        if nccl_enabled:
            from cupy.cuda import nccl  # noqa: F401
    except ImportError as e:
        warnings.warn(
            f"Importing cupy.cuda.nccl failed. "
            f"For more details, see the error stack below:\n{e}"
        )
        nccl_enabled = False
else:
    cudnn_enabled = False
    nccl_enabled = False

mpi4py_enabled = util.find_spec("mpi4py") is not None

# This is to catch an import error when the cudnn in cupy (system) and pytorch
# (built in) are in conflict.
if util.find_spec("torch") is not None:
    try:
        import torch  # noqa
        pytorch_enabled = True
    except ImportError:
        print('Warning : Pytorch installed but can import')
        pytorch_enabled = False
else:
    pytorch_enabled = False
