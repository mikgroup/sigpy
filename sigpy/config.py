# -*- coding: utf-8 -*-
"""Configuration.

This module contains flags to turn on and off optional modules.

"""
import warnings
from importlib import util

cupy_enabled = util.find_spec("cupy") is not None
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
    cudnn_enabled = util.find_spec("cupy.cuda.cudnn") is not None
    nccl_enabled = util.find_spec("cupy.cuda.nccl") is not None
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
