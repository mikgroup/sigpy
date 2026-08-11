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
    try:
        cupycudnn_enabled = util.find_spec("cupy.cuda.cudnn") is not None
        if cupycudnn_enabled:
            from cupy import cudnn  # noqa: F401
    except ImportError as e:
        warnings.warn(
            f"Importing cupy.cuda.cudnn failed. "
            f"For more details, see the error stack below:\n{e}"
        )
        cupycudnn_enabled = False
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
    cudnn_frontend_enabled = False
    if not cupycudnn_enabled:
        # CuPy 14 dropped cuDNN; try using cudnn-frontend instead. 
        if util.find_spec("cudnn") is not None:
            try:
                import cudnn as _cudnn_frontend

                cudnn_frontend_enabled = _cudnn_frontend.backend_version() > 0
            except Exception as e:
                warnings.warn(
                    f"cudnn-frontend is installed but no cuDNN backend "
                    f"CuPy array convolutions will use the FFT implementation, not cuDNN."
                    f"For more details, see the error stack below:\n{e}"
                )
                cudnn_frontend_enabled = False
else:
    cupycudnn_enabled = False
    cudnn_frontend_enabled = False
    nccl_enabled = False

mpi4py_enabled = util.find_spec("mpi4py") is not None

# This is to catch an import error when the cudnn in cupy (system) and pytorch
# (built in) are in conflict.
if util.find_spec("torch") is not None:
    try:
        import torch  # noqa

        pytorch_enabled = True
    except ImportError:
        print("Warning : Pytorch installed but can import")
        pytorch_enabled = False
else:
    pytorch_enabled = False
