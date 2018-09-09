# -*- coding: utf-8 -*-
"""Configuration.

This module contains flags to turn on and off optional modules.

"""
try:
    import cupy
    cupy_enabled = True
except ImportError:
    cupy_enabled = False
    cudnn_enabled = False
    nccl_enabled = False

if cupy_enabled:
    try:
        from cupy import cudnn
        from cupy.cuda import cudnn
        cudnn_enabled = True
    except ImportError:
        cudnn_enabled = False

    try:
        from cupy import cudnn
        from cupy.cuda import nccl
        nccl_enabled = True
    except ImportError:
        nccl_enabled = False


try:
    import mpi4py
    mpi4py_enabled = True
except ImportError:
    mpi4py_enabled = False
