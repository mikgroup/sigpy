# -*- coding: utf-8 -*-
"""Configuration.

This module contains flags to turn on and off optional modules.

"""
import importlib

cupy_enabled = importlib.util.find_spec("cupy") is not None
if cupy_enabled:
    cudnn_enabled = importlib.util.find_spec("cupy.cuda.cudnn") is not None
    nccl_enabled = importlib.util.find_spec("cupy.cuda.nccl") is not None
else:
    cudnn_anbled = False
    nccl_enabled = False

mpi4py_enabled = importlib.util.find_spec("mpi4py") is not None
