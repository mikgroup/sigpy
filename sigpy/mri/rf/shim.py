# -*- coding: utf-8 -*-
"""MRI RF shimming.
"""

import sigpy as sp
import numpy as np
from sigpy.mri import rf as rf
from sigpy import backend

__all__ = ['calc_shims']


def calc_shims(shim_roi, sens, dt, lamb=0, max_iter=50, minibatch=False, batchsize=1):
    """RF shim designer.

     Args:
        shim_roi (array): region within volume to be shimmed. Mask of 1's and
            0's. [dim_x dim_y dim_z]
        sens (array): sensitivity maps. [Nc dim_x dim_y dim_z]
        dt (float): hardware sampling dwell time.
        lamb (float): regularization term.
        max_iter (int): max number of iterations.

    Returns:
        array: complex shims.
    """

    k1 = np.expand_dims(np.array((0, 0, 0)), 0)
    A = sp.mri.rf.PtxSpatialExplicit(sens, coord=k1, dt=dt,
                                     img_shape=shim_roi.shape, ret_array=False)


    alg_method = sp.alg.GerchbergSaxton(A, shim_roi,
                                        max_iter=max_iter, tol=10E-9, lamb=lamb,
                                        minibatch=minibatch,minisize=batchsize)
    while (not alg_method.done()):
        alg_method.update()

    return alg_method.x
