# -*- coding: utf-8 -*-
"""MRI RF shimming.
"""

import sigpy as sp
import numpy as np
from sigpy import backend
from sigpy.mri import rf as rf


__all__ = ['calc_shims', 'init_optimal_spectral', 'init_circ_polar']


def calc_shims(shim_roi, sens, x0, dt, lamb=0, max_iter=50):
    """RF shim designer. Uses the Gerchberg Saxton algorithm.

     Args:
        shim_roi (array): region within volume to be shimmed. Mask of 1's and
            0's. [dim_x dim_y dim_z]
        sens (array): sensitivity maps. [Nc dim_x dim_y dim_z]
        x0 (array) initial guess for shim values. [Nc 1]
        dt (float): hardware sampling dwell time.
        lamb (float): regularization term.
        max_iter (int): max number of iterations.

     Returns:
         Vector of complex shim weights.
    """

    k1 = np.expand_dims(np.array((0, 0, 0)), 0)
    A = rf.PtxSpatialExplicit(sens, coord=k1, dt=dt,
                              img_shape=shim_roi.shape, ret_array=False)

    alg_method = sp.alg.GerchbergSaxton(A, shim_roi, x0, max_iter=max_iter,
                                        tol=10E-9, lamb=lamb)
    while not alg_method.done():
        alg_method.update()

    return alg_method.x


def init_optimal_spectral(A, sens, preproc=False):
    """Function to return initial shim weights based on an optimal spectral
    method, an eigenvector-based method.

        Args:
            A (linop): sigpy Linear operator.
            sens (array): sensitivity maps. [Nc dim_x dim_y]
            preproc (bool): option to apply preprocessing function before \
                finding eigenvectors

        Returns:
            Vector of complex shim weights.

        References:
            Chandra, R., Zhong, Z., Hontz, J., McCulloch, V., Studer, C.,
            Goldstein, T. (2017) 'PhasePack: A Phase Retrieval Library.'
            arXiv:1711.10175.
    """
    device = backend.get_device(sens)
    xp = device.xp
    with device:
        if hasattr(A, 'repr_str') and A.repr_str == 'pTx spatial explicit':
            Anum = A.linops[1].mat
        else:
            Anum = A

        sens = sens.flatten()
        n = Anum.shape[1]
        Anumt = xp.transpose(Anum)

        m = sens.size
        y = sens ** 2

        # normalize the measurements
        delta = m / n
        ymean = y / xp.mean(y)

        # apply pre-processing function
        yplus = xp.amax(y)
        Y = (1 / m) * Anumt @ Anum

        if preproc:
            T = (yplus - 1) / (yplus + xp.sqrt(delta) - 1)

            # unnormalize
            T *= ymean
            T = xp.transpose(xp.expand_dims(T, axis=1))

            for mm in range(m):
                col = Anum[mm, :]
                aat = col * xp.transpose(col)
                Y = Y + (1 / m) * T[mm] * aat

        w, v = xp.linalg.eigh(Y)

        return xp.expand_dims(v[:, 0], 1)


def init_circ_polar(sens):
    """Function to return circularly polarized initial shim weights. Provides
     shim weights that set the phase to be even in the middle of the sens
     profiles.

        Args:
            sens (array): sensitivity maps. [Nc dim_x dim_y]

        Returns:
            Vector of complex shim weights.
    """
    dim = sens.shape[1]
    device = backend.get_device(sens)
    xp = device.xp
    with device:
        # As a rough approximation, assume that the center of sens profile is
        # also the center of the object within the profile to be imaged.
        phs = xp.angle(sens[:, xp.int(dim / 2), xp.int(dim / 2)])
        phs_wt = xp.exp(-phs * 1j)

    return xp.expand_dims(phs_wt, 1)
