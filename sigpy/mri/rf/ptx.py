# -*- coding: utf-8 -*-
"""MRI RF excitation pulse design functions,
    including SLR and small tip spatial design
"""

import numpy as np
import sigpy as sp

from sigpy.mri import linop

__all__ = ['stspa']


def stspa(target, sens, coord=None, mask=None, pinst=float('inf'),
          pavg=float('inf'), max_iter=1000, tol=1E-6):
    """Small tip spatial domain method for multicoil parallel excitation.
       Allows for constrained or unconstrained designs.

    Args:
        target (array): desired magnetization profile.
        sens (array): sensitivity maps.
        coord (array): coordinates for noncartesian trajectories
        mask (array): kspace sampling mask for cartesian patterns only
        pinst (float): maximum instantaneous power
        pavg (float): maximum average power
        max_iter (int): max number of iterations
        tol (float): allowable error

    References:
        Grissom, W., Yip, C., Zhang, Z., Stenger, V. A., Fessler, J. A.
        & Noll, D. C.(2006).
        Spatial Domain Method for the Design of RF Pulses in Multicoil
        Parallel Excitation. Magnetic resonance in medicine, 56, 620-629.
    """

    A = linop.Sense(sens, coord, mask, ishape=target.shape).H

    if coord is not None:
        # Nc*Nt pulses
        pulses = np.zeros((sens.shape[0], coord.shape[0]), np.complex)
    else:
        pulses = np.zeros(sens.shape, np.complex)

    u = np.zeros(target.shape, np.complex)

    lipschitz = np.linalg.svd(A * A.H * np.ones(target.shape, np.complex),
                              compute_uv=False)[0]
    tau = 1.0 / lipschitz
    sigma = 0.01
    lamda = 0.01

    # build proxg, includes all constraints:
    def proxg(alpha, pulses):
        # instantaneous power constraint
        func = (pulses / (1 + lamda * alpha)) * \
               np.minimum(pinst/np.abs(pulses) ** 2, 1)
        # avg power constraint for each of Nc channels
        for i in range(pulses.shape[0]):
            norm = np.linalg.norm(func[i], 2, axis=0)
            func[i] *= np.minimum(pavg / (norm ** 2 / len(pulses[i])), 1)

        return func

    if pinst == float('inf') and pavg == float('inf'):
        alg_method = sp.alg.ConjugateGradient(A.H * A, pulses,
                                              A.H * target, P=None,
                                              max_iter=max_iter, tol=tol)
    else:
        alg_method = sp.alg.PrimalDualHybridGradient(
            lambda alpha, u: (u - alpha * target) / (1 + alpha),
            lambda alpha, pulses: proxg(alpha, pulses),
            lambda pulses: A * pulses,
            lambda pulses: A.H * pulses,
            pulses, u, tau, sigma, max_iter=max_iter, tol=tol)

    while not alg_method.done():
        alg_method.update()

    return pulses
