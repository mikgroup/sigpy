# -*- coding: utf-8 -*-
"""MRI RF excitation pulse design functions
"""

import numpy as np
import sigpy as sp

from sigpy.mri import linop

__all__ = ['stspa']

def stspa(target, sens, mask, coord=None,pinst=float('inf'),pavg=float('inf'), max_iter=1000, tol=1E-6):
    """Small tip spatial domain method for multicoil parallel excitation.
       Allows for constrained or unconstrained designs.

    Args:
        target (array): desired magnetization profile.
        sens (array): sensitivity maps.
        mask (array): kspace sampling pattern
        coord (array): coordinates for noncartesian trajectories
        pinst (float): maximum instantaneous power
        pavg (float): maximum average power
        max_iter (int): max number of iterations
        tol (float): allowable error

    References:
        Grissom, W., Yip, C., Zhang, Z., Stenger, V. A., Fessler, J. A.
        & Noll, D. C.(2006).
        Spatial Domain Method for the Design of RF Pulses in Multicoil
        Parallel Excitation.Magnetic resonance in medicine, 56, 620-629.
    """

    A = linop.Sense(sens, coord, weights=mask, ishape=target.shape).H
    anp = A * np.repeat(np.eye(sens.shape[1])[np.newaxis, :, :], 8, axis=0)

    pulses = np.zeros(sens.shape, np.complex)
    u = np.zeros(target.shape, np.complex)

    lipschitz = np.linalg.svd(anp.T @ anp, compute_uv=False)[0]
    tau = 1.0 / lipschitz
    sigma = 0.1
    lamda = 0.1

    alg_method = sp.alg.PrimalDualHybridGradient(
        lambda alpha, u: (u - alpha * target) / (1 + alpha),
        lambda alpha, pulses: (pulses / (1 + lamda * alpha)) * np.minimum(pinst/np.abs(pulses ) ** 2 , 1)
                              * np.minimum(pavg/((np.linalg.norm(np.concatenate(np.concatenate(pulses)) / (1 + lamda * alpha),2, axis=0) ** 2)/len(pulses)) , 1),
        lambda pulses: A * pulses,
        lambda pulses: A.H * pulses,
        pulses, u, tau, sigma, max_iter=max_iter, tol=tol)

    while not alg_method.done():
        alg_method.update()

    return pulses

