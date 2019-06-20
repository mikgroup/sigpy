# -*- coding: utf-8 -*-
"""MRI RF excitation pulse design functions
"""
import numpy as np
import sigpy as sp

from sigpy.mri import linop

__all__ = ['stspa']


def stspa(target, sens, mask, coord=None, max_iter=50, tol=1E-3):
    """Small tip spatial domain method for multicoil parallel excitation

    Args:
        target (array): desired magnetization profile.
        sens (array): sensitivity maps.
        mask (array): kspace sampling pattern
        coord (array): coordinates for noncartesian trajectories
        max_iter (int): max number of iterations
        tol (float): allowable error

    References:
        Grissom, W., Yip, C., Zhang, Z., Stenger, V. A., Fessler, J. A.
        & Noll, D. C. (2006).
        Spatial Domain Method for the Design of RF Pulses in Multicoil
        Parallel Excitation. Magnetic resonance in medicine, 56, 620-629.
    """

    A = linop.Sense(sens, coord, weights=mask, ishape=target.shape).H

    pulses = np.zeros(sens.shape, np.complex)

    alg_method = sp.alg.ConjugateGradient(A.H*A, A.H*target, pulses,
                                          P=None, max_iter=max_iter, tol=tol)

    while not alg_method.done():
        alg_method.update()

    return pulses
