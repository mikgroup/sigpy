# -*- coding: utf-8 -*-
"""MRI RF excitation pulse design functions,
    including SLR and small tip spatial design
"""

import numpy as np
import sigpy as sp

from sigpy.mri import linop

__all__ = ['stspa']


def stspa(target, sens, coord, dt, alpha=0, B0=None, pinst=float('inf'),
          pavg=float('inf'), explicit=False, phase_update_interval =float('inf'), max_iter=1000, tol=1E-6):
    """Small tip spatial domain method for multicoil parallel excitation.
       Allows for constrained or unconstrained designs.

    Args:
        target (array): desired magnetization profile. [dim dim]
        sens (array): sensitivity maps. [Nc dim dim]
        coord (array): coordinates for noncartesian trajectories [Nt 2]
        dt (float): hardware sampling dwell time
        alpha (float): regularization term
        B0 (array): B0 inhomogeneity map [dim dim]. Not supported for nonexplicit matrix
        pinst (float): maximum instantaneous power
        pavg (float): maximum average power
        explicit (bool): Use explicit matrix
        max_iter (int): max number of iterations
        tol (float): allowable error

    References:
        Grissom, W., Yip, C., Zhang, Z., Stenger, V. A., Fessler, J. A.
        & Noll, D. C.(2006).
        Spatial Domain Method for the Design of RF Pulses in Multicoil
        Parallel Excitation. Magnetic resonance in medicine, 56, 620-629.
    """

    Nc = sens.shape[0]
    T = dt * coord.shape[0]  # duration of pulse, in seconds
    t = np.expand_dims(np.linspace(0, T, coord.shape[0]), axis=1)  # create time vector

    pulses = np.zeros((sens.shape[0], coord.shape[0]), np.complex)

    if explicit:
        # reshape pulses to be Nc * Nt, 1 - all 1 vector
        pulses = np.concatenate(pulses)
        pulses = np.transpose(np.expand_dims(pulses, axis=0))  # add empty dimension to make (Nt, 1)

        # explicit matrix design linop
        A = sp.mri.rf.linop.PtxSpatialExplicit(sens, coord, dt, target.shape, B0, comm=None)

        # explicit AND constrained, must reshape A output
        # TODO: more elegant solution with matrix shape should be found than custom or varying between cases
        # TODO: part of problem is that explicit formulation has different shape than nonexplicit formulation.
        # TODO: Reshape nonexplicit so consistent?

        if pinst != float('inf') or pavg != float('inf'):
            # reshape output to play nicely with PDHG
            R = sp.linop.Reshape(target.shape, A.oshape)
            A = R * A

    # using non-explicit formulation
    else:
        A = sp.mri.linop.Sense(sens, coord, None, ishape=target.shape).H
        I = sp.linop.Identity((Nc, coord.shape[0]))

    # Unconstrained, use conjugate gradient
    if pinst == float('inf') and pavg == float('inf'):

        if explicit:
            I = sp.linop.Identity((coord.shape[0] * Nc, 1))
            b = A.H * np.transpose(np.expand_dims(np.concatenate(target), axis=0))

        else:
            I = sp.linop.Identity((Nc, coord.shape[0]))
            b = A.H * target

        alg_method = sp.alg.ConjugateGradient(A.H * A + alpha * I, b, pulses, P=None,
                                              max_iter=max_iter, tol=tol)

    # Constrained, use primal dual hybrid gradient
    else:
        u = np.zeros(target.shape, np.complex)
        lipschitz = np.linalg.svd(A * A.H * np.ones(A.H.ishape, np.complex),
                                  compute_uv=False)[0]
        tau = 1.0 / lipschitz
        sigma = 0.01
        lamda = 0.01

        # build proxg, includes all constraints:
        def proxg(alpha, pulses):
            # instantaneous power constraint
            func = (pulses / (1 + lamda * alpha)) * \
                   np.minimum(pinst / np.abs(pulses) ** 2, 1)
            # avg power constraint for each of Nc channels
            for i in range(pulses.shape[0]):
                norm = np.linalg.norm(func[i], 2, axis=0)
                func[i] *= np.minimum(pavg / (norm ** 2 / len(pulses[i])), 1)

            return func

        alg_method = sp.alg.PrimalDualHybridGradient(
            lambda alpha, u: (u - alpha * target) / (1 + alpha),
            lambda alpha, pulses: proxg(alpha, pulses),
            lambda pulses: A * pulses,
            lambda pulses: A.H * pulses,
            pulses, u, tau, sigma, max_iter=max_iter, tol=tol)

    # finally, apply optimization method to find solution pulse
    while not alg_method.done():

        # phase_update switch
        if (alg_method.iter % phase_update_interval == 0) and (alg_method.iter > 0):
            target = np.abs(target) * np.exp(1j * np.angle(np.reshape(A * pulses, target.shape)))
            # put correct m into alg_method (Ax=b notation)
            if explicit:
                b = A.H * np.transpose(np.expand_dims(np.concatenate(target), axis=0))
            else:
                b = A.H * target
            alg_method.b = b

        alg_method.update()

    return pulses
