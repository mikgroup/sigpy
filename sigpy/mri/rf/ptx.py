# -*- coding: utf-8 -*-
"""MRI RF excitation pulse design functions,
    including SLR and small tip spatial design
"""

import sigpy as sp
import sigpy.mri.rf as rf
from sigpy import backend

__all__ = ['stspa']


def stspa(target, sens, coord, dt, alpha=0, B0=None, pinst=float('inf'),
          pavg=float('inf'), phase_update_interval=0, tseg=False,
          explicit=False, max_iter=1000, tol=1E-6):
    """Small tip spatial domain method for multicoil parallel excitation.
       Allows for constrained or unconstrained designs.

    Args:
        target (array): desired magnetization profile. [dim dim]
        sens (array): sensitivity maps. [Nc dim dim]
        coord (array): coordinates for noncartesian trajectories. [Nt 2]
        dt (float): hardware sampling dwell time.
        alpha (float): regularization term, if unconstrained.
        B0 (array): B0 inhomogeneity map [dim dim]. Explicit matrix only.
        pinst (float): maximum instantaneous power.
        pavg (float): maximum average power.
        phase_update_interval (int): number of iters between exclusive phase
         updates. If 0, no phase updates performed.
        explicit (bool): Use explicit matrix.
        max_iter (int): max number of iterations.
        tol (float): allowable error.

    Returns:
        array: pulses out.

    References:
        Grissom, W., Yip, C., Zhang, Z., Stenger, V. A., Fessler, J. A.
        & Noll, D. C.(2006).
        Spatial Domain Method for the Design of RF Pulses in Multicoil
        Parallel Excitation. Magnetic resonance in medicine, 56, 620-629.
    """
    Nc = sens.shape[0]
    Nt = coord.shape[0]
    device = backend.get_device(target)
    xp = device.xp
    with device:

        pulses = xp.zeros((sens.shape[0], coord.shape[0]), xp.complex)
        # set up the system matrix
        if explicit:
            # reshape pulses to be Nc * Nt, 1 - all 1 vector
            pulses = xp.concatenate(pulses)
            # add empty dimension to make (Nt, 1)
            pulses = xp.transpose(xp.expand_dims(pulses, axis=0))

            # explicit matrix design linop
            A = rf.linop.PtxSpatialExplicit(sens, coord, dt,
                                                   target.shape, B0)

            # explicit AND constrained, must reshape A output for PDHG
            if pinst != float('inf') or pavg != float('inf'):
                A = sp.linop.Reshape(target.shape, A.oshape) * A

        # using non-explicit formulation
        else:
            A = sp.mri.linop.Sense(sens, coord, weights=None, B0=B0, dt=dt,
                                   ishape=target.shape).H

        # Unconstrained, use conjugate gradient
        if pinst == float('inf') and pavg == float('inf'):

            if explicit:
                I = sp.linop.Identity((coord.shape[0] * Nc, 1))
                b = A.H * xp.transpose(xp.expand_dims(xp.concatenate(target),
                                                      axis=0))

                alg_method = sp.alg.ConjugateGradient(A.H * A + alpha * I,
                                                      b, pulses, P=None,
                                                      max_iter=max_iter, tol=tol)

            else:
                I = sp.linop.Identity((Nc, coord.shape[0]))
                b = A.H * target

                alg_method = sp.alg.ConjugateGradient(A.H * A + alpha * I,
                                                      b, pulses, P=None,
                                                      max_iter=max_iter, tol=tol)

        # Constrained case, use primal dual hybrid gradient
        else:
            u = xp.zeros(target.shape, xp.complex)
            lipschitz = xp.linalg.svd(A * A.H *
                                      xp.ones(A.H.ishape, xp.complex),
                                      compute_uv=False)[0]
            tau = 1.0 / lipschitz
            sigma = 0.01
            lamda = 0.01

            # build proxg, includes all constraints:
            def proxg(alpha, pulses):
                # instantaneous power constraint
                func = (pulses / (1 + lamda * alpha)) * \
                       xp.minimum(pinst / xp.abs(pulses) ** 2, 1)
                # avg power constraint for each of Nc channels
                for i in range(pulses.shape[0]):
                    norm = xp.linalg.norm(func[i], 2, axis=0)
                    func[i] *= xp.minimum(pavg /
                                          (norm ** 2 / len(pulses[i])), 1)

                return func

            alg_method = sp.alg.PrimalDualHybridGradient(
                lambda alpha, u: (u - alpha * target) / (1 + alpha),
                lambda alpha, pulses: proxg(alpha, pulses),
                lambda pulses: A * pulses,
                lambda pulses: A.H * pulses,
                pulses, u, tau, sigma, max_iter=max_iter, tol=tol)

        # perform the design: apply optimization method to find solution pulse
        while not alg_method.done():

            # phase_update switch
            if (alg_method.iter % phase_update_interval == 0) and (
                    alg_method.iter > 0):
                # put correct m into alg_method (Ax=b notation)
                if explicit:
                    b = A.H * xp.transpose(
                        xp.expand_dims(xp.concatenate(target), axis=0))
                else:
                    target = xp.abs(target) * xp.exp(
                        1j * xp.angle(
                            xp.reshape(A * alg_method.x, target.shape)))
                    b = A.H * target
                alg_method.b = b

            alg_method.update()

        # reshape explicit pulses so output is always [Nc Nt]
        if explicit:
            pulses = xp.reshape(pulses, (Nc, Nt))

        return pulses
