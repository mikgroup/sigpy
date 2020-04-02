# -*- coding: utf-8 -*-
"""MRI RF excitation pulse design functions,
    including SLR and small tip spatial design
"""

import sigpy as sp
from sigpy.mri import rf as rf
from sigpy import backend

__all__ = ['stspa', 'wstspa']


def stspa(target, sens, coord, dt, alpha=0, B0=None, tseg=None,
          pinst=float('inf'), pavg=float('inf'),
          phase_update_interval=float('inf'), explicit=False, max_iter=1000,
          tol=1E-6):
    """Small tip spatial domain method for multicoil parallel excitation.
       Allows for constrained or unconstrained designs.

    Args:
        target (array): desired magnetization profile. [dim dim]
        sens (array): sensitivity maps. [Nc dim dim]
        coord (array): coordinates for noncartesian trajectories. [Nt 2]
        dt (float): hardware sampling dwell time.
        alpha (float): regularization term, if unconstrained.
        B0 (array): B0 inhomogeneity map [dim dim]. For explicit matrix
            building.
        tseg (None or Dictionary): parameters for time-segmented off-resonance
            correction. Parameters are 'b0' (array), 'dt' (float),
            'lseg' (int), and 'n_bins' (int). Lseg is the number of
            time segments used, and n_bins is the number of histogram bins.
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

        pulses = xp.zeros((Nc, Nt), xp.complex)

        # set up the system matrix
        if explicit:
            A = rf.linop.PtxSpatialExplicit(sens, coord, dt,
                                            target.shape, B0)
        else:
            A = sp.mri.linop.Sense(sens, coord, weights=None, tseg=tseg,
                                   ishape=target.shape).H

        # Unconstrained, use conjugate gradient
        if pinst == float('inf') and pavg == float('inf'):
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
            lipschitz = sum(lipschitz)/len(lipschitz)
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
            if (alg_method.iter > 0) and \
                    (alg_method.iter % phase_update_interval == 0):
                target = xp.abs(target) * xp.exp(
                    1j * xp.angle(
                        xp.reshape(A * alg_method.x, target.shape)))
                b = A.H * target
                alg_method.b = b

            alg_method.update()

        return pulses


def wstspa(target, sens, coord, alpha=0, max_iter=1000,
           tol=1E-6, P=None):
    """Experimental wavelet spatial domain pulse designer for ptx.

    Args:
        target (array): desired magnetization profile. [dim dim]
        sens (array): sensitivity maps. [Nc dim dim]
        coord (array): coordinates for noncartesian trajectories. [Nt 2]
        alpha (float): regularization term.
        max_iter (int): max number of iterations.
        tol (float): allowable error.
        P (linop): matrix for taking a coefficient subset of the wavelet
            transform.

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

        pulses = xp.zeros((Nc, Nt), xp.complex)

        # building out inverse sense linop explicitly
        img_shape = sens.shape[1:]
        W = sp.linop.Wavelet(img_shape)

        S = sp.linop.Multiply(img_shape, sens)
        F = sp.linop.NUFFT(S.oshape, coord)

        A = (F * S * W.H).H

        I = sp.linop.Identity((Nc, coord.shape[0]))

        cg = False
        # CG implementation
        if P is None:
            if cg:
                b = A.H * W * target

                alg_method = sp.alg.ConjugateGradient(A.H * A + alpha * I,
                                                      b, pulses, P=None,
                                                      max_iter=max_iter,
                                                      tol=tol)
            else:
                # gradient method implementation
                proxg = sp.prox.L1Reg(A.ishape, alpha)
                a = 0.01
                def gradf(x):
                    return A.H * (A * pulses - W * target)

                alg_method = sp.alg.GradientMethod(gradf, pulses, a, proxg=proxg,
                                                   max_iter=max_iter)

        # sampled wavelet implementation
        else:
            A = (F * S * W.H * P.H).H
            if cg:
                b = A.H * P * W * target

                alg_method = sp.alg.ConjugateGradient(A.H * A + alpha * I,
                                                      b, pulses, P=None,
                                                      max_iter=max_iter,
                                                      tol=tol)
            else:
                # gradient method implementation
                proxg = sp.prox.L1Reg(A.ishape, alpha)
                a = 0.01

                def gradf(x):
                    return A.H * (A * pulses - P * W * target)

                alg_method = sp.alg.GradientMethod(gradf, pulses, a,
                                                   proxg=proxg,
                                                   max_iter=max_iter)

        # perform the design: apply optimization method to find solution pulse
        while not alg_method.done():
            alg_method.update()

        return pulses
