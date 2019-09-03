# -*- coding: utf-8 -*-
"""MRI applications.
"""
import numpy as np
import sigpy as sp

from sigpy.mri import linop

__all__ = ['SpatialPtxPulses']

class SpatialPtxPulses(sp.app.App):
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

    def __init__(self, target, sens,
                 coord=None, mask=None, pinst=float('inf'),
                 pavg=float('inf'), lamda=0.01, max_iter=1000,
                 tol=1E-6, device=sp.cpu_device,
                 show_pbar=True):
        self.target = target
        self.sens = sens
        self.coord = coord
        self.mask = mask
        self.pinst = pinst
        self.pavg = pavg
        self.lamda = lamda
        self.max_iter = max_iter
        self.tol = tol
        self.device = sp.Device(device)
        self.show_pbar = show_pbar

        A = linop.Sense(self.sens, self.coord,
                        self.mask, ishape=self.target.shape).H

        if coord is not None:
            # Nc*Nt pulses
            self.pulses = np.zeros((sens.shape[0], coord.shape[0]), np.complex)
        else:
            self.pulses = np.zeros(sens.shape, np.complex)

        self.u = np.zeros(target.shape, np.complex)

        lipschitz = np.linalg.svd(A * A.H * np.ones(target.shape, np.complex),
                                  compute_uv=False)[0]
        tau = 1.0 / lipschitz
        sigma = 0.01

        # build proxg, includes all constraints:
        def proxg(alpha, pulses):
            # instantaneous power constraint
            func = (pulses / (1 + self.lamda * alpha)) * \
                   np.minimum(pinst / np.abs(pulses) ** 2, 1)
            # avg power constraint for each of Nc channels
            for i in range(pulses.shape[0]):
                norm = np.linalg.norm(func[i], 2, axis=0)
                func[i] *= np.minimum(pavg / (norm**2 / len(pulses[i])), 1)

            return func

        if self.pinst == float('inf') and self.pavg == float('inf'):
            self.alg = sp.alg.ConjugateGradient(A.H*A, A.H*self.target,
                                                self.pulses,
                                                max_iter=max_iter, tol=tol)
        else:
            self.alg = sp.alg.PrimalDualHybridGradient(
                lambda alpha, u: (u - alpha * target) / (1 + alpha),
                lambda alpha, pulses: proxg(alpha, pulses),
                lambda pulses: A * pulses,
                lambda pulses: A.H * pulses,
                self.pulses, self.u, tau, sigma, max_iter=max_iter, tol=tol)

        super().__init__(self.alg, show_pbar=show_pbar)

    def _summarize(self):
        if self.show_pbar:
            self.pbar.set_postfix(resid='{0:.2E}'.format(self.alg.resid))

    def _output(self):
        return self.pulses
