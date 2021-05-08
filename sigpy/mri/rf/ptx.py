# -*- coding: utf-8 -*-
"""MRI RF excitation pulse design functions,
    including SLR and small tip spatial design
"""

import sigpy as sp
import numpy as np
from sigpy.mri import rf as rf
from sigpy import backend
from scipy.interpolate import interp1d


__all__ = ['stspa', 'stspk']


def stspa(target, sens, coord, dt, roi=None, alpha=0, b0=None, tseg=None,
          st=None, phase_update_interval=float('inf'), explicit=False,
          max_iter=1000, tol=1E-6):
    """Small tip spatial domain method for multicoil parallel excitation.
       Allows for constrained or unconstrained designs.

    Args:
        target (array): desired magnetization profile. [dim dim]
        sens (array): sensitivity maps. [Nc dim dim]
        coord (array): coordinates for noncartesian trajectories. [Nt 2]
        dt (float): hardware sampling dwell time.
        roi (array): array for error weighting, specify spatial ROI. [dim dim]
        alpha (float): regularization term, if unconstrained.
        b0 (array): B0 inhomogeneity map [dim dim]. For explicit matrix
            building.
        tseg (None or Dictionary): parameters for time-segmented off-resonance
            correction. Parameters are 'b0' (array), 'dt' (float),
            'lseg' (int), and 'n_bins' (int). Lseg is the number of
            time segments used, and n_bins is the number of histogram bins.
        st (None or Dictionary): 'subject to' constraint parameters. Parameters
            are avg power 'cNorm' (float), peak power 'cMax' (float),
            'mu' (float), 'rhoNorm' (float), 'rhoMax' (float), 'cgiter' (int),
            'max_iter' (int), 'L' (list of arrays), 'c' (float), 'rho' (float),
            and 'lam' (float). These parameters are explained in detail in the
            SDMM documentation.
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
                                            target.shape, b0)
        else:
            A = sp.mri.linop.Sense(sens, coord, weights=None, tseg=tseg,
                                   ishape=target.shape).H

        # handle the Ns * Ns error weighting ROI matrix
        W = sp.linop.Multiply(A.oshape, xp.ones(target.shape))
        if roi is not None:
            W = sp.linop.Multiply(A.oshape, roi)

        # apply ROI
        A = W * A

        # Unconstrained, use conjugate gradient
        if st is None:
            I = sp.linop.Identity((Nc, coord.shape[0]))
            b = A.H * W * target

            alg_method = sp.alg.ConjugateGradient(A.H * A + alpha * I,
                                                  b, pulses, P=None,
                                                  max_iter=max_iter, tol=tol)

        # Constrained case, use SDMM
        else:
            # vectorize target for SDMM
            target = W * target
            d = xp.expand_dims(target.flatten(), axis=0)
            alg_method = sp.alg.SDMM(A, d, st['lam'], st['L'], st['c'],
                                     st['mu'], st['rho'], st['rhoMax'],
                                     st['rhoNorm'], 10**-5, 10**-2, st['cMax'],
                                     st['cNorm'], st['cgiter'], st['max_iter'])

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

        if st is not None:
            pulses = xp.reshape(alg_method.x, [Nc, Nt])
        return pulses


def stspk(mask, sens, n_spokes, fov, dx_max, gts, sl_thick, tbw, dgdtmax, gmax,
          alpha=1, iter_dif=0.01):
    """Small tip spokes parallel transmit pulse designer.

    Args:
       mask (ndarray): region in which to optimize flip angle uniformity
           in slice. [dim dim]
       sens (ndarray): sensitivity maps. [nc dim dim]
       n_spokes (int): number of spokes to be created in the design.
       fov (float): excitation FOV (cm).
       dx_max (float): max. resolution of the trajectory (cm).
       gts (float): hardware sampling dwell time (s).
       sl_thick (float): slice thickness (mm).
       tbw (int): time-bandwidth product.
       dgdtmax (float): max gradient slew (g/cm/s).
       gmax (float): max gradient amplitude (g/cm).
       alpha (float): regularization parameter.
       iter_dif (float): for each spoke, the difference in cost btwn.
          successive iterations at which to terminate MLS iterations.

    Returns:
       2-element tuple containing

        - **pulses** (*array*): RF waveform out.
        - **g** (*array*): corresponding gradient, in g/cm.

    References:
       Grissom, W., Khalighi, M., Sacolick, L., Rutt, B. & Vogel, M (2012).
       Small-tip-angle spokes pulse design using interleaved greedy and
       local optimization methods. Magnetic Resonance in Medicine, 68(5),
       1553-62.
    """

    device = backend.get_device(sens)
    xp = device.xp
    with device:
        nc = sens.shape[0]

        kmax = 1 / dx_max  # /cm, max spatial freq of trajectory
        # greedy kx, ky grid
        kxs, kys = xp.meshgrid(xp.linspace(-kmax / 2, kmax / 2 - 1 / fov,
                                           xp.int(fov * kmax)),
                               xp.linspace(-kmax / 2, kmax / 2 - 1 / fov,
                                           xp.int(fov * kmax)))
        # vectorize the grid
        kxs = kxs.flatten()
        kys = kys.flatten()

        # remove DC
        dc = xp.intersect1d(xp.where((kxs == 0)), xp.where((kys == 0)))[0]
        kxs = xp.concatenate([kxs[:dc], kxs[dc+1:]])
        kys = xp.concatenate([kys[:dc], kys[dc+1:]])

        # step 2: design the weights
        # initial kx/ky location is DC
        k = xp.expand_dims(xp.array([0, 0]), 0)

        # initial target phase
        phs = xp.zeros((xp.count_nonzero(mask), 1), dtype=xp.complex)

        for ii in range(n_spokes):

            # build Afull (and take only 0 locations into matrix)
            Anum = rf.PtxSpatialExplicit(sens, k, gts, mask.shape,
                                         ret_array=True)
            Anum = Anum[~(Anum == 0).all(1)]

            # design wfull using MLS:
            # initialize wfull
            sys_a = (Anum.conj().T @ Anum + alpha * xp.eye((ii+1)*nc))
            sys_b = (Anum.conj().T @ xp.exp(1j*phs))
            w_full = xp.linalg.solve(sys_a, sys_b)

            err = Anum @ w_full - xp.exp(1j * phs)
            cost = err.conj().T @ err + alpha * w_full.conj().T @ w_full
            cost = xp.real(cost)
            cost_old = 10 * cost  # to get the loop going
            while xp.absolute(cost - cost_old) > iter_dif * cost_old:
                cost_old = cost
                phs = xp.angle(Anum @ w_full)
                w_full = xp.linalg.solve(
                    (Anum.conj().T @ Anum + alpha * xp.eye((ii + 1) * nc)),
                    (Anum.conj().T @ xp.exp(1j * phs)))
                err = Anum @ w_full - xp.exp(1j * phs)
                cost = xp.real(err.conj().T @ err +
                               alpha * w_full.conj().T @ w_full)

            # add a spoke using greedy method
            if ii < n_spokes - 1:

                r = xp.exp(1j * phs) - Anum @ w_full
                rfnorm = xp.zeros(kxs.shape, dtype=xp.complex)
                for jj in range(kxs.size):
                    ks_test = xp.expand_dims(xp.array([kxs[jj], kys[jj]]), 0)
                    Anum = rf.PtxSpatialExplicit(sens, ks_test, gts,
                                                 mask.shape, ret_array=True)
                    Anum = Anum[~(Anum == 0).all(1)]

                    rfm = xp.linalg.solve((Anum.conj().T @ Anum),
                                          (Anum.conj().T @ r))
                    rfnorm[jj] = xp.linalg.norm(rfm)

                ind = xp.argmax(rfnorm)
                k_new = xp.expand_dims(xp.array([kxs[ind], kys[ind]]), 0)

                if ii % 2 != 0:  # add to end of pulse
                    k = xp.concatenate((k, k_new))
                else:  # add to beginning of pulse
                    k = xp.concatenate((k_new, k))

                # remove chosen point from candidates
                kxs = xp.concatenate([kxs[:ind], kxs[ind + 1:]])
                kys = xp.concatenate([kys[:ind], kys[ind + 1:]])

        # from our spoke selections, build the whole waveforms

        # first, design our gradient waveforms:
        g = rf.spokes_grad(k, tbw, sl_thick, gmax, dgdtmax, gts)

        # design our rf
        # calc. the size of the traps in our gz waveform- will use to calc rf
        area = tbw / (sl_thick / 10) / 4257  # thick*kwid=twb, kwid=gam*area
        [subgz, nramp] = rf.min_trap_grad(area, gmax, dgdtmax, gts)
        npts = 128
        subrf = rf.dzrf(npts, tbw, 'st')

        n_plat = subgz.size - 2 * nramp  # time points on trap plateau
        # interpolate to stretch out waveform to appropriate length
        f = interp1d(np.arange(0, npts, 1) / npts, subrf,
                     fill_value='extrapolate')
        subrf = f(xp.arange(0, n_plat, 1) / n_plat)
        subrf = xp.concatenate((xp.zeros(nramp), subrf, xp.zeros(nramp)))

        pulses = xp.kron(xp.reshape(w_full, (nc, n_spokes)), subrf)

        # add zeros for gzref
        rf_ref = xp.zeros((nc, g.shape[1] - pulses.shape[1]))
        pulses = xp.concatenate((pulses, rf_ref), 1)

        return pulses, g
