# -*- coding: utf-8 -*-
"""MRI RF shimming.
"""

import sigpy as sp
import numpy as np

from sigpy import backend
from sigpy.mri import rf as rf
from scipy.ndimage import center_of_mass



__all__ = ['calc_shims', 'minibatch', 'multivariate_gaussian', 'gaussian_1d',
           'init_optimal_spectral']


def calc_shims(shim_roi, sens, dt, lamb=0, max_iter=50, mini=False):
    """RF shim designer.

     Args:
        shim_roi (array): region within volume to be shimmed. Mask of 1's and
            0's. [dim_x dim_y dim_z]
        sens (array): sensitivity maps. [Nc dim_x dim_y dim_z]
        dt (float): hardware sampling dwell time.
        lamb (float): regularization term.
        max_iter (int): max number of iterations.
        mini (bool): option to use system matrix minibatching.

    Returns:
        array: complex shims.
    """

    k1 = np.expand_dims(np.array((0, 0, 0)), 0)
    A = sp.mri.rf.PtxSpatialExplicit(sens, coord=k1, dt=dt,
                                     img_shape=shim_roi.shape, ret_array=False)

    alg_method = sp.alg.GerchbergSaxton(A, shim_roi, max_iter=max_iter,
                                        tol=10E-9, lamb=lamb, minibatch=mini)
    while not alg_method.done():
        alg_method.update()

    return alg_method.x


def minibatch(A, ncol, mask, currentm=None, pdf_dist=True, linop_o=True,
              sigfact=2):
    """Function to minibatch col of a non-composed linear operator. Returns a
    subset of the columns, along with the corresponding indices relative to A.
    If mask is included, only select columns corresponding to nonzero y
    indices.

        Args:
            A (linop): sigpy Linear operator
            ncol (int): number of columns to include in minibatch.
            mask (array): area in y within which indices are selected.
            currentm (array): current magnitude |B1+| shim.
            pdf_dist (bool): use a spatially varying centroid centered
                multivariate gaussian pdf for sample selection.
            linop_o (bool): return a sigpy Linop. Else return a numpy ndarray.
            redfact (float): takes 1/redfact * col in minibatch.
    """
    device = backend.get_device(mask)
    xp = device.xp
    with device:
        # first, extract the numpy array from the Linop
        if hasattr(A, 'repr_str') and A.repr_str == 'pTx spatial explicit':
            Anum = A.linops[1].mat
        else:
            Anum = A
        Anum = sp.to_device(Anum, device)

        a_col_num = Anum.shape[0]
        if mask is not None:

            n = mask.shape[0]

            # code for superimposing 2D Gaussian distribution
            if pdf_dist:
                # get PDF sigma and centroid
                nonzero = xp.nonzero(mask)
                nonzero_x, nonzero_y = nonzero[0], nonzero[1]
                rx = (max(nonzero_x) - min(nonzero_x)) * sigfact
                ry = (max(nonzero_y) - min(nonzero_y)) * sigfact
                # center of mass of mask
                cx, cy = center_of_mass(mask)
                mu = xp.array([cx, cy])
                sigma = xp.zeros((2, 2))
                sigma[0, 0] = rx
                sigma[1, 1] = ry

                # create the 2D pdf from with columns will be pulled
                centered_pdf = multivariate_gaussian(n, mu, sigma, device)
                centered_pdf = sp.to_device(centered_pdf, device)
                centered_pdf = centered_pdf.flatten()

                m = currentm.flatten()
                mask = mask.flatten()
                mask_inds = mask.nonzero()[0]

                centered_pdf = centered_pdf[mask_inds]
                m = abs(m[mask_inds])
                p = abs(1-m)/m
                p[p < 0] = 0
                p[p > 1] = 1
                p = p * centered_pdf  # apply centered multivariate pdf
                p = p / sum(p)
                ncol = ncol

            # else: just use uniform random distribution
            else:
                mask = mask.flatten()
                mask_inds = mask.nonzero()[0]
                p = xp.squeeze(xp.ones((len(mask_inds),1))/len(mask_inds))

            if ncol < 2:
                inds = mask_inds
            elif ncol < len(mask_inds):
                # replace = False is preferable, but not implemented in cupy
                inds = xp.random.choice(mask_inds, ncol, replace=True, p=p)
            else:
                # asking for more indices than exist in mask
                inds = mask_inds
        else:
            inds = xp.random.choice(a_col_num, ncol, replace=True)
        inds = sp.to_device(inds, device)
        inds = xp.sort(inds)
        samps = xp.zeros((n * n, 1))
        samps[inds] = 1
        Anum = Anum[inds, :]

        # finally, "rebundle" A as a linop again if desired
        if linop_o:
            Anum = sp.linop.MatMul(A.ishape, Anum)

        return Anum, inds


def multivariate_gaussian(n, mu, sigma, device=-1):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """
    xp = device.xp
    with device:
        x = xp.linspace(0, n, n)
        y = xp.linspace(0, n, n)
        x, y = xp.meshgrid(x, y)

        # Pack X and Y into a single 3-dimensional array
        pos = xp.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y

        n = mu.shape[0]
        sigma_det = xp.linalg.det(sigma)
        sigma_inv = xp.linalg.inv(sigma)
        n = xp.sqrt((2 * xp.pi) ** n * sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = xp.einsum('...k,kl,...l->...', pos-mu, sigma_inv, pos-mu)

        return xp.exp(-fac / 2) / n


def gaussian_1d(x, m, sigma):
    f_x = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x-m)/sigma) ** 2)
    return f_x


def init_optimal_spectral(A, b0, preproc=False):
    # convert to numpy linop
    device = backend.get_device(b0)
    xp = device.xp
    with device:
        if hasattr(A, 'repr_str') and A.repr_str == 'pTx spatial explicit':
            Anum = A.linops[1].mat
        else:
            Anum = A

        b0 = b0.flatten()
        n = Anum.shape[1]
        Anumt = xp.transpose(Anum)

        m = len(b0)
        y = b0 ** 2

        # normalize the measurements
        delta = m / n
        ymean = y / xp.mean(y)

        # apply pre-processing function
        yplus = max(y)
        if preproc:
            T = (yplus - 1) / (yplus + xp.sqrt(delta) - 1)

            # unnormalize
            T = T * ymean
            T = xp.transpose(xp.expand_dims(T, axis=1))

            #Y = np.zeros((n, 1))
            for mm in range(m):
                col = Anum[mm, :]
                aat = col * xp.transpose(col)
                Tbi = T[mm]
                Y = Y + (1 / m) * T[mm] * aat

        Y = (1 / m) * Anumt @ Anum

        w, v = xp.linalg.eigh(Y)

        return xp.expand_dims(v[:,0], 1)

