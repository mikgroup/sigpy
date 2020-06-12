# -*- coding: utf-8 -*-
"""MRI RF shimming.
"""

import sigpy as sp
import numpy as np

from sigpy import backend
from sigpy.mri import rf as rf


__all__ = ['calc_shims', 'minibatch', 'multivariate_gaussian', 'gaussian_1d',
           'gaussian_pdf', 'init_optimal_spectral', 'init_circ_polar']


def calc_shims(shim_roi, sens, x0, dt, lamb=0, max_iter=50, mini=False):
    """RF shim designer. Uses the Gerchberg Saxton algorithm to calculate
    shim weights.

     Args:
        shim_roi (array): region within volume to be shimmed. Mask of 1's and
            0's. [dim_x dim_y dim_z]
        sens (array): sensitivity maps. [Nc dim_x dim_y dim_z]
        x0 (array) initial guess for shim values. [Nc 1]
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

    alg_method = sp.alg.GerchbergSaxton(A, shim_roi, x0, max_iter=max_iter,
                                        tol=10E-9, lamb=lamb)
    while not alg_method.done():
        alg_method.update()

    return alg_method.x


def minibatch(A, ncol, mask, p, linop_o=True, check_small=False):
    """Function to minibatch col of a non-composed linear operator. Returns a
    subset of the columns, along with the corresponding indices relative to A.
    If mask is included, only select columns corresponding to nonzero y
    indices. Also returns the entries of the mask corresponding to the indices.

        Args:
            A (linop): sigpy Linear operator.
            ncol (int): number of columns to include in minibatch.
            mask (array): area in y within which indices are selected.
            p (array): a 2D pdf distribution, with the same dimensions as mask.
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

        a_col_num = A.oshape[0] * A.oshape[1]
        if mask is not None:
            n = mask.shape[0]

            # process mask to make sure it's correct shape/type for sampling
            mask = mask.flatten().astype(xp.int)
            mask_inds = xp.nonzero(mask)[0]

            # do not minibatch in the case of very small numbers of nonzero
            # columns.
            if ncol < n * n * 0.005 and check_small:
                inds = mask_inds
            elif ncol < xp.size(mask_inds):
                # developer's note: replace=False not implemented in cupy
                inds = xp.random.choice(mask_inds, ncol, replace=True, p=p)
            else:
                # asking for more indices than exist in mask
                inds = mask_inds
        else:
            # no mask provided, any columns
            inds = xp.random.choice(a_col_num, ncol, replace=True)

        # create a sampling matrix S
        S = xp.zeros((inds.size, Anum.shape[0]))
        xind = xp.linspace(0, inds.size - 1, inds.size).astype(int)
        S[xind, inds] = 1
        inds = sp.to_device(inds, device)
        inds = xp.sort(inds)

        # sample using our sampling matrix
        Anum = S @ Anum

        # sample the mask at the corresponding locations
        mask = np.expand_dims(mask.flatten()[inds], 1)

        # finally, "rebundle" A as a linop again if desired
        if linop_o:
            Anum = sp.linop.MatMul(A.ishape, Anum)

        return Anum, mask, inds


def multivariate_gaussian(n, mu, sigma, device=-1):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    Based on code by Christian Hill, International Atomic Energy Agency.
    https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/
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


def gaussian_pdf(mask, sigfact):
    device = backend.get_device(mask)
    xp = device.xp
    with device:
        # get PDF sigma and centroid
        n = mask.shape[0]
        nonzero = xp.nonzero(mask)
        nonzero_x, nonzero_y = nonzero[0], nonzero[1]
        max_x, max_y = xp.amax(nonzero_x), xp.amax(nonzero_y)
        min_x, min_y = xp.amin(nonzero_x), xp.amin(nonzero_y)

        rx = max_x - min_x
        ry = max_y - min_y
        cx = (max_x + min_x) / 2
        cy = (max_y + min_y) / 2

        mu = xp.array([cy, cx])
        sigma = xp.zeros((2, 2))
        sigma[0, 0] = ry * sigfact
        sigma[1, 1] = rx * sigfact

        # create the 2D pdf from with columns will be pulledgit
        centered_pdf = multivariate_gaussian(n, mu, sigma, device)
        centered_pdf = centered_pdf.flatten()

        mask = mask.flatten()
        mask = mask.astype(xp.int)
        mask_inds = xp.nonzero(mask)[0]

        centered_pdf = centered_pdf[mask_inds]
        p = centered_pdf / xp.sum(centered_pdf)
        return p


def init_optimal_spectral(A, sens, preproc=False):
    """Function to return initial shim weights based on an optimal spectral
     method, an eigenvector-based method.

        Args:
            A (linop): sigpy Linear operator.
            sens (array): sensitivity maps. [Nc dim_x dim_y]
            preproc (bool): option to apply preprocessing function before
                finding eigenvectors

        References:
            Chandra, R., Zhong, Z., Hontz, J., McCulloch, V., Studer, C.,
            Goldstein, T. (2017) 'PhasePack: A Phase Retrieval Library.'
            arXiv:1711.10175.
    """
    # convert to numpy linop
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
        if preproc:
            T = (yplus - 1) / (yplus + xp.sqrt(delta) - 1)

            # unnormalize
            T = T * ymean
            T = xp.transpose(xp.expand_dims(T, axis=1))

            for mm in range(m):
                col = Anum[mm, :]
                aat = col * xp.transpose(col)
                Y = Y + (1 / m) * T[mm] * aat

        Y = (1 / m) * Anumt @ Anum

        w, v = xp.linalg.eigh(Y)

        return xp.expand_dims(v[:, 0], 1)


def init_circ_polar(sens):
    """Function to return circularly polarized initial shim weights. Provides
     shim weights that set the phase to be even in the middle of the sens
     profiles.

        Args:
            sens (array): sensitivity maps. [Nc dim_x dim_y]
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
