# -*- coding: utf-8 -*-
"""MRI RF shimming.
"""

import sigpy as sp
import numpy as np
import cv2
from sigpy.mri import rf as rf


__all__ = ['calc_shims', 'minibatch', 'minibatch_orthogonal', 'multivariate_gaussian']


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


def minibatch(A, ncol, mask=None, currentm=None, pdf_dist=True, linop_o=True, redfact=1):
    """Function to minibatch a non-composed linear operator. Returns a
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

    # first, extract the numpy array from the Linop
    if hasattr(A, 'repr_str') and A.repr_str == 'pTx spatial explicit':
        Anum = A.linops[1].mat
    else:
        Anum = A

    a_col_num = Anum.shape[0]
    rng = np.random.default_rng()
    if mask is not None:

        n = mask.shape[0]

        # code for superimposing 2D Gaussian distribution
        if pdf_dist:
            centroid = cv2.moments(mask)
            cx = int(centroid["m10"] / centroid["m00"])
            cy = int(centroid["m01"] / centroid["m00"])
            # Mean vector and covariance matrix
            mu = np.array([cx, cy])
            sigma = np.array([[n, 0], [0, n]])

            centered_pdf = multivariate_gaussian(n, mu, sigma)
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
            ncol = int(len(np.nonzero(p)[0])/redfact)

        # else: just use uniform random distribution
        else:
            mask = mask.flatten()
            mask_inds = mask.nonzero()[0]
            p = np.squeeze(np.ones((len(mask_inds),1))/len(mask_inds))

        if ncol < 2:
            inds = mask_inds
        elif ncol < len(mask_inds):
            inds = rng.choice(mask_inds, ncol, replace=False, p=p)
        else:
            # asking for more indices than exist in mask
            inds = mask_inds
    else:
        inds = rng.choice(a_col_num, ncol, replace=False)
    inds = np.sort(inds)
    samps = np.zeros((n * n, 1))
    samps[inds] = 1
    Anum = Anum[inds, :]

    # finally, "rebundle" A as a linop again if desired
    if linop_o:
        Anum = sp.linop.MatMul(A.ishape, Anum)

    return Anum, inds

def minibatch_orthogonal(A, ncol, mask=None, currentm=None,  linop_o=True, redfact=1):
    # first, extract the numpy array from the Linop
    if hasattr(A, 'repr_str') and A.repr_str == 'pTx spatial explicit':
        Anum = A.linops[1].mat
    else:
        Anum = A

    a_col_num = Anum.shape[0]
    rng = np.random.default_rng()
    if mask is not None:
        n = mask.shape[0]
        mask = mask.flatten()
        mask_inds = mask.nonzero()[0]

def multivariate_gaussian(n, mu, sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    x = np.linspace(0, n, n)
    y = np.linspace(0, n, n)
    x, y = np.meshgrid(x, y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    n = mu.shape[0]
    sigma_det = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)
    n = np.sqrt((2 * np.pi) ** n * sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, sigma_inv, pos-mu)

    return np.exp(-fac / 2) / n
