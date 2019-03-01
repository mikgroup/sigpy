# -*- coding: utf-8 -*-
"""MRI utilities.
"""
import numpy as np


__all__ = ['get_cov', 'whiten']


def get_cov(noise):
    """Get covariance matrix from noise measurements.

    Args:
        noise (array): Noise measurements of shape [num_coils, ...]

    Returns:
        array: num_coils x num_coils covariance matrix.

    """
    num_coils = noise.shape[0]
    X = noise.reshape([num_coils, -1])
    X -= np.mean(X, axis=-1, keepdims=True)
    cov = np.matmul(X, X.T.conjugate())

    return cov


def whiten(ksp, cov):
    """Whitens k-space measurements.

    Args:
        ksp (array): k-space measurements of shape [num_coils, ...]
        cov (array): num_coils x num_coils covariance matrix.

    Returns:
        array: whitened k-space array.

    """
    num_coils = ksp.shape[0]

    x = ksp.reshape([num_coils, -1])

    L = np.linalg.cholesky(cov)
    x_w = np.linalg.solve(L, x)
    ksp_w = x_w.reshape(ksp.shape)

    return ksp_w
