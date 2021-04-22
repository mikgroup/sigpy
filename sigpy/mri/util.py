# -*- coding: utf-8 -*-
"""MRI utilities.
"""
import numpy as np
import sigpy as sp

__all__ = ['get_cov', 'whiten', 'tseg_off_res_b_ct', 'apply_tseg']


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


def tseg_off_res_b_ct(b0, bins, lseg, dt, T):
    """ Creates B and Ct matrices needed for time-segmented off-resonance
    compensation.

    Args:
        b0 (array): inhomogeneity matrix.
        bins (int): number of histogram bins to use.
        lseg (int): number of time segments.
        dt (float): hardware dwell time (ms).
        T (float): length of pulse (ms).

    Returns:
        2-element tuple containing

        - **B** (*array*): temporal interpolator.
        - **Ct** (*array*): off-resonance phase at each time segment center.
    """

    # create time vector
    t = np.linspace(0, T, np.int(T/dt))
    hist_wt, bin_edges = np.histogram(np.imag(2j * np.pi * np.concatenate(b0)),
                                      bins)

    # Build B and Ct
    bin_centers = bin_edges[1:] - bin_edges[1]/2
    zk = 0 + 1j * bin_centers
    tl = np.linspace(0, lseg, lseg) / lseg * T / 1000  # time seg centers
    # calculate off-resonance phase @ each time seg, for hist bins
    ch = np.exp(-np.expand_dims(tl, axis=1) @ np.expand_dims(zk, axis=0))
    w = np.diag(np.sqrt(hist_wt))
    p = np.linalg.pinv(w @ np.transpose(ch)) @ w
    b = p @ np.exp(-np.expand_dims(zk, axis=1)
                   @ np.expand_dims(t, axis=0) / 1000)
    b = np.transpose(b)
    b0_v = np.expand_dims(2j * np.pi * np.concatenate(b0), axis=0)
    ct = np.transpose(np.exp(-np.expand_dims(tl, axis=1) @ b0_v))

    return b, ct


def apply_tseg(array_in, coord, b, ct, fwd=True):
    """Apply the temporal interpolator and phase shift maps calculated

        Args:
            array_in (array): array to apply correction to.
            coord (array): coordinates for noncartesian trajectories. [Nt 2].
            b (array): temporal interpolator.
            ct (array): off-resonance phase at each time segment center.
            fwd (Boolean): indicates forward direction (img -> kspace) or
                backward (kspace->img)

        Returns:
            out (array): array with correction applied.
    """

    # get number of time segments from B input.
    lseg = b.shape[1]
    dim = array_in.shape[0]

    out = 0
    if fwd:
        for ii in range(lseg):
            ctd = np.reshape(ct[:, ii] * array_in.flatten(), (dim, dim))
            out = out + b[:, ii] * sp.fourier.nufft(ctd, coord * 20)

    else:
        for ii in range(lseg):
            ctd = np.reshape(np.conj(ct[:, ii]) * array_in.flatten(),
                             (dim, dim))
            out = out + sp.fourier.nufft(ctd, coord * 20) * np.conj(b[:, ii])

    return np.expand_dims(out, 1)
