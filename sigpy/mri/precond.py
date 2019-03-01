# -*- coding: utf-8 -*-
"""MRI preconditioners.
"""
import sigpy as sp


__all__ = ['kspace_precond', 'circulant_precond']


def kspace_precond(mps, weights=None, coord=None,
                   lamda=0, device=sp.cpu_device):
    r"""Compute a diagonal preconditioner in k-space.

    Considers the optimization problem:

    .. math::
        \min_P \| P A A^H - I \|_F^2

    where A is the Sense operator.

    Args:
        mps (array): sensitivity maps of shape [num_coils] + image shape.
        weights (array): k-space weights.
        coord (array): k-space coordinates of shape [...] + [ndim].
        lamda (float): regularization.

    Returns:
        array: k-space preconditioner of same shape as k-space.

    """
    dtype = mps.dtype

    if weights is not None:
        weights = sp.to_device(weights, device)

    device = sp.Device(device)
    xp = device.xp

    mps_shape = list(mps.shape)
    img_shape = mps_shape[1:]
    img2_shape = [i * 2 for i in img_shape]
    ndim = len(img_shape)

    scale = sp.prod(img2_shape)**1.5 / sp.prod(img_shape)
    with device:
        if coord is None:
            idx = (slice(None, None, 2), ) * ndim

            ones = xp.zeros(img2_shape, dtype=dtype)
            if weights is None:
                ones[idx] = 1
            else:
                ones[idx] = weights**0.5

            psf = sp.ifft(ones)
        else:
            coord2 = coord * 2
            ones = xp.ones(coord.shape[:-1], dtype=dtype)
            if weights is not None:
                ones *= weights**0.5

            psf = sp.nufft_adjoint(ones, coord2, img2_shape)

        p_inv = []
        for mps_i in mps:
            mps_i = sp.to_device(mps_i, device)
            mps_i_norm2 = xp.linalg.norm(mps_i)**2
            xcorr_fourier = 0
            for mps_j in mps:
                mps_j = sp.to_device(mps_j, device)
                xcorr_fourier += xp.abs(sp.fft(mps_i *
                                               xp.conj(mps_j), img2_shape))**2

            xcorr = sp.ifft(xcorr_fourier)
            xcorr *= psf
            if coord is None:
                p_inv_i = sp.fft(xcorr)[idx]
            else:
                p_inv_i = sp.nufft(xcorr, coord2)

            if weights is not None:
                p_inv_i *= weights**0.5

            p_inv.append(p_inv_i * scale / mps_i_norm2)

        p_inv = (xp.abs(xp.stack(p_inv, axis=0)) + lamda) / (1 + lamda)
        p_inv[p_inv == 0] = 1
        p = 1 / p_inv

        return p.astype(dtype)


def circulant_precond(mps, weights=None, coord=None,
                      lamda=0, device=sp.cpu_device):
    r"""Compute circulant preconditioner.

    Considers the optimization problem:

    .. math::
        \min_P \| A^H A - F P F^H  \|_2^2

    where A is the Sense operator,
    and F is a unitary Fourier transform operator.

    Args:
        mps (array): sensitivity maps of shape [num_coils] + image shape.
        weights (array): k-space weights.
        coord (array): k-space coordinates of shape [...] + [ndim].
        lamda (float): regularization.

    Returns:
        array: circulant preconditioner of image shape.

    """
    if coord is not None:
        coord = sp.to_device(coord, device)

    if weights is not None:
        weights = sp.to_device(weights, device)

    dtype = mps.dtype
    device = sp.Device(device)
    xp = device.xp

    mps_shape = list(mps.shape)
    img_shape = mps_shape[1:]
    img2_shape = [i * 2 for i in img_shape]
    ndim = len(img_shape)

    scale = sp.prod(img2_shape)**1.5 / sp.prod(img_shape)**2
    with device:
        idx = (slice(None, None, 2), ) * ndim
        if coord is None:
            ones = xp.zeros(img2_shape, dtype=dtype)
            if weights is None:
                ones[idx] = 1
            else:
                ones[idx] = weights**0.5

            psf = sp.ifft(ones)
        else:
            coord2 = coord * 2
            ones = xp.ones(coord.shape[:-1], dtype=dtype)
            if weights is not None:
                ones *= weights**0.5

            psf = sp.nufft_adjoint(ones, coord2, img2_shape)

        p_inv = 0
        for mps_i in mps:
            mps_i = sp.to_device(mps_i, device)
            xcorr_fourier = xp.abs(sp.fft(xp.conj(mps_i), img2_shape))**2
            xcorr = sp.ifft(xcorr_fourier)
            xcorr *= psf
            p_inv_i = sp.fft(xcorr)
            p_inv_i = p_inv_i[idx]
            p_inv_i *= scale
            if weights is not None:
                p_inv_i *= weights**0.5

            p_inv += p_inv_i

        p_inv += lamda
        p_inv[p_inv == 0] = 1
        p = 1 / p_inv

        return p.astype(dtype)
