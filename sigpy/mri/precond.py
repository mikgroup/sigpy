# -*- coding: utf-8 -*-
"""MRI preconditioners.
"""
import sigpy as sp


__all__ = ['kspace_precond', 'circulant_precond']


def kspace_precond(mps, weights=None, coord=None, lamda=0, device=sp.cpu_device):
    """Compute a diagonal preconditioner in k-space.

    Considers the optimization problem:
        p = argmin_p || diag(p) A A^H - I ||_2^2
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
    img_shape = list(mps_shape[1:])
    img2_shape = [i * 2 for i in img_shape]
    ndim = len(img_shape)
    num_coils = mps.shape[0]

    scale = sp.prod(img2_shape)**1.5 / sp.prod(img_shape)
    with device:
        if coord is None:
            idx = [slice(None, None, 2)] * ndim

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

        mps = sp.to_device(mps, device)
        mps_ij = mps * xp.conj(mps.reshape([num_coils, 1] + img_shape))
        xcorr_fourier = xp.sum(xp.abs(sp.fft(mps_ij, [num_coils, num_coils] + img2_shape, axes=range(-ndim, 0)))**2, axis=0)
        xcorr = sp.ifft(xcorr_fourier, axes=range(-ndim, 0))
        xcorr *= psf
        
        if coord is None:
            p_inv = sp.fft(xcorr, axes=range(-ndim, 0))[[slice(None)] + idx]
        else:
            p_inv = sp.nufft(xcorr, coord2)

        if weights is not None:
            p_inv *= weights**0.5

        mps_norm2 = sp.norm2(mps, axes=range(-ndim, 0)).reshape([num_coils] + [1] * (p_inv.ndim - 1))
        p_inv *= scale / mps_norm2
        p_inv += lamda
        p_inv /= 1 + lamda
        p_inv[p_inv == 0] = 1
        p = 1 / p_inv

        return p.astype(dtype)


def circulant_precond(mps, weights=None, coord=None, lamda=0, device=sp.cpu_device):
    """Compute circulant preconditioner.

    Considers the optimization problem:
        p = argmin_p || A^H A - F diag(p) F^H  ||_2^2
    where A is the Sense operator, and F is a unitary Fourier transform operator.

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
    num_coils = mps.shape[0]

    scale = sp.prod(img2_shape)**1.5 / sp.prod(img_shape)**2
    with device:
        idx = [slice(None, None, 2)] * ndim
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

        mps = sp.to_device(mps, device)
        xcorr_fourier = xp.abs(sp.fft(xp.conj(mps), [num_coils] + img2_shape, axes=range(-ndim, 0)))**2
        xcorr = sp.ifft(xcorr_fourier, axes=range(-ndim, 0))
        xcorr *= psf
        p_inv = sp.fft(xcorr, axes=range(-ndim, 0))
        p_inv = p_inv[[slice(None)] + idx]
        p_inv *= scale
        if weights is not None:
            p_inv *= weights**0.5

        p_inv = xp.sum(p_inv, axis=0)
        p_inv += lamda
        p_inv[p_inv == 0] = 1
        p = 1 / p_inv

        return p.astype(dtype)
