import numpy as np
import logging
import sigpy as sp


def sense_kspace_precond(mps, weights=1, coord=None, lamda=0, device=sp.util.cpu_device):
    """Compute L2 optimized Sense diagonal preconditioner in Fourier domain.

    Considers the optimization problem:
        p = argmin_p 1 / 2 || diag(p) W F S S^H F^H W^H - I ||_2^2
    where W is the weighting operator,
    F is the Fourier operator, and S is the sensitivity operator.

    Args:
        mps (array) - sensitivity maps of shape [num_coils] + image shape.
        weights (array) - k-space weights.
        coord (array) - k-space coordinates of shape [...] + [ndim].
        lamda (float) - regularization.

    Returns:
        precond (array) - k-space preconditioner of same shape as k-space.
    """
    dtype = mps.dtype
    mps = sp.util.move(mps, device)

    if coord is not None:
        coord = sp.util.move(coord, device)

    if not np.isscalar(weights):
        weights = sp.util.move(weights, device)

    device = sp.util.Device(device)
    xp = device.xp
    logger = logging.getLogger(__name__)

    mps_shape = list(mps.shape)
    img_shape = mps_shape[1:]
    img2_shape = [i * 2 for i in img_shape]
    ndim = len(img_shape)
    num_coils = mps.shape[0]

    scale = sp.util.prod(img2_shape)**1.5 / sp.util.prod(img_shape)

    with device:
        logger.debug('Getting 2x over-sampled point spread function.')
        if coord is None:
            slc = [slice(None, None, 2)] * ndim

            ones = sp.util.zeros(img2_shape, dtype=dtype, device=device)
            ones[slc] = weights**0.5

            psf = sp.fft.ifft(ones)
        else:
            coord2 = coord * 2
            ones = sp.util.ones(coord.shape[:-1], dtype=dtype, device=device)
            ones *= weights**0.5

            psf = sp.nufft.nufft_adjoint(ones, coord2, img2_shape)

        logger.debug('Getting cross-correlation.')
        density = []
        for mps_i in mps:
            mps_i_norm2 = sp.util.norm2(mps_i)
            xcorr_fourier = 0
            for mps_j in mps:
                xcorr_fourier += xp.abs(sp.fft.fft(mps_i *
                                                   xp.conj(mps_j), img2_shape))**2

            xcorr = sp.fft.ifft(xcorr_fourier)
            del xcorr_fourier
            xcorr *= psf
            if coord is None:
                density_i = sp.fft.fft(xcorr)[slc]
            else:
                density_i = sp.nufft.nufft(xcorr, coord2)

            density_i *= weights**0.5
            density.append(density_i * scale / mps_i_norm2)

        density = (xp.abs(xp.stack(density, axis=0)) + lamda) / (1 + lamda)
        density[density == 0] = 1
        precond = 1 / density

        return precond.astype(dtype)
