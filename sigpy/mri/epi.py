# -*- coding: utf-8 -*-
"""Methods for Echo-Planar Imaging (EPI) acquisition
with an focus on diffusion tensor/kurtosis imaging.

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""
import numpy as np
from sigpy import fourier

MIN_POSITIVE_SIGNAL = 0.0001


def phase_corr(kdat, pcor, topup_dim=-11):
    """perform phase correction.

    Args:
        kdat (array): k-space data to be corrected
        pcor (array): phase-correction reference data
        topup_dim (int): dimension of the top-up and top-down measurements

    Output:
        phase-corrected k-space data

    Reference:
        Ehses P. https://github.com/pehses/twixtools
    """
    col_dim = -1

    ncol = pcor.shape[col_dim]
    npcl = pcor.shape[topup_dim]

    # if three lines are present,
    # average the 1st and the 3rd line.
    if npcl == 3:
        pcors = np.swapaxes(pcor, 0, topup_dim)
        pcor1 = pcors[[0, 2], ...]
        pcor_odd = np.mean(pcor1, axis=0, keepdims=True)
        pcor_eve = pcors[[1], ...]
        pcor = np.concatenate((pcor_odd, pcor_eve))
        pcor = np.swapaxes(pcor, 0, topup_dim)

    oshape = list(kdat.shape)
    oshape[topup_dim] = 1

    output = np.zeros(oshape)

    pcor_img = fourier.ifft(pcor, axes=[col_dim])
    kdat_img = fourier.ifft(kdat, axes=[col_dim])

    slope = np.angle((np.conj(pcor_img[..., 1:]) * pcor_img[..., :-1])
                     .sum(col_dim, keepdims=True).sum(-2, keepdims=True))
    x = np.arange(ncol) - ncol//2

    pcor_fac = np.exp(1j * slope * x)

    kdat_img *= pcor_fac
    kdat_img = kdat_img.sum(topup_dim, keepdims=True)
    output = fourier.fft(kdat_img, axes=[-1])

    return output


def get_B(b, g):
    """Compute B matrix from b values and g vectors

    Args:
        b (1D array): b values
        g (2D array): g vectors

    Output:
        B (array): [gx**2, gy**2, gz**2,
                    2*gx*gy, 2*gx*gz, 2*gy*gz] of every pixel
    """
    num_g, num_axis = g.shape

    assert num_axis == 3
    assert num_g == len(b)

    gx = g[:, 0]
    gy = g[:, 1]
    gz = g[:, 2]

    return - b * np.array([gx**2, 2*gx*gy, gy**2,
                           2*gx*gz, 2*gy*gz, gz**2]).transpose()


def get_B2(b, g):
    """For Diffusion Kurtosis:
        Compute B2 matrix from b values and g vectors

    Args:
        b (1D array): b values
        g (2D array): g vectors

    Output:
        B (array)
    """
    num_g, num_axis = g.shape

    assert num_axis == 3
    assert num_g == len(b)

    gx = g[:, 0]
    gy = g[:, 1]
    gz = g[:, 2]

    BT = get_B(b, g)

    BK = b * b * np.array([
                     gx**4 / 6,
                     gy**4 / 6,
                     gz**4 / 6,
                     4 * gx**3 * gy / 6,
                     4 * gx**3 * gz / 6,
                     4 * gy**3 * gx / 6,
                     4 * gy**3 * gz / 6,
                     4 * gz**3 * gx / 6,
                     4 * gz**3 * gy / 6,
                     gx**2 * gy**2,
                     gx**2 * gz**2,
                     gy**2 * gz**2,
                     2 * gx**2 * gy * gz,
                     2 * gy**2 * gx * gz,
                     2 * gz**2 * gx * gy]).transpose()

    return np.concatenate((BT, BK), axis=1)


def get_D(B, sig, fit_method='wls', fit_only_tensor=False,
          min_signal=0, fit_kt=False):
    """Compute D matrix (diffusion tensor)

    Args:
        B (array): see above.
        sig (array): b0 image and diffusion-weighted images.
        fit_method (string): [default: 'wls']
            - 'wls' weighted least square
            - 'ols' ordinary least square
        fit_only_tensor (boolean): excluding b0 [default: False]
        min_signal (float): minimal signal intensity in DWI
            [Default: MIN_POSITIVE_SIGNAL]
            better set to 0.
        fit_kt (boolean): fit kurtosis tensor directly [default: False]

    Output:
        D (array): [Dxx, Dxy, Dyy, Dxz, Dyz, Dzz] of every pixel.
        Please refer to get_B() and get_B2() for the actual order
        of the D array.

    References:
        Chung S. W., Lu Y., Henry R. G. (2006).
        Comparison of bootstrap approaches for
        estimation of uncertainties of DTI parameters.
        NeuroImage 33, 531-541.

        DiPy. https://github.com/dipy/dipy
    """
    sig = np.abs(sig)
    sig = np.maximum(sig, min_signal)
    S = np.log(sig, out=np.zeros_like(sig), where=(sig != 0))

    ndiff = S.shape[0]
    image_shape = S.shape[1:]

    if fit_only_tensor is True:
        y = S[0, ...] - S
    else:
        y = S
        dummy = np.ones((B.shape[0], 1))
        B = np.concatenate((B, dummy), axis=1)

    nparam = B.shape[1]
    yr = y.reshape(ndiff, -1)

    # print('> OLS Fitting')
    xr = np.dot(np.linalg.pinv(B), yr)
    D_fit = xr.reshape([nparam] + list(image_shape))

    if fit_method == 'wls':
        # print('> WLS Fitting')

        if fit_kt is True:
            eigvals, eigvecs = get_eig(D_fit, B)
            MD2 = get_MD(eigvals)**2
            scale = np.tile(MD2[None, ...], [nparam] + [1] * len(image_shape))
            scale = np.reshape(scale, (nparam, -1))
            scale[:6, ...] = 1
            scale[-1, ...] = 1
        else:
            scale = np.ones_like(xr)

        scale = np.expand_dims(scale.T, axis=1)

        w = np.exp(np.dot(B, xr)).T  # weight

        lhs = np.linalg.pinv(B * w[..., None] * scale, rcond=1e-15)
        lhs = np.swapaxes(lhs, 0, 1)

        rhs = (w.T * yr).T

        xr = np.sum(lhs * rhs, axis=-1)

    return xr.reshape([nparam] + list(image_shape))


_lt_indices = np.array([[0, 1, 3],
                        [1, 2, 4],
                        [3, 4, 5]])


def DT_vec2mat(Dvec):
    """Convert the 6 elements of diffusion tensor (DT)
    to a 3x3 symmetric matrix
    """
    assert 6 == Dvec.shape[0]

    return Dvec[_lt_indices, ...]


def get_eig(D, B=None):
    """Compute eigenvalues and eigenvectors of the D matrix

    Args:
        D (array): output from get_D(B, sig)

    Output:
        eigvals: eigenvalues
        eigvecs: eigenvectors
    """
    image_shape = D.shape[1:]
    image_size = np.prod(image_shape)

    Dmat = DT_vec2mat(D[:6, ...])
    temp = np.rollaxis(Dmat, 0, len(Dmat.shape))
    Dmat = np.rollaxis(temp, 0, len(Dmat.shape))
    eigvals, eigvecs = np.linalg.eigh(Dmat)

    # flatten eigvals and eigenvecs
    eigvals = eigvals.reshape(-1, 3)
    eigvecs = eigvecs.reshape(-1, 3, 3)

    order = eigvals.argsort()[:, ::-1]

    xi = np.ogrid[:image_size, :3][0]
    eigvals = eigvals[xi, order]

    xi, yi = np.ogrid[:image_size, :3, :3][:2]
    eigvecs = eigvecs[xi, yi, order[:, None, :]]

    eigvals = eigvals.reshape(image_shape + (3, ))
    eigvecs = eigvecs.reshape(image_shape + (3, 3))

    eigvals = np.rollaxis(eigvals, -1, 0)

    eigvecs = np.rollaxis(eigvecs, -1, 0)
    eigvecs = np.rollaxis(eigvecs, -1, 0)

    if B is not None:
        min_diffusivity = 1e-6 / -B.min()
        eigvals = eigvals.clip(min=min_diffusivity)

    return eigvals, eigvecs


def get_FA(eigvals):
    """Compute Fractional Anisotropy (FA) map

    Args:
        eigvals (array): output from get_eig(D)

    Output:
        FA (array): FA map
    """
    l1 = eigvals[0, ...]
    l2 = eigvals[1, ...]
    l3 = eigvals[2, ...]

    nomi = 0.5 * ((l1-l2)**2 + (l2-l3)**2 + (l3-l1)**2)
    deno = l1**2 + l2**2 + l3**2

    FA = np.sqrt(np.divide(nomi, deno,
                           out=np.zeros_like(nomi),
                           where=deno != 0))

    return FA


def get_cFA(FA, eigvecs):
    """Compute color-coded Fractional Anisotropy (cFA) map

    Args:
        FA (array): FA map
        eigvecs (array): eigen vectors

    Output:
        cFA (array): cFA map
    """
    return np.abs(eigvecs[:, 0, ...]) * FA


def get_MD(eigvals):
    """Compute Mean Diffusivity (MD) map

    Args:
        eigvals (array): output from get_eig(D)

    Output:
        MD (array): MD map
    """
    assert 3 == eigvals.shape[0]

    return np.mean(eigvals, axis=0)


def get_KT(D, B=None):
    """Compute Kurtosis Tensor (KT) map

    Args:
        D (array): output from get_D(B, sig)

    Output:
        KT (array): KT map
    """
    assert 21 <= D.shape[0]
    DT = D[:6, ...]
    DK = D[6:21, ...]

    eigvals, eigvecs = get_eig(DT, B=B)
    MD = get_MD(eigvals)

    return DK / (MD**2)
