# -*- coding: utf-8 -*-
"""Functions for GRAPPA.

This implementation makes use of the function "sp.array_to_blocks" to slide through calibration as well as undersampled k-space data to perform GRAPPA kernel fitting and reconstruction.

Alternatively, one can also use other functions, e.g. "view_as_windows" in skimage.util.

Reference:
    https://users.fmrib.ox.ac.uk/~mchiew/Tools.html
    https://github.com/mckib2/pygrappa

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""
import numpy as np
import sigpy as sp

__all__ = ['Grappa', 'SliceGrappa']


def _set_kspace_boundary(kspace, pad_shape):

    py, px = pad_shape
    kspace[..., :px] = kspace[..., -px*2:-px]
    kspace[..., -px:] = kspace[..., px:2*px]
    kspace[..., :py, :] = kspace[..., -2*py:-py, :]
    kspace[..., -py:, :] = kspace[..., py:2*py, :]

    return kspace


def Grappa(kspace, calib, R=[2, 1], lamda=1E-4,
           kernel_shape=[4, 5],
           kernel_stride=[1, 1]):
    """
    Args:
        kspace (array): undersampled 2D k-space data.
        calib (array): fully-sampled calibration data.
        R (scalar or length-2 list): acceleration factor.
        lamda (float): weight fitting regularization.
        kernel_shape (length-2 list): [ker_y (even), ker_x (odd)].
        kernel_stride (length-2 list): kernel stride.

    Reference:
        Griswold M. A., Jakob P. M., Heidemann R. M., Nittka M., Jellus V., Wang J., Kiefer B., Haase A. (2002).
        Generalized autocalibrating partially parallel acquisitions (GRAPPA).
        Magn. Reson. Med., 47, 1202-1210.
    """
    acc_y, acc_x = R[:]
    assert(acc_x == 1) # 1D Grappa with only ky undersampling

    NC, NY, NX = kspace.shape
    _C, _Y, _X = calib.shape

    assert(NC == _C)

    ker_y, ker_x = kernel_shape[:]
    assert((ker_y % 2 == 0) and (ker_x % 2 == 1))

    ker_acc_y, ker_acc_x = acc_y * ker_y, acc_x * ker_x
    pad_y, pad_x = ker_acc_y // 2 - 1, ker_acc_x // 2

    # %% train Grappa kernel weights

    None

def SliceGrappa(kspace, calib, R=2, lamda=1E-4,
                kernel_shape=[5, 5],
                kernel_stride=[1, 1]):
    """
    Args:
        kspace (array): undersampled collapsed SMS k-space data.
        calib (array): fully-sampled multi-slice calibration data.
        lamda (float): weight fitting regularization.
        kernel_shape (length-2 list): [ker_y, ker_x] - both must be odd.
        kernel_stride (length-2 list): kernel stride.

    Reference:
        Setsompop K., Gagoski B. A., Polimeni J. R., Witzel T., Wedeen V. J., Wald L. L. (2012).
        Blipped-controlled aliasing in parallel imaging for simultaneous multislice echo planar imaging with reduced g-factor penalty.
        Magn. Reson. Med., 67, 1210-1224.
    """
    NC, NY, NX = kspace.shape
    NS, _NC, _NY, _NX = calib.shape

    assert(NC == _NC)

    ker_y, ker_x = kernel_shape[:]  # kernel y and x
    pad_y, pad_x = ker_y//2, ker_x//2  # pad y and x

    kernel_len = np.prod(kernel_shape)
    weight_len = kernel_len * NC

    # pad kspace and calib
    kshape = kspace.shape
    kpad_shape = list(kshape[:-2]) + \
        [NY + 2*pad_y] + [NX + 2*pad_x]

    kspace_pad = sp.util.resize(kspace, kpad_shape)
    kspace_pad = _set_kspace_boundary(kspace_pad, [pad_y, pad_x])

    cshape = calib.shape
    cpad_shape = list(cshape[:-2]) + \
        [_NY - 2*pad_y] + [_NX - 2*pad_x]
    calib_pad = sp.util.resize(calib, cpad_shape)


    # %% Train kernels slice by slice

    # construct source blocks from calib data
    A2B = sp.linop.ArrayToBlocks(cshape, kernel_shape, kernel_stride)
    src = A2B(calib)
    # src = np.sum(src, axis=0)  # sum over the slice axis
    src = np.transpose(src, (0, 2, 3, 1, 4, 5))

    assert(np.prod(src.shape[-3:]) == weight_len)

    src = np.reshape(src, (NS, -1, weight_len))
    src = np.sum(src, axis=0)

    # construct target blocks
    A2B = sp.linop.ArrayToBlocks(cpad_shape, [1,1], [1,1])
    trg = A2B(calib_pad)
    trg = np.reshape(trg, (NS, NC, np.prod(trg.shape[-4:])))

    W = np.zeros_like(calib, shape=(NS, NC, weight_len))

    for s in range(NS):
        x1 = src.T
        y1 = trg[s, ...]

        SHS = x1 @ x1.T.conj() + lamda * np.linalg.norm(x1) * np.eye(weight_len)
        x2 = x1.T.conj() @ np.linalg.pinv(SHS)

        W[s, ...] = y1 @ x2

    # %% apply trained weights to un-collapse kspace data

    # construct source blocks from kspace data
    A2B = sp.linop.ArrayToBlocks(kpad_shape, kernel_shape, kernel_stride)
    src = A2B(kspace_pad)
    src = np.transpose(src, (1, 2, 0, 3 ,4))
    src = np.reshape(src, (-1, weight_len)).T

    res = np.zeros_like(kspace, shape=(NS, NC, NY, NX))
    for s in range(NS):
        res[s, ...] = (W[s, ...] @ src).reshape((-1, NY, NX))

    return res
