# -*- coding: utf-8 -*-
"""MRI linear operators.
"""
import sigpy as sp


def Sense(mps, coord=None, weights=None, ishape=None, coil_batch_size=None):
    """Sense linear operator.
    
    Args:
        mps (array): sensitivity maps of length = number of channels.
        coord (None or array): coordinates.

    """
    if ishape is None:
        ishape = mps.shape[1:]
        img_ndim = mps.ndim - 1
    else:
        img_ndim = len(ishape)

    num_coils = len(mps)
    if coil_batch_size is None:
        coil_batch_size = num_coils

    if coil_batch_size < len(mps):
        num_coil_batches = (num_coils + coil_batch_size - 1) // coil_batch_size
        return sp.linop.Vstack([Sense(mps[c::num_coil_batches], coord=coord, ishape=ishape)
                                for c in range(num_coil_batches)], axis=0)

    S = sp.linop.Multiply(ishape, mps)
    if coord is None:
        F = sp.linop.FFT(S.oshape, axes=range(-img_ndim, 0))
    else:
        F = sp.linop.NUFFT(S.oshape, coord)

    A = F * S
    
    if weights is not None:
        P = sp.linop.Multiply(F.oshape, weights**0.5)
        A = P * A
        
    A.repr_str = 'Sense'
    return A


def ConvSense(img_ker_shape, mps_ker, coord=None, weights=None):
    """Convolution linear operator with sensitivity maps kernel in k-space.
    
    Args:
        img_ker_shape (tuple of ints): image kernel shape.
        mps_ker (array): sensitivity maps kernel.
        coord (array): coordinates.

    """
    ndim = len(img_ker_shape)
    A = sp.linop.ConvolveInput(img_ker_shape, mps_ker, mode='valid', output_multi_channel=True)

    if coord is not None:
        num_coils = mps_ker.shape[0]
        grd_shape = [num_coils] + sp.estimate_shape(coord)
        iF = sp.linop.IFFT(grd_shape, axes=range(-ndim, 0))
        N = sp.linop.NUFFT(grd_shape, coord)
        A = N * iF * A
        
    if weights is not None:
        P = sp.linop.Multiply(A.oshape, weights**0.5)
        A = P * A

    return A


def ConvImage(mps_ker_shape, img_ker, coord=None, weights=None):
    """Convolution linear operator with image kernel in k-space.
    
    Args:
        mps_ker_shape (tuple of ints): sensitivity maps kernel shape.
        img_ker (array): image kernel.
        coord (array): coordinates.

    """
    ndim = img_ker.ndim

    A = sp.linop.ConvolveFilter(mps_ker_shape, img_ker, mode='valid', output_multi_channel=True)

    if coord is not None:
        num_coils = mps_ker_shape[0]
        grd_shape = [num_coils] + sp.estimate_shape(coord)
        iF = sp.linop.IFFT(grd_shape, axes=range(-ndim, 0))
        N = sp.linop.NUFFT(grd_shape, coord)
        A = N * iF * A
        
    if weights is not None:
        P = sp.linop.Multiply(A.oshape, weights**0.5)
        A = P * A

    return A
