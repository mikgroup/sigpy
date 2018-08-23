import numpy as np
import sigpy as sp


__all__ = ['Sense', 'SenseMultiply', 'SenseCombine', 'ConvImage', 'ConvSense']


def Sense(mps, coord=None, ishape=None):
    """Sense linear operator.
    
    Args:
        mps (array): sensitivity maps of length = number of channels.
        coord (None or array): coordinates.
    """

    img_ndim = mps.ndim - 1
    if ishape is None:
        ishape = mps.shape[1:]

    S = sp.linop.Multiply(ishape, mps)

    if coord is None:
        F = sp.linop.FFT(S.oshape, axes=range(-img_ndim, 0))
    else:
        F = sp.linop.NUFFT(S.oshape, coord)

    A = F * S
    A.repr_str = 'Sense'

    return A


def ConvSense(img_ker_shape, mps_ker, coord=None):
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
        grd_shape = [num_coils] + sp.nufft.estimate_shape(coord)
        iF = sp.linop.IFFT(grd_shape, axes=range(-ndim, 0))
        N = sp.linop.NUFFT(grd_shape, coord)
        A = N * iF * A

    return A


def ConvImage(mps_ker_shape, img_ker, coord=None):
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
        grd_shape = [num_coils] + sp.nufft.estimate_shape(coord)
        iF = sp.linop.IFFT(grd_shape, axes=range(-ndim, 0))
        N = sp.linop.NUFFT(grd_shape, coord)
        A = N * iF * A

    return A
