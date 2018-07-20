import numpy as np
import sigpy as sp
from sigpy.mri import util, sense


__all__ = ['Sense', 'SenseMultiply', 'SenseCombine', 'ConvImage', 'ConvSense']


def Sense(mps, coord=None):
    """Sense linear operator.
    
    Args:
        mps (array or sense.SenseMaps): sensitivity maps of length = number of channels.
        coord (None or array): coordinates.
    """

    ndim = mps.ndim - 1

    S = SenseMultiply(mps)

    if coord is None:
        F = sp.linop.FFT(S.oshape, axes=range(-ndim, 0))
    else:
        F = sp.linop.NUFFT(S.oshape, coord)

    A = F * S
    A.repr_str = 'Sense'

    return A


class SenseMultiply(sp.linop.Linop):
    """Sense multiply linear operator.
    
    Args:
        mps (array or sense.SenseMaps): sensitivity maps of length = number of channels.
    """

    def __init__(self, mps):
        self.mps = mps
        ishape = self.mps.shape[1:]
        oshape = self.mps.shape

        super().__init__(oshape, ishape)

    def _apply(self, input):
        device = sp.util.get_device(input)

        with device:
            if isinstance(self.mps, sense.SenseMaps):
                mps = self.mps.asarray()
            else:
                mps = self.mps

            return input * mps

    def _adjoint_linop(self):

        return SenseCombine(self.mps)


class SenseCombine(sp.linop.Linop):
    """Sense combine linear operator.
    
    Args:
        mps (array or sense.SenseMaps): sensitivity maps of length = number of channels.
    """

    def __init__(self, mps):
        self.mps = mps
        oshape = self.mps.shape[1:]
        ishape = self.mps.shape

        super().__init__(oshape, ishape)

    def _apply(self, input):
        device = sp.util.get_device(input)
        xp = device.xp

        with device:
            if isinstance(self.mps, sense.SenseMaps):
                mps = self.mps.asarray()
            else:
                mps = self.mps

            return xp.sum(input * xp.conj(mps), axis=0)

    def _adjoint_linop(self):

        return SenseMultiply(self.mps)


def ConvSense(img_ker_shape, mps_ker, coord=None):
    """Convolution linear operator with sensitivity maps kernel in k-space.
    
    Args:
        img_ker_shape (tuple of ints): image kernel shape.
        mps_ker (array): sensitivity maps kernel.
        coord (array): coordinates.
    """
    
    ndim = len(img_ker_shape)
    A = sp.linop.Convolve(
        img_ker_shape, mps_ker, axes=range(-ndim, 0), mode='valid')

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

    A = sp.linop.Convolve(
        mps_ker_shape, img_ker, axes=range(-ndim, 0), mode='valid')

    if coord is not None:
        num_coils = mps_ker_shape[0]
        grd_shape = [num_coils] + sp.nufft.estimate_shape(coord)
        iF = sp.linop.IFFT(grd_shape, axes=range(-ndim, 0))
        N = sp.linop.NUFFT(grd_shape, coord)
        A = N * iF * A

    return A
