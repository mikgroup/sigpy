import numpy as np
import sigpy as sp
from sigpy.mri import util, sense


class SenseMultiply(sp.linop.Linop):

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


def Sense(mps, coord=None):

    ndim = mps.ndim - 1

    S = SenseMultiply(mps)

    if coord is None:
        F = sp.linop.FFT(S.oshape, axes=range(-ndim, 0))
    else:
        F = sp.linop.NUFFT(S.oshape, coord)

    A = F * S
    A.repr_str = 'Sense'

    return A


def ConvSense(img_ker_shape, mps_ker, coord=None):
    ndim = len(img_ker_shape)
    A = sp.linop.Convolve(
        img_ker_shape, mps_ker, axes=range(-ndim, 0), mode='valid')

    if coord is not None:
        num_coils = mps_ker.shape[0]
        grd_shape = [num_coils] + util.estimate_img_shape(coord)
        iF = sp.linop.IFFT(grd_shape, axes=range(-ndim, 0))
        N = sp.linop.NUFFT(grd_shape, coord)
        A = N * iF * A

    return A


def ConvImage(mps_ker_shape, img_ker, coord=None):
    ndim = img_ker.ndim

    A = sp.linop.Convolve(
        mps_ker_shape, img_ker, axes=range(-ndim, 0), mode='valid')

    if coord is not None:
        num_coils = mps_ker_shape[0]
        grd_shape = [num_coils] + util.estimate_img_shape(coord)
        iF = sp.linop.IFFT(grd_shape, axes=range(-ndim, 0))
        N = sp.linop.NUFFT(grd_shape, coord)
        A = N * iF * A

    return A
