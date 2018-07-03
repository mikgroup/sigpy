import pickle
import numpy as np
import sigpy as sp
from sigpy.util import prod


class SenseMaps(object):

    def __init__(self, mps_ker, img_mask, device=sp.util.cpu_device):
        self.num_coils = len(mps_ker)
        self.shape = (self.num_coils, ) + img_mask.shape
        self.ndim = len(self.shape)
        self.mps_ker = mps_ker
        self.img_mask = img_mask
        self.use_device(device)
        self.dtype = self.mps_ker.dtype

    def use_device(self, device):
        self.device = sp.util.Device(device)
        self.mps_ker = sp.util.move(self.mps_ker, device)
        self.img_mask = sp.util.move(self.img_mask, device)
        
    def __getitem__(self, slc):

        with self.device:
            if isinstance(slc, int):
                mps_c = sp.fft.ifft(self.mps_ker[slc], oshape=self.img_mask.shape)
                mps_c *= self.img_mask
                return mps_c

            elif isinstance(slc, slice):
                return SenseMaps(self.mps_ker[slc], self.img_mask, device=self.device)

            elif isinstance(slc, tuple) or isinstance(slc, list):
                if isinstance(slc[0], int):
                    mps = sp.fft.ifft(self.mps_ker[slc[0]], oshape=self.img_mask.shape)
                    mps *= self.img_mask
                    return mps[slc[1:]]

    def asarray(self):
        ndim = self.img_mask.ndim
        with self.device:
            mps = sp.fft.ifft(self.mps_ker, oshape=self.shape, axes=range(-ndim, 0))
            mps *= self.img_mask
            return mps

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def save(self, filename):
        self.use_device(sp.util.cpu_device)
        with open(filename, "wb") as f:
            pickle.dump(self, f)
