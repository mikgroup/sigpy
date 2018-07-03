import os
import logging
import itertools
import numpy as np
import sigpy as sp

from scipy.signal import triang
from sigpy.util import prod
from sigpy.mri import linop, precond, util, sense


if sp.config.mpi4py_enabled:
    from mpi4py import MPI


def estimate_weights(ksp, weights, coord):
    if np.isscalar(weights) and coord is None:
        with sp.util.get_device(ksp):
            weights = sp.util.rss(ksp, axes=(0, )) > 0
    return weights


def move_to_device(ksp, mps, weights, coord, device):
    ksp = sp.util.move(ksp, device=device)
    
    if isinstance(mps, sense.SenseMaps):
        mps.use_device(device)
    else:
        mps = sp.util.move(mps, device=device)

    if not np.isscalar(weights):
        weights = sp.util.move(weights, device=device)
        
    if coord is not None:
        coord = sp.util.move(coord, device=device)

    return ksp, mps, weights, coord


class SenseRecon(sp.app.LinearLeastSquares):
    '''
    Sense Reconstruction
    min_1 / 2|| P F S x - y ||_2^2 + lamda / 2 || x ||_2^2
    '''
    def __init__(
            self, ksp, mps,
            lamda=0, weights=1, coord=None, device=sp.util.cpu_device, **kwargs):
        ksp, mps, weights, coord = move_to_device(ksp, mps, weights, coord, device)

        weights = estimate_weights(ksp, weights, coord)
        A = linop.Sense(mps, coord=coord)
        self.img = sp.util.zeros(mps.shape[1:], dtype=ksp.dtype, device=device)

        super().__init__(A, ksp, self.img, lamda=lamda, weights=weights, **kwargs)


class SenseConstrainedRecon(sp.app.SecondOrderConeConstraint):
    '''
    Sense Constrained Reconstruction
    min || x ||_2^2
    s.t. || P F S x - y ||_2 <= eps
    '''

    def __init__(
            self, ksp, mps, eps,
            weights=1, coord=None, device=sp.util.cpu_device, **kwargs):
        ksp, mps, weights, coord = move_to_device(ksp, mps, weights, coord, device)
        weights = estimate_weights(ksp, weights, coord)
        
        A = linop.Sense(mps, coord=coord)
        proxg = sp.prox.L2Reg(A.ishape, 1)
        self.img = sp.util.zeros(mps.shape[1:], dtype=ksp.dtype, device=device)

        super().__init__(A, ksp, self.img, proxg, eps, weights=weights, **kwargs)


class WaveletRecon(sp.app.LinearLeastSquares):
    '''
    l1 wavelet Reconstruction
    min_x 1 / 2|| P F S W^H x - y ||_2^2 + lamda || x ||_1
    '''

    def __init__(
            self, ksp, mps, lamda,
            weights=1, coord=None, wave_name='db4', device=sp.util.cpu_device, **kwargs):
        ksp, mps, weights, coord = move_to_device(ksp, mps, weights, coord, device)
        weights = estimate_weights(ksp, weights, coord)
        
        A = linop.Sense(mps, coord=coord)
        self.img = sp.util.zeros(mps.shape[1:], dtype=ksp.dtype, device=device)

        W = sp.linop.Wavelet(A.ishape, wave_name=wave_name)
        proxg = sp.prox.L1Reg(A.ishape, lamda, transform=W)

        def g(input):
            device = sp.util.get_device(input)
            xp = device.xp
            
            with device:
                return lamda * xp.sum(abs(W(input)))

        super().__init__(A, ksp, self.img, proxg=proxg, g=g, weights=weights, **kwargs)


class WaveletConstrainedRecon(sp.app.SecondOrderConeConstraint):
    '''
    Wavelet Constrained Reconstruction
    min || x ||_1
    s.t. || A x - y ||_2 <= eps
    '''
    def __init__(
            self, ksp, mps, eps,
            wave_name='db4', weights=1, coord=None, device=sp.util.cpu_device, **kwargs):
        ksp, mps, weights, coord = move_to_device(ksp, mps, weights, coord, device)
        weights = estimate_weights(ksp, weights, coord)
        
        A = linop.Sense(mps, coord=coord)
        self.img = sp.util.zeros(mps.shape[1:], dtype=ksp.dtype, device=device)
        W = sp.linop.Wavelet(A.ishape, wave_name=wave_name)
        proxg = sp.prox.L1Reg(A.ishape, 1, transform=W)

        super().__init__(A, ksp, self.img, proxg, eps, weights=weights, **kwargs)


class TotalVariationRecon(sp.app.LinearLeastSquares):
    '''
    l1 wavelet Reconstruction
    min_x 1 / 2|| A x - y ||_2^2 + lamda || G x ||_1
    '''
    def __init__(self, ksp, mps, lamda,
                 weights=1, coord=None, device=sp.util.cpu_device, **kwargs):
        ksp, mps, weights, coord = move_to_device(ksp, mps, weights, coord, device)
        weights = estimate_weights(ksp, weights, coord)
        
        A = linop.Sense(mps, coord=coord)
        self.img = sp.util.zeros(mps.shape[1:], dtype=ksp.dtype, device=device)

        G = sp.linop.Gradient(A.ishape)
        proxg = sp.prox.L1Reg(G.oshape, lamda)

        def g(input):
            xp = device.xp
            
            with device:
                return lamda * xp.sum(abs(G(input)))

        super().__init__(A, ksp, self.img, proxg=proxg, g=g, G=G, weights=weights, **kwargs)


class TotalVariationConstrainedRecon(sp.app.SecondOrderConeConstraint):
    '''
    TotalVariation Constrained Reconstruction
    min || G x ||_1
    s.t. || A x - y ||_2 <= eps
    '''

    def __init__(
            self, ksp, mps, eps,
            weights=1, coord=None, device=sp.util.cpu_device, **kwargs):
        ksp, mps, weights, coord = move_to_device(ksp, mps, weights, coord, device)
        weights = estimate_weights(ksp, weights, coord)
        
        A = linop.Sense(mps, coord=coord)
        self.img = sp.util.zeros(mps.shape[1:], dtype=ksp.dtype, device=device)
        G = sp.linop.Gradient(A.ishape)
        proxg = sp.prox.L1Reg(G.oshape, 1)

        super().__init__(A, ksp, self.img, proxg, eps, G=G, weights=weights, **kwargs)


class JsenseRecon(sp.app.App):
    '''
    Joint Sense Reconstruction
    min 1 / 2 || A(l, r) - y ||_2^2 + lamda / 2 (||l||_2^2 + ||r||_2^2)
    '''

    def __init__(
            self, ksp,
            mps_ker_width=12, ksp_calib_width=24, lamda=0, device=-1,
            weights=1, coord=None, max_iter=5, max_inner_iter=5, thresh=0):

        self.ksp = ksp
        self.mps_ker_width = mps_ker_width
        self.ksp_calib_width = ksp_calib_width
        self.lamda = lamda
        self.weights = weights
        self.coord = coord
        self.max_iter = max_iter
        self.max_inner_iter = max_inner_iter
        self.thresh = thresh
        
        self.device = sp.util.Device(device)
        self.dtype = ksp.dtype
        self.num_coils = len(ksp)

        self._init_data()
        self._init_vars()
        self._init_model()
        self._init_alg()
        
    def _init_data(self):
        if self.coord is None:
            self.img_shape = self.ksp.shape[1:]
            ndim = len(self.img_shape)

            self.ksp = sp.util.resize(self.ksp, [self.num_coils] + ndim * [self.ksp_calib_width])

            if not np.isscalar(self.weights):
                self.weights = sp.util.resize(self.weights, ndim * [self.ksp_calib_width])

        else:
            self.img_shape = util.estimate_img_shape(self.coord)
            calib_idx = np.amax(np.abs(self.coord), axis=-1) < self.ksp_calib_width / 2
            
            self.coord = self.coord[calib_idx]
            self.ksp = self.ksp[:, calib_idx]

            if not np.isscalar(self.weights):
                self.weights = self.weights[calib_idx]

        self.ksp = self.ksp / np.abs(self.ksp).max()
        self.ksp = sp.util.move(self.ksp, self.device)
        if self.coord is not None:
            self.coord = sp.util.move(self.coord, self.device)
        if not np.isscalar(self.weights):
            self.weights = sp.util.move(self.weights, self.device)
            
        self.weights = estimate_weights(self.ksp, self.weights, self.coord)

    def _init_vars(self):
        ndim = len(self.img_shape)
        
        mps_ker_shape = [self.num_coils] + [self.mps_ker_width] * ndim
        if self.coord is None:
            img_ker_shape = [i + self.mps_ker_width - 1 for i in self.ksp.shape[1:]]
        else:
            grd_shape = util.estimate_img_shape(self.coord)
            img_ker_shape = [i + self.mps_ker_width - 1 for i in grd_shape]

        self.img_ker = sp.util.dirac(img_ker_shape, dtype=self.dtype, device=self.device)
        self.mps_ker = sp.util.zeros(mps_ker_shape, dtype=self.dtype, device=self.device)

    def _init_model(self):
        self.A_img_ker = linop.ConvSense(self.img_ker.shape, self.mps_ker, coord=self.coord)
        
        self.A_mps_ker = linop.ConvImage(self.mps_ker.shape, self.img_ker, coord=self.coord)

    def _init_alg(self):
        self.app_mps = sp.app.LinearLeastSquares(
            self.A_mps_ker, self.ksp, self.mps_ker, weights=self.weights,
            lamda=self.lamda, max_iter=self.max_inner_iter)

        self.app_img = sp.app.LinearLeastSquares(
            self.A_img_ker, self.ksp, self.img_ker, weights=self.weights,
            lamda=self.lamda, max_iter=self.max_inner_iter)

        _alg = sp.alg.AltMin(self.app_mps.run, self.app_img.run, max_iter=self.max_iter)

        super().__init__(_alg)

    def _output(self):
        xp = self.device.xp
        # Coil by coil to save memory
        with self.device:
            mps_rss = 0
            for mps_ker_c in self.mps_ker:
                mps_c = sp.fft.ifft(sp.util.resize(mps_ker_c, self.img_shape))
                mps_rss += xp.abs(mps_c)**2
                
            mps_rss = mps_rss**0.5

            img = xp.abs(sp.fft.ifft(sp.util.resize(self.img_ker, self.img_shape)))
            img *= mps_rss

            img_weights = 1 / mps_rss
            img_weights *= img > self.thresh * img.max()

        return sense.SenseMaps(self.mps_ker, img_weights)
