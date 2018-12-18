import numpy as np
import sigpy as sp
if sp.config.cupy_enabled:
    import cupy as cp


class FftSuite:
    if sp.config.cupy_enabled:
        params = [np, cp]
    else:
        params = [np]
        
    def setup(self, xp):
        self.x = xp.random.randn(100)

    def time_fft(self, xp):
        y = sp.fft(self.x)

    def time_fft_non_centered(self, xp):
        y = sp.fft(self.x, center=False)

    def time_ifft(self, xp):
        y = sp.ifft(self.x)

    def time_ifft_non_centered(self, xp):
        y = sp.ifft(self.x, center=False)

        
class NufftSuite:
    if sp.config.cupy_enabled:
        params = [np, cp]
    else:
        params = [np]
    
    def setup(self, xp):
        self.x = xp.random.randn(100)
        self.coord = xp.random.randn(100, 1)

    def time_nufft(self, xp):
        y = sp.nufft(self.x, self.coord)

    def time_nufft_adjoint(self, xp):
        y = sp.nufft_adjoint(self.x, self.coord)


class ThreshSuite:
    if sp.config.cupy_enabled:
        params = [np, cp]
    else:
        params = [np]
    
    def setup(self, xp):
        self.x = xp.random.randn(100)

    def time_soft_thresh(self, xp):
        y = sp.soft_thresh(1e-3, self.x)

    def time_hard_thresh(self, xp):
        y = sp.hard_thresh(1e-3, self.x)

    def time_elitist_thresh(self, xp):
        y = sp.elitist_thresh(1e-3, self.x)

    def time_l1_proj(self, xp):
        y = sp.l1_proj(1, self.x)


class AlgSuite:
    
    def setup(self):
        x = np.random.randn(100)
        A = np.random.randn(100, 100)
        self.AHA = A.T @ A
        self.AHy = self.AHA @ x
        self.x = np.zeros(100)

    def time_ConjugateGradient(self):
        alg = sp.alg.ConjugateGradient(lambda x: self.AHA @ x, self.AHy, self.x, max_iter=100)
        while not alg.done():
            alg.update()

    def time_GradientMethod(self):
        alg = sp.alg.GradientMethod(lambda x: self.AHA @ x - self.AHy, self.x,
                                    alpha=1e-3, max_iter=100)
        while not alg.done():
            alg.update()

    def time_AcceleratedGradientMethod(self):
        alg = sp.alg.GradientMethod(lambda x: self.AHA @ x - self.AHy, self.x,
                                    alpha=1e-3, max_iter=100, accelerate=True)
        while not alg.done():
            alg.update()
