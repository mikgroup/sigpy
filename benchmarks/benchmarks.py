import numpy as np
import sigpy as sp


class FftSuite:
        
    def setup(self):
        self.x = np.random.randn(100)

    def time_fft(self):
        y = sp.fft(self.x)

    def time_fft_non_centered(self):
        y = sp.fft(self.x, center=False)

    def time_ifft(self):
        y = sp.ifft(self.x)

    def time_ifft_non_centered(self):
        y = sp.ifft(self.x, center=False)

        
class NufftSuite:
    
    def setup(self):
        self.x = np.random.randn(100)
        self.coord = np.random.randn(100, 1)

    def time_nufft(self):
        y = sp.nufft(self.x, self.coord)

    def time_nufft_adjoint(self):
        y = sp.nufft_adjoint(self.x, self.coord)


class ThreshSuite:
    
    def setup(self):
        self.x = np.random.randn(100)

    def time_soft_thresh(self):
        y = sp.soft_thresh(1e-3, self.x)

    def time_hard_thresh(self):
        y = sp.hard_thresh(1e-3, self.x)

    def time_l1_proj(self):
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
