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
