"""The core module contains functions and classes for building iterative signal reconstruction applications.

SigPy provides simple interfaces to commonly used signal processing functions, including convolution, FFT, NUFFT, wavelet transform, and thresholding functions. All functions, except wavelet transform, can run on both CPU and GPU.

These functions are wrapped in two higher level classes to better interface with iterative methods: Linop, which abstracts linear operator, and Prox, which abstracts proximal operator. SigPy provides an abstraction class Alg for iterative algorithms, and implements commonly used methods, including conjugate gradient, (accelerated/proximal) gradient method, and primal dual hybrid gradient.

These classes can then be used to build an App as a final deliverable. An App simplifies the usage of Alg, and provides convenient features such as iteration progress bars. A particularly useful App implemented is the LinearLeastSquares App.
"""
from sigpy import alg, app, config, linop, prox

from sigpy import backend, block, conv, interp, fourier, thresh, util
from sigpy.backend import *
from sigpy.block import *
from sigpy.conv import *
from sigpy.interp import *
from sigpy.fourier import *
from sigpy.sim import *
from sigpy.thresh import *
from sigpy.util import *

__all__ = ['alg', 'app', 'config', 'linop', 'prox']
__all__.extend(backend.__all__)
__all__.extend(block.__all__)
__all__.extend(conv.__all__)
__all__.extend(interp.__all__)
__all__.extend(fourier.__all__)
__all__.extend(thresh.__all__)
__all__.extend(util.__all__)
