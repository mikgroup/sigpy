"""The core module contains functions and classes for signal processing.

SigPy provides simple interfaces to commonly used signal processing functions,
including convolution, FFT, NUFFT, wavelet transform, and thresholdings.
All functions, except wavelet transform, can run on both CPU and GPU.

These functions are wrapped into higher level classes (Linop and Prox)
that can be used in conjuction with Alg to form an App.

"""
from sigpy import alg, app, config, linop, prox

from sigpy import (backend, block, conv, interp,
                   fourier, pytorch, sim, thresh, util, wavelet)
from sigpy.backend import *  # noqa
from sigpy.block import *  # noqa
from sigpy.conv import *  # noqa
from sigpy.interp import *  # noqa
from sigpy.fourier import *  # noqa
from sigpy.pytorch import * # noqa
from sigpy.sim import *  # noqa
from sigpy.thresh import *  # noqa
from sigpy.util import *  # noqa
from sigpy.wavelet import *  # noqa

__all__ = ['alg', 'app', 'config', 'linop', 'prox']
__all__.extend(backend.__all__)
__all__.extend(block.__all__)
__all__.extend(conv.__all__)
__all__.extend(interp.__all__)
__all__.extend(fourier.__all__)
__all__.extend(pytorch.__all__)
__all__.extend(sim.__all__)
__all__.extend(thresh.__all__)
__all__.extend(util.__all__)
__all__.extend(wavelet.__all__)
