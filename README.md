Overview
========

Introduction
------------
SigPy is a package for developing iterative signal reconstruction methods. It is built to operate directly on numpy arrays on CPU and cupy arrays on GPU. This allows users to easily use Sigpy with other packages. It also provides preliminary support for distributed computing using mpi4py. 

SigPy provides simple interfaces to commonly used signal processing functions, including convolution, FFT, NUFFT, wavelet transform, and thresholding functions. All functions, except wavelet transform, can run on both CPU and GPU.

These functions are wrapped in two higher level classes to better interface with iterative methods: Linop, which abstracts linear operator, and Prox, which abstracts proximal operator. SigPy provides an abstraction class Alg for iterative algorithms, and implements commonly used methods, including conjugate gradient, (accelerated/proximal) gradient method, and primal dual hybrid gradient.

These classes can then be used to build an App as a final deliverable. An App simplifies the usage of Alg, and provides convenient features such as iteration progress bars. A particularly useful App implemented is the LinearLeastSquares App.

SigPy also provides a submodule sigpy.mri that uses the core module to implement common MRI iterative reconstruction methods, including SENSE reconstruction, l1-wavelet reconstruction, total-variation reconstruction, and JSENSE reconstruction. In addition, it provides convenient simulation and sampling functions, such as poisson-disc sampling function, and shepp-logan phantom generation function.

SigPy provides a preliminary submodule sigpy.learn that implements convolutional sparse coding, and linear regression, using the core module.

Installation
------------
The package is on PyPI, and can be installed via pip:

	pip install sigpy

To enable GPU support, the package requires CuPy.

To enable distributed programming support, the package requires mpi4py.
	
Alternatively, the package can be installed from source with the following requirements:

- python3
- numpy
- pywavelets
- numba

Documentation
-------------
Our documentation is hosted on Read the Docs: https://sigpy.readthedocs.io
