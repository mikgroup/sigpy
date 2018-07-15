``sigpy`` Overview
================

Introduction
------------

``sigpy`` is a Python package for signal reconstruction, with GPU support using ``cupy``.

``sigpy`` provides commonly used signal processing functions, including convolution, FFT, NUFFT, wavelet transform, and thresholding functions. All operations, except wavelet transform, can run on GPU. These operations are wrapped either in a linear operator class (``Linop``) or a proximal operator class (``Prox``) for easy usage in iterative algorithms. ``sigpy`` also implements popular iterative algorithms, such as conjugate gradient, (accelerated/proximal) gradient method, and primal dual hybrid gradient.

``sigpy`` provides a submodule ``sigpy.mri`` that uses the core module to implement common MRI iterative reconstruction methods, including SENSE reconstruction, L1-wavelet reconstruction, and total-variation reconstruction. In addition, it provides convenient simulation and sampling functions, such as poisson-disk sampling function.

``sigpy`` also provides a preliminary submodule ``sigpy.learn`` that implements convolutional sparse coding, and linear regression.

Installation
------------
The package is on PyPI, and can be installed via pip

	pip install sigpy
	
Alternatively, the package can also be installed with the following required packages.

Requirements
------------
This package requires python3, numpy, scipy, pywavelets, and numba.

For optional gpu support, the package requires cupy.

For optional distributed programming support, the package requires mpi4py.

Documentation
-------------

Our documentation is hosted on Read the Docs: https://sigpy.readthedocs.io
