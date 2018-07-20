Overview
========

Introduction
------------
``sigpy`` is a Python package for signal reconstruction, with GPU support using ``cupy``.

``sigpy`` provides commonly used signal processing functions, including convolution, FFT, NUFFT, wavelet transform, and thresholding functions. All operations, except wavelet transform, can run on GPU. These operations are wrapped in a linear operator class (``Linop``) or a proximal operator class (``Prox``) for easy usage in iterative algorithms. ``sigpy`` also implements commonly used iterative algorithms, such as conjugate gradient, (accelerated/proximal) gradient method, and primal dual hybrid gradient.

``sigpy`` provides a submodule ``sigpy.mri`` that uses the core module to implement common MRI iterative reconstruction methods, including SENSE reconstruction, L1-wavelet reconstruction, total-variation reconstruction, and JSENSE reconstruction. In addition, it provides convenient simulation and sampling functions, such as poisson-disc sampling function.

``sigpy`` also provides a preliminary submodule ``sigpy.learn`` that implements convolutional sparse coding, and linear regression.

Installation
------------
The package is on PyPI, and can be installed via pip:

	pip install sigpy

For optional gpu support, the package requires ``cupy``.

For optional distributed programming support, the package requires ``mpi4py``.
	
Alternatively, the package can be installed from source with the following requirements:

- python3
- numpy
- scipy
- pywavelets
- numba

Documentation
-------------
Our documentation is hosted on Read the Docs: https://sigpy.readthedocs.io
