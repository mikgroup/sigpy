.. image:: https://travis-ci.com/mikgroup/sigpy.svg?branch=master
	   :target: https://travis-ci.com/mikgroup/sigpy
	
.. image:: https://codecov.io/gh/mikgroup/sigpy/branch/master/graph/badge.svg
	   :target: https://codecov.io/gh/mikgroup/sigpy
	   
.. image:: https://readthedocs.org/projects/sigpy/badge/?version=latest
	   :target: https://sigpy.readthedocs.io/en/latest/?badge=latest
	   :alt: Documentation Status

Overview
========

Introduction
------------
SigPy is a package for signal processing, with emphasis on iterative methods. It is built to operate directly on numpy arrays on CPU and cupy arrays on GPU. Its main features include:

* A unified CPU/GPU interface to signal processing functions, including convolution, FFT, NUFFT, wavelet transform, and thresholding functions.
* Linear operator classes (``Linop``) that can do adjoint, addition, composing, and stacking.
* Proximal operator classes (``Prox``) that can do stacking, and conjugation.
* Iterative algorithm classes (``Alg``), including conjugate gradient, (accelerated/proximal) gradient method, and primal dual hybrid gradient.
* Application classes (``App``) that wrap ``Alg``, ``Linop``, and ``Prox`` to form a final deliverable for each application.

SigPy also provides a submodule sigpy.mri for MRI iterative reconstruction methods. Its main features include:

* Commonly used MRI reconstruction methods as an ``App``: SENSE reconstruction, l1-wavelet reconstruction, total-variation reconstruction, and JSENSE reconstruction
* Convenient simulation and sampling functions, including poisson-disc sampling function, and shepp-logan phantom generation function.

Finally, SigPy provides a preliminary submodule sigpy.learn that implements convolutional sparse coding, and linear regression, using the core module.

Installation
------------
The package can be installed via pip::

	# (optional for CUDA support) pip install cupy
	# (optional for MPI support) pip install mpi4py
	pip install sigpy

	
Or via conda::

	# (optional for CUDA support) conda install cupy
	# (optional for MPI support) conda install mpi4py
	conda install -c frankong sigpy

Alternatively, the package can be installed from source with the following requirements:

* python3
* numpy
* pywavelets
* numba

Documentation
-------------
Our documentation is hosted on Read the Docs: https://sigpy.readthedocs.io
