sigpy
=====

`sigpy` is a Python package for signal reconstruction, with GPU support using cupy.

`sigpy` provides commonly used signal processing functions, including convolution, FFT, NUFFT, wavelet transform, and thresholding functions. These operations are wrapped in a linear operator class (Linop), which allows easy manipulation, such as adjoint, addition, multiplication and composition. `sigpy` also implements popular iterative algorithms, such as conjugate gradient, (proximal) gradient method, primal dual hybrid gradient. All operations, except wavelet transform, can run on GPU.

`sigpy` also provides a submodule `sigpy.mri` that implements common MRI iterative reconstruction methods, including SENSE reconstruction, L1-wavelet reconstruction, and total-variation reconstruction.

Requirements
------------
This package requires python3, numpy, scipy, pywavelets, and numba. 

For optional gpu support, the package requires cupy.

For optional distributed programming support, the package requires mpi4py.

Documentation
-------------

Our documentation is hosted on readthedocs: https://sigpy.readthedocs.io
