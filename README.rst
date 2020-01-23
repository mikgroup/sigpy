SigPy.RF
=====

.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
	:target: https://opensource.org/licenses/BSD-3-Clause
	   
.. image:: https://travis-ci.com/mikgroup/sigpy.svg?branch=master
	:target: https://travis-ci.com/mikgroup/sigpy
	   
.. image:: https://readthedocs.org/projects/sigpy/badge/?version=latest
	:target: https://sigpy.readthedocs.io/en/latest/?badge=latest
	:alt: Documentation Status
	
.. image:: https://codecov.io/gh/mikgroup/sigpy/branch/master/graph/badge.svg
	:target: https://codecov.io/gh/mikgroup/sigpy


`RF Design Fork Source Code <https://github.com/jonbmartin/sigpy-rf>`_ | `Source Code for SigPy <https://github.com/mikgroup[/sigpy>`_  | `Documentation <https://sigpy.readthedocs.io>`_ | `RF Design Tutorials <https://github.com/jonbmartin/sigpy-rf-tutorials>`_

SigPy.RF is an expanded version of the SigPy package for signal processing, and includes a wide range of RF pulse design tools for MRI. It is built to operate directly on NumPy arrays on CPU and CuPy arrays on GPU. SigPy.RF provides several domain-specific submodules: ``sigpy.plot`` for multi-dimensional array plotting, ``sigpy.mri`` for MRI iterative reconstruction, and ``sigpy.learn`` for dictionary learning. The goal of this fork is to develop the ``sigpy.mri.rf`` submodule for RF pulse design. Features are gradually being moved from the sigpy-rf fork to sigpy. 

Installation
------------

SigPy.RF requires Python version >= 3.5. The core module depends on ``numba``, ``numpy``, ``PyWavelets``, ``scipy``, and ``tqdm``.

Additional features can be unlocked by installing the appropriate packages. To enable the plotting functions, you will need to install ``matplotlib``. To enable CUDA support, you will need to install ``cupy``. And to enable MPI support, you will need to install ``mpi4py``.

Features
--------

CPU/GPU Signal Processing Functions
***********************************
SigPy provides signal processing functions with a unified CPU/GPU interface. For example, the same code can perform a CPU or GPU convolution on the input array device:

.. code:: python

	  # CPU convolve
	  x = numpy.array([1, 2, 3, 4, 5])
	  y = numpy.array([1, 1, 1])
	  z = sigpy.convolve(x, y)

	  # GPU convolve
	  x = cupy.array([1, 2, 3, 4, 5])
	  y = cupy.array([1, 1, 1])
	  z = sigpy.convolve(x, y)

Iterative Algorithms
********************
SigPy also provides convenient abstractions and classes for iterative algorithms. A compressed sensing experiment can be implemented in four lines using SigPy:

.. code:: python

	  # Given some observation vector y, and measurement matrix mat
	  A = sigpy.linop.MatMul([n, 1], mat)  # define forward linear operator
	  proxg = sigpy.prox.L1Reg([n, 1], lamda=0.001)  # define proximal operator
	  x_hat = sigpy.app.LinearLeastSquares(A, y, proxg=proxg).run()  # run iterative algorithm

PyTorch Interoperability
************************
Want to do machine learning without giving up signal processing? SigPy has convenient functions to convert arrays and linear operators into PyTorch Tensors and Functions. For example, given a cupy array ``x``, and a ``Linop`` ``A``, we can convert them to Pytorch:

.. code:: python

	  x_torch = sigpy.to_pytorch(x)
	  A_torch = sigpy.to_pytorch_function(A)


Citation
***********************
If you use and wish to cite SigPy's RF pulse design features, please use:

.. [MOMT20] Martin, J.B.; Ong, F.; Ma, J.; Tamir, J.I.; Lustig, M.; Grissom, W.A. SigPy.RF: A Package for Comprehensive Open-Source RF Pulse Design. ISMRM Workshop on Data Sampling \& Image Reconstruction, 2020.

