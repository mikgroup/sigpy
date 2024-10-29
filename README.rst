SigPy
=====

SigPy is a package for signal processing, with emphasis on iterative methods. It is built to operate directly on NumPy arrays on CPU and CuPy arrays on GPU. SigPy also provides several domain-specific submodules: ``sigpy.plot`` for multi-dimensional array plotting, ``sigpy.mri`` for MRI reconstruction, and ``sigpy.mri.rf`` for MRI pulse design.

Installation
------------

SigPy requires Python version >= 3.5. The core module depends on ``numba``, ``numpy``, ``PyWavelets``, ``scipy``, and ``tqdm``.

Additional features can be unlocked by installing the appropriate packages. To enable the plotting functions, you will need to install ``matplotlib``. To enable CUDA support, you will need to install ``cupy``. And to enable MPI support, you will need to install ``mpi4py``.

Via ``conda``
*************

We recommend installing SigPy through ``conda``::

	conda install -c frankong sigpy
	# (optional for plot support) conda install matplotlib
	# (optional for CUDA support) conda install cupy
        # (optional for MPI support) conda install mpi4py

Via ``pip``
***********

SigPy can also be installed through ``pip``::

	pip install sigpy
	# (optional for plot support) pip install matplotlib
	# (optional for CUDA support) pip install cupy
        # (optional for MPI support) pip install mpi4py

Installation for Developers
***************************

If you want to contribute to the SigPy source code, we recommend you install it with ``pip`` in editable mode::

	cd /path/to/sigpy
	pip install -e .

To run tests and contribute, we recommend installing the following packages::

	pip install coverage ruff sphinx sphinx_rtd_theme black isort

and run the script ``run_tests.sh``.

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
