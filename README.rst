Overview
========

.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
           :target: https://opensource.org/licenses/BSD-3-Clause
	   
.. image:: https://travis-ci.com/mikgroup/sigpy.svg?branch=master
	   :target: https://travis-ci.com/mikgroup/sigpy
	
.. image:: https://codecov.io/gh/mikgroup/sigpy/branch/master/graph/badge.svg
	   :target: https://codecov.io/gh/mikgroup/sigpy
	   
.. image:: https://readthedocs.org/projects/sigpy/badge/?version=latest
	   :target: https://sigpy.readthedocs.io/en/latest/?badge=latest
	   :alt: Documentation Status

`Source Code <https://github.com/mikgroup/sigpy>`_ | `Documentation <https://sigpy.readthedocs.io>`_ | `Tutorial <https://github.com/mikgroup/sigpy-tutorials>`_

SigPy is a package for signal processing, with emphasis on iterative methods. It is built to operate directly on numpy arrays on CPU and cupy arrays on GPU. SigPy also provides several submodules that build on top of the core module: sigpy.plot for multi-dimensional array plotting, sigpy.mri for MRI iterative reconstruction, and sigpy.learn for dictionary learning.

Installation
------------

SigPy requires Python version >= 3.5. The core module depends on:

* ``numba``
* ``numpy``
* ``PyWavelets``
* ``tqdm``

To enable the plotting functions, you will need to install ``matplotlib``. To enable CUDA support, you will need to install ``cupy``. And to enable MPI support, you will need to install ``mpi4py``.

Via ``conda``
*************

For general users, we recommend installing SigPy through ``conda``::

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

If you want to contribute to the SigPy source code, we recommend you install with ``pip`` in editable mode::

	cd /path/to/sigpy
	pip install -e .
