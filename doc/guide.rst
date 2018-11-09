User Guide
----------

**This guide is still under construction**

This user guide introduces several elements of SigPy, including:

- Basic usage
- Choosing computing device
- Using multi-CPU/GPU
- Building iterative methods


Basic Usage
===========

SigPy is designed to have as little learning curve as possible. Since almost all Python users already use NumPy, SigPy operates on NumPy arrays directly on CPU, and avoids defining any redundant functions.  When NumPy implementation is slow, SigPy uses Numba instead to translate Python functions to optimized machine code at runtime. For example, gridding functions in SigPy are implemented using Numba. For GPU, SigPy operates on CuPy arrays, which have the same interface as NumPy but are implemented in CUDA. 

SigPy does not bundle CuPy installation by default.
To enable CUDA support, you must install CuPy as an additional step.

While we try to make this documentation as self-contained as possible,
we refer you to the `NumPy documentation <https://docs.scipy.org/doc/numpy/index.html>`_,
and `CuPy documentation <https://docs-cupy.chainer.org/en/latest/index.html>`_
for general questions about NumPy/CuPy arrays and functions.

In the following, we will use the following abbreviations:

>>> import numpy as np
>>> import cupy as cp
>>> import sigpy as sp


Choosing Computing Device
=========================

SigPy provides a device class :class:`sigpy.Device` to allow you to specify the current computing device for functions and arrays.
It extends the ``Device`` class from CuPy.
Similar approach is also used by machine learning packages, such as TensorFlow, and PyTorch.

For example to create an array on GPU 1, we can do:

>>> with sp.Device(1):
>>>     x_on_gpu1 = cp.array([1, 2, 3, 4])

Note that this can also be done with ``cupy.cuda.Device``, and you can choose to use it as well.
The main difference is that :class:`sigpy.Device` maps -1 to CPU, and makes it easier to develop CPU/GPU generic code.

.. image:: figures/device.pdf
   :align: center

To transfer an array between device, we can use :class:`sigpy.to_device`. For example, to transfer a numpy array to GPU 1, we can do:

>>> x = np.array([1, 2, 3, 4])
>>> x_on_gpu1 = sp.to_device(x, 1)

Finally, we can use :func:`sigpy.Device.xp` to choose NumPy or CuPy adaptively.
For example, given a device id,
the following code creates an array on the appropriate device using the appropriate module:

>>> device = Device(id)
>>> xp = device.xp  # Returns NumPy if id == -1, otherwise returns CuPy
>>> with device:
>>>    x = xp.array([1, 2, 3, 4])


Using Multi-CPU/GPU
===================

SigPy uses MPI and MPI4Py for multi-CPU/GPU programming. We note that this is still under heavy development.

Although MPI may incur some overhead (for example redundant memory usage) for shared memory system,
we find an MPI solution to be the simplest for multi-threading in Python.
Another benefit is that an MPI parallelized code can run on both shared memory and distributed memory systems.

For example, if we consider the following shared memory configuration (one multi-core CPU and two GPUs),
and want to run the blue and red tasks concurrently:

.. image:: figures/multiprocess_desired.pdf
   :align: center

Then, using MPI, we can split the tasks to two MPI nodes as follows:

.. image:: figures/multiprocess_mpi.pdf
   :align: center

Note that tasks on each MPI node can run on any CPU/GPU device, and in our example, the blue task uses CPU and GPU 0, and
the red task uses CPU and GPU 1.

SigPy provides a communicator class :class:`sigpy.Communicator` that can be used to synchronize variables between ranks.
It extends the ``Communicator`` class from ChainerMN.


Building iterative methods
==========================

SigPy provides four abstraction classes (Linop, Prox, Alg, and App) for optimization based iterative methods. Such abstraction is inspired by similar structure in BART.

.. image:: figures/architecture.pdf
   :align: center

The Linop class abstracts a linear operator, and supports adjoint, addition, composing, and stacking. Prepackaged Linops include FFT, NUFFT, and wavelet, and common array manipulation functions. In particular, given a Linop ``A``, the following operations can be performed:

>>> A.H  # adjoint
>>> A.H * A  # compose
>>> A.H * A + lamda * I  # addition and scalar multiplication
>>> Hstack([A, B])  # horizontal stack
>>> Vstack([A, B])  # vertical stack
>>> Diag([A, B])  # diagonal stack

The Prox class abstracts a proximal operator, and can do stacking and conjugation. Prepackaged Proxs include L1/L2 regularization and projection functions. In particular, given a proximal operator ``proxg``, the following operations can be performed:

>>> Conj(proxg)  # convex conjugate
>>> UnitaryTransform(proxg, A)  # A.H * proxg * A
>>> Stack([proxg1, proxg2])  # diagonal stack

The Alg class abstracts iterative algorithms. Prepackaged Algs include conjugate gradient, (accelerated/proximal) gradient method, and primal dual hybrid gradient. A typical usage is as follows:

>>> while not alg.done():
>>>     alg.update()

Finally, the App class wraps the above three classes into a final deliverable application. Users can run an App without knowing the internal implementation. A typical usage of an App is as follows:

>>> out = app.run()
