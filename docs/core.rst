Functions (`sigpy`)
===================
.. automodule::
   sigpy
		
Computing Backend Functions
---------------------------
.. automodule::
   sigpy.backend

.. autosummary::
   :toctree: generated
   :nosignatures:

   sigpy.Device
   sigpy.get_device
   sigpy.get_array_module
   sigpy.cpu_device
   sigpy.to_device
   sigpy.copyto
   sigpy.Communicator

Block Reshape Functions
-----------------------
.. automodule::
   sigpy.block

.. autosummary::
   :toctree: generated
   :nosignatures:

   sigpy.array_to_blocks
   sigpy.blocks_to_array

Convolution Functions
---------------------
.. automodule::
   sigpy.conv

.. autosummary::
   :toctree: generated
   :nosignatures:

   sigpy.convolve
   sigpy.convolve_data_adjoint
   sigpy.convolve_filter_adjoint

Fourier Functions
-----------------
.. automodule::
   sigpy.fourier

.. autosummary::
   :toctree: generated
   :nosignatures:

   sigpy.fft
   sigpy.ifft
   sigpy.nufft
   sigpy.nufft_adjoint
   sigpy.estimate_shape

Interpolation Functions
-----------------------
.. automodule::
   sigpy.interp

.. autosummary::
   :toctree: generated
   :nosignatures:

   sigpy.interpolate
   sigpy.gridding

Pytorch Interop Functions
-------------------------
.. automodule::
   sigpy.pytorch

.. autosummary::
   :toctree: generated
   :nosignatures:

   sigpy.to_pytorch
   sigpy.from_pytorch
   sigpy.to_pytorch_function

Simulation Functions
--------------------
.. automodule::
   sigpy.sim

.. autosummary::
   :toctree: generated
   :nosignatures:

   sigpy.shepp_logan

Thresholding Functions
----------------------
.. automodule::
   sigpy.thresh

.. autosummary::
   :toctree: generated
   :nosignatures:

   sigpy.soft_thresh
   sigpy.hard_thresh
   sigpy.l1_proj
   sigpy.l2_proj
   sigpy.linf_proj
   sigpy.psd_proj

Utility Functions
-----------------
.. automodule::
   sigpy.util

.. autosummary::
   :toctree: generated
   :nosignatures:

   sigpy.resize
   sigpy.flip
   sigpy.circshift
   sigpy.downsample
   sigpy.upsample
   sigpy.dirac
   sigpy.randn
   sigpy.triang
   sigpy.hanning
   sigpy.monte_carlo_sure
   sigpy.axpy
   sigpy.xpay

Wavelet Functions
-----------------
.. automodule::
   sigpy.wavelet

.. autosummary::
   :toctree: generated
   :nosignatures:

   sigpy.fwt
   sigpy.iwt
