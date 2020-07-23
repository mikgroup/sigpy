Linear Operators (`sigpy.linop`)
================================

.. automodule::
   sigpy.linop

The Linear Operator Class
-------------------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   sigpy.linop.Linop

Linop Manipulation
------------------

The following are classes that take in Linops and compose them to form a new Linop.

.. autosummary::
   :toctree: generated
   :nosignatures:   
   
   sigpy.linop.Conj
   sigpy.linop.Add
   sigpy.linop.Compose
   sigpy.linop.Hstack
   sigpy.linop.Vstack
   sigpy.linop.Diag

Basic Linops
------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   sigpy.linop.Embed
   sigpy.linop.Identity
   sigpy.linop.Reshape
   sigpy.linop.Slice
   sigpy.linop.Transpose

Computing Related Linops
------------------------
.. autosummary::
   :toctree: generated
   :nosignatures:
   
   sigpy.linop.ToDevice
   sigpy.linop.AllReduce
   sigpy.linop.AllReduceAdjoint

Convolution Linops
------------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   sigpy.linop.ConvolveData
   sigpy.linop.ConvolveDataAdjoint
   sigpy.linop.ConvolveFilter
   sigpy.linop.ConvolveFilterAdjoint

Fourier Linops
--------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   sigpy.linop.FFT
   sigpy.linop.IFFT
   sigpy.linop.NUFFT
   sigpy.linop.NUFFTAdjoint

Multiplication Linops
---------------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   sigpy.linop.MatMul
   sigpy.linop.RightMatMul
   sigpy.linop.Multiply

Interapolation Linops
---------------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   sigpy.linop.Interpolate
   sigpy.linop.Gridding

Array Manipulation Linops
-------------------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   sigpy.linop.Resize
   sigpy.linop.Flip
   sigpy.linop.Downsample
   sigpy.linop.Upsample
   sigpy.linop.Circshift
   sigpy.linop.Sum
   sigpy.linop.Tile
   sigpy.linop.FiniteDifference

Wavelet Transform Linops
------------------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   sigpy.linop.Wavelet
   sigpy.linop.InverseWavelet

Block Reshape Linops
--------------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   sigpy.linop.ArrayToBlocks
   sigpy.linop.BlocksToArray


