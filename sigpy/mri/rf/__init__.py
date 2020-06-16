"""This MRI submodule contains functions and classes for MRI pulse design.

It contains functions to design a variety of RF pulses for MRI, such as SLR,
adiabatic, parallel transmit, multibanded, and others. The submodule also
includes other functions to assist with pulse design, such as I/O functions,
trajectory/gradient designers, and Bloch simulators.

Explore RF design tutorials at `sigpy-rf-tutorials`_. These are primarily
Jupyter Notebooks, and provide more detailed instruction on pulse design
workflow and function use.

See in-progress features at `sigpy-rf`_.

.. _sigpy-rf-tutorials: https://github.com/jonbmartin/sigpy-rf-tutorials
.. _sigpy-rf: https://github.com/jonbmartin/sigpy-rf

"""
from sigpy.mri import linop

from sigpy.mri.rf import adiabatic, b1sel, io, multiband, optcont, ptx,\
    shim, sim, slr, trajgrad, util
from sigpy.mri.rf.adiabatic import *  # noqa
from sigpy.mri.rf.b1sel import *  # noqa
from sigpy.mri.rf.io import *  # noqa
from sigpy.mri.rf.linop import *  # noqa
from sigpy.mri.rf.multiband import *  # noqa
from sigpy.mri.rf.optcont import *  # noqa
from sigpy.mri.rf.ptx import *  # noqa
from sigpy.mri.rf.shim import *  # noqa
from sigpy.mri.rf.sim import *  # noqa
from sigpy.mri.rf.slr import *  # noqa
from sigpy.mri.rf.trajgrad import *  # noqa
from sigpy.mri.rf.util import *  # noqa

__all__ = ['linop']
__all__.extend(adiabatic.__all__)
__all__.extend(b1sel.__all__)
__all__.extend(io.__all__)
__all__.extend(multiband.__all__)
__all__.extend(optcont.__all__)
__all__.extend(ptx.__all__)
__all__.extend(sim.__all__)
__all__.extend(shim.__all__)
__all__.extend(slr.__all__)
__all__.extend(trajgrad.__all__)
__all__.extend(util.__all__)
