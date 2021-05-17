MRI RF Design (`sigpy.mri.rf`)
==============================

.. automodule::
   sigpy.mri.rf

Adiabatic Pulse Design Functions
--------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    sigpy.mri.rf.adiabatic.bir4
    sigpy.mri.rf.adiabatic.hypsec
    sigpy.mri.rf.adiabatic.wurst
    sigpy.mri.rf.adiabatic.goia_wurst
    sigpy.mri.rf.adiabatic.bloch_siegert_fm

B1-Selective Pulse Design Functions
-----------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    sigpy.mri.rf.b1sel.dz_b1_rf
    sigpy.mri.rf.b1sel.dz_b1_gslider_rf
    sigpy.mri.rf.b1sel.dz_b1_hadamard_rf

RF Linear Operators
--------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    sigpy.mri.rf.linop.PtxSpatialExplicit

Pulse Multibanding Functions
----------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    sigpy.mri.rf.multiband.mb_rf
    sigpy.mri.rf.multiband.dz_pins

Optimal Control Design Functions
--------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    sigpy.mri.rf.optcont.blochsim
    sigpy.mri.rf.optcont.deriv

Parallel Transmit Pulse Designers
---------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    sigpy.mri.rf.ptx.stspa
    sigpy.mri.rf.ptx.stspk

RF Shimming Functions
--------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    sigpy.mri.rf.shim.calc_shims
    sigpy.mri.rf.shim.init_optimal_spectral
    sigpy.mri.rf.shim.init_circ_polar

RF Pulse Simulation
--------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    sigpy.mri.rf.sim.abrm
    sigpy.mri.rf.sim.abrm_nd
    sigpy.mri.rf.sim.abrm_hp
    sigpy.mri.rf.sim.abrm_ptx

SLR Pulse Design Functions
--------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    sigpy.mri.rf.slr.dzrf
    sigpy.mri.rf.slr.root_flip
    sigpy.mri.rf.slr.dz_gslider_rf
    sigpy.mri.rf.slr.dz_gslider_b
    sigpy.mri.rf.slr.dz_hadamard_b
    sigpy.mri.rf.slr.dz_recursive_rf

Trajectory and Gradient Design Functions
----------------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    sigpy.mri.rf.trajgrad.min_trap_grad
    sigpy.mri.rf.trajgrad.trap_grad
    sigpy.mri.rf.trajgrad.spiral_varden
    sigpy.mri.rf.trajgrad.spiral_arch
    sigpy.mri.rf.trajgrad.epi
    sigpy.mri.rf.trajgrad.rosette
    sigpy.mri.rf.trajgrad.stack_of
    sigpy.mri.rf.trajgrad.spokes_grad
    sigpy.mri.rf.traj_complex_to_array
    sigpy.mri.rf.traj_array_to_complex
    sigpy.mri.rf.min_time_gradient

RF Utility
--------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    sigpy.mri.rf.util.dinf

I/O
--------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    sigpy.mri.rf.io.siemens_rf
    sigpy.mri.rf.io.signa
    sigpy.mri.rf.io.ge_rf_params
    sigpy.mri.rf.io.philips_rf_params
