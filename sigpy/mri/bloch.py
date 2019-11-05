r"""Bloch equation functions for spin-:math:`\frac{1}{2}` particles.

The functions consider both Bloch vector and density
matrix representations. In particular, a Bloch vector :math:`M`
has three components :math:`\left[M_x, M_y, M_z\right]`.
A density matrix :math:`\rho` is a 2 x 2 matrix and can be computed
from a Bloch vector :math:`M` as:

.. math::
    \rho = \frac{1}{2}
    \begin{bmatrix}
    1 + M_z & M_x - i M_y\\
    M_x + i M_y & 1 - M_z
    \end{bmatrix}.

Internally, all functions use the density matrices.

"""
import numpy as np
import sigpy as sp


__all__ = ['bloch_forward',
           'free_induction_decay', 'hard_pulse_rotation',
           'init_density_matrix', 'init_bloch_vector',
           'is_density_matrix', 'is_bloch_vector',
           'to_density_matrix', 'to_bloch_vector']


def init_bloch_vector(shape=None, dtype=np.complex, device=sp.cpu_device):
    """Initialize magnetization in Bloch vector representation.

    Args:
        shape (tuple): batch shape.
        dtype (Dtype): data type.
        device (Device): device.

    Returns:
        array: magnetization array.

    """
    device = sp.Device(device)
    xp = device.xp
    with device:
        if shape is None:
            shape = []

        m = xp.zeros(list(shape) + [3], dtype=dtype)
        m[..., 2] = 1
        return m


def init_density_matrix(shape=None, dtype=np.complex, device=sp.cpu_device):
    """Initialize magnetization in density matrix representation.

    Args:
        shape (tuple): batch shape.
        dtype (Dtype): data type.
        device (Device): device.

    Returns:
        array: magnetization array.

    """
    device = sp.Device(device)
    xp = device.xp
    with device:
        if shape is None:
            shape = []

        p = xp.zeros(list(shape) + [2, 2], dtype=dtype)
        p[..., 0, 0] = 1
        return p


def is_bloch_vector(input):
    """Determine if the input array can be a Bloch vector.

    Args:
        input (array): input array.

    Returns:
        bool: whether input can be a Bloch vector.

    """
    device = sp.get_device(input)
    xp = device.xp
    return input.shape[-1] == 3 and xp.isrealobj(input)


def is_density_matrix(input):
    """Determine if the input array can be a density matrix.

    Args:
        input (array): input array.

    Returns:
        bool: whether input can be a density matrix.

    """
    return input.shape[-1] == 2 and input.shape[-2] == 2


def to_bloch_vector(input):
    """Convert magnetization array to Bloch vector representation.

    Args:
        input (array): magnetization array.

    Returns:
        m (array): Bloch vector reprsentation of shape [..., 2, 2].

    """
    device = sp.get_device(input)
    xp = device.xp

    with device:
        if is_bloch_vector(input):
            m = input
        elif is_density_matrix(input):
            mx = 2 * input[..., 1, 0].real
            my = 2 * input[..., 1, 0].imag
            mz = input[..., 0, 0] - input[..., 1, 1]
            m = xp.stack([mx, my, mz], axis=-1)
        else:
            raise ValueError('Input is not in either Bloch vector or '
                             'density matrix representation.')

    return m


def to_density_matrix(input):
    """Convert magnetization array to density matrix.

    Args:
        input (array): magnetization array.

    Returns:
        p (array): density matrix reprsentation of shape [..., 2, 2].

    """
    device = sp.get_device(input)
    xp = device.xp

    with device:
        if is_bloch_vector(input):
            mx, my, mz = input[..., 0], input[..., 1], input[..., 2]
            p = xp.stack([xp.stack([1 + mz, mx - 1j * my], -1),
                          xp.stack([mx + 1j * my, 1 - mz], -1)], -2)
            p /= 2
        elif is_density_matrix(input):
            p = input
        else:
            raise ValueError('Input is not in either Bloch vector or '
                             'density matrix representation.')

    return p


def _exp(b1):
    device = sp.get_device(b1)
    xp = device.xp

    with device:
        alpha = xp.abs(b1)
        phi = xp.angle(b1)

        cos_alpha = xp.cos(alpha / 2)
        sin_alpha = xp.sin(alpha / 2)
        cos_phi = xp.cos(phi)
        sin_phi = xp.sin(phi)

        return xp.array(
            [[cos_alpha, -1j * sin_alpha * cos_phi - sin_alpha * sin_phi],
             [-1j * sin_alpha * cos_phi + sin_alpha * sin_phi, cos_alpha]])


def hard_pulse_rotation(input, b1):
    """Apply hard pulse rotation to input magnetization.

    Args:
        input (array): magnetization array.
        b1 (complex float): complex B1 value in radian.

    Returns:
        array: magnetization array after hard pulse rotation,
            in representation consistent with input.

    """
    p = to_density_matrix(input)

    device = sp.get_device(p)
    with device:
        b1 = sp.to_device(b1, device)
        p = _exp(b1) @ p @ _exp(-b1)

    if is_bloch_vector(input):
        return to_bloch_vector(p)
    else:
        return p


def free_induction_decay(input, f0, t1, t2, dt):
    """Simulate free induction decay to input magnetization.

    Off-resonance, T1 recovery, and T2 relaxation array dimensions must be
    consistent with the input batch dimensions.

    Args:
        input (array): magnetization array.
        f0 (array): off-resonance frequency values.
        t1 (array): T1 recovery values.
        t2 (array): T2 relaxation values.
        dt (float): free induction decay duration.

    Returns:
        array: magnetization array after hard pulse rotation,
            in representation consistent with input.

    """
    p = to_density_matrix(input)

    device = sp.get_device(input)
    xp = device.xp

    with device:
        e2 = xp.exp(-dt / t2)
        e1 = xp.exp(-dt / t1)
        e0 = xp.exp(-1j * dt * 2 * np.pi * f0)

        p = p.copy()
        p[..., 0, 0] *= e1
        p[..., 1, 1] *= e1
        p[..., 1, 0] *= e0 * e2
        p[..., 0, 1] *= xp.conj(e0) * e2

        p[..., 0, 0] += 1 - e1

    if is_bloch_vector(input):
        return to_bloch_vector(p)
    else:
        return p


def bloch_forward(input, b1, f0, t1, t2, dt):
    """Bloch equation forward evolution.

    The function uses the hard pulse approximation. Given an array of
    B1 complex amplitudes, it simulates a sequence of free induction decay
    followed by a hard pulse rotation.

    The units of ``f0``, ``t1``, ``t2``, and ``dt`` must be consistent.

    Args:
        input (array): magnetization array either in Bloch vector
            representation with shape [..., 3] or in density matrix
            representation with [..., 2, 2].
        b1 (array): complex B1 array in radian.
        f0 (array): off resonance frequency array.
        t1 (array): T1 recovery array.
        t2 (array): T2 relaxation array.
        dt (scalar): time duration for free induction decay.

    Returns:
        array: resulting magnetization array with the same shape as input.

    Examples:
        Simulating an on-resonant spin under 90 degree pulse.
        The 90 degree pulse is discretized into 1000 time points.

        >>> input = np.array([0, 0, 1])
        >>> b1 = np.pi / 2 * np.ones(1000) / 1000
        >>> dt = 1
        >>> f0 = 0
        >>> t1 = np.infty
        >>> t2 = np.infty
        >>> output = bloch_forward(input, b1, f0, t1, t2, dt)

        Simulating spins under 90 degree pulse across frequencies.
        Off-resonance frequencies are discretized into 100 values.

        >>> input = np.repeat([[0, 0, 1]], 100, axis=0)
        >>> b1 = np.pi / 2 * np.ones(1000) / 1000
        >>> dt = 1
        >>> f0 = np.linspace(-np.pi, np.pi, 100)
        >>> t1 = np.full(100, np.infty)
        >>> t2 = np.full(100, np.infty)
        >>> output = bloch_forward(input, b1, f0, t1, t2, dt)

    """
    p = to_density_matrix(input)

    for n in range(len(b1)):
        p = free_induction_decay(p, f0, t1, t2, dt)
        p = hard_pulse_rotation(p, b1[n])

    if is_bloch_vector(input):
        return to_bloch_vector(p)
    else:
        return p
