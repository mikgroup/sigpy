# -*- coding: utf-8 -*-
"""Adiabatic Pulse Design functions.

"""
import numpy as np

__all__ = ['bir4', 'hypsec', 'wurst', 'goia_wurst', 'bloch_siegert_fm']


def bir4(n, beta, kappa, theta, dw0):
    r"""Design a BIR-4 adiabatic pulse.

    BIR-4 is equivalent to two BIR-1 pulses back-to-back.

    Args:
        n (int): number of samples (should be a multiple of 4).
        beta (float): AM waveform parameter.
        kappa (float): FM waveform parameter.
        theta (float): flip angle in radians.
        dw0: FM waveform scaling (radians/s).

    Returns:
        2-element tuple containing

        - **a** (*array*): AM waveform.
        - **om** (*array*): FM waveform (radians/s).

    References:
        Staewen, R.S. et al. (1990). '3-D FLASH Imaging using a single surface
        coil and a new adiabatic pulse, BIR-4'.
        Invest. Radiology, 25:559-567.
     """

    dphi = np.pi + theta/2

    t = np.arange(0, n) / n

    a1 = np.tanh(beta * (1 - 4 * t[:n // 4]))
    a2 = np.tanh(beta * (4 * t[n // 4:n // 2] - 1))
    a3 = np.tanh(beta * (3 - 4 * t[n // 2:3 * n // 4]))
    a4 = np.tanh(beta * (4 * t[3 * n // 4:] - 3))

    a = np.concatenate((a1, a2, a3, a4)).astype(complex)
    a[n // 4:3 * n // 4] = a[n // 4:3 * n // 4] * np.exp(1j * dphi)

    om1 = dw0 * np.tan(kappa * 4 * t[:n // 4]) / np.tan(kappa)
    om2 = dw0 * np.tan(kappa * (4 * t[n // 4:n // 2] - 2)) / np.tan(kappa)
    om3 = dw0 * np.tan(kappa * (4 * t[n // 2:3 * n // 4] - 2)) / np.tan(kappa)
    om4 = dw0 * np.tan(kappa * (4 * t[3 * n // 4:] - 4)) / np.tan(kappa)

    om = np.concatenate((om1, om2, om3, om4))

    return a, om


def hypsec(n=512, beta=800, mu=4.9, dur=0.012):
    r"""Design a hyperbolic secant adiabatic pulse.

    mu * beta becomes the amplitude of the frequency sweep

    Args:
        n (int): number of samples (should be a multiple of 4).
        beta (float): AM waveform parameter.
        mu (float): a constant, determines amplitude of frequency sweep.
        dur (float): pulse time (s).

    Returns:
        2-element tuple containing

        - **a** (*array*): AM waveform.
        - **om** (*array*): FM waveform (radians/s).

    References:
        Baum, J., Tycko, R. and Pines, A. (1985). 'Broadband and adiabatic
        inversion of a two-level system by phase-modulated pulses'.
        Phys. Rev. A., 32:3435-3447.
     """

    t = np.arange(-n // 2, n // 2) / n * dur

    a = np.cosh(beta * t) ** -1
    om = -mu * beta * np.tanh(beta * t)

    return a, om


def wurst(n=512, n_fac=40, bw=40e3, dur=2e-3):
    r"""Design a WURST (wideband, uniform rate, smooth truncation) adiabatic
     inversion pulse

    Args:
        n (int): number of samples (should be a multiple of 4).
        n_fac (int): power to exponentiate to within AM term. ~20 or greater is
         typical.
        bw (float): pulse bandwidth.
        dur (float): pulse time (s).


    Returns:
        2-element tuple containing

        - **a** (*array*): AM waveform.
        - **om** (*array*): FM waveform (radians/s).

    References:
        Kupce, E. and Freeman, R. (1995). 'Stretched Adiabatic Pulses for
        Broadband Spin Inversion'.
        J. Magn. Reson. Ser. A., 117:246-256.
     """

    t = np.arange(0, n) * dur / n

    a = 1 - np.power(np.abs(np.cos(np.pi * t / dur)), n_fac)
    om = np.linspace(-bw / 2, bw / 2, n) * 2 * np.pi

    return a, om


def goia_wurst(n=512, dur=3.5e-3, f=0.9, n_b1=16, m_grad=4,
               b1_max=817, bw=20000):
    r"""Design a GOIA (gradient offset independent adiabaticity) WURST
     inversion pulse

    Args:
        n (int): number of samples.
        dur (float): pulse duration (s).
        f (float): [0,1] gradient modulation factor
        n_b1 (int): order for B1 modulation
        m_grad (int): order for gradient modulation
        b1_max (float): maximum b1 (Hz)
        bw (float): pulse bandwidth (Hz)

    Returns:
        3-element tuple containing:

        - **a** (*array*): AM waveform (Hz)
        - **om** (*array*): FM waveform (Hz)
        - **g** (*array*): normalized gradient waveform

    References:
        O. C. Andronesi, S. Ramadan, E.-M. Ratai, D. Jennings, C. E. Mountford,
        A. G. Sorenson.
        J Magn Reson, 203:283-293, 2010.

    """

    t = np.arange(0, n) * dur / n

    a = b1_max*(1 - np.abs(np.sin(np.pi / 2 * (2 * t / dur - 1))) ** n_b1)
    g = (1 - f) + f * np.abs(np.sin(np.pi / 2 * (2 * t / dur - 1))) ** m_grad
    om = np.cumsum((a ** 2) / g) * dur / n
    om = om - om[n//2 + 1]
    om = g * om
    om = om / np.max(np.abs(om)) * bw / 2

    return a, om, g


def bloch_siegert_fm(n=512, dur=2e-3, b1p=20., k=42.,
                     gamma=2*np.pi*42.58):
    r"""
    U-shaped FM waveform for adiabatic Bloch-Siegert :math:`B_1^{+}` mapping
    and spatial encoding.

    Args:
        n (int): number of time points
        dur (float): duration in seconds
        b1p (float): nominal amplitude of constant AM waveform
        k (float): design parameter that affects max in-band
            perturbation
        gamma (float): gyromagnetic ratio

    Returns:
        om (array): FM waveform (radians/s).

    References:
        M. M. Khalighi, B. K. Rutt, and A. B. Kerr.
        Adiabatic RF pulse design for Bloch-Siegert B1+ mapping.
        Magn Reson Med, 70(3):829–835, 2013.

        M. Jankiewicz, J. C. Gore, and W. A. Grissom.
        Improved encoding pulses for Bloch-Siegert B1+ mapping.
        J Magn Reson, 226:79–87, 2013.

    """

    t = np.arange(1, n//2) * dur / n

    om = gamma * b1p / np.sqrt((1 - gamma * b1p / k * t) ** -2 - 1)
    om = np.concatenate((om, om[::-1]))

    return om
