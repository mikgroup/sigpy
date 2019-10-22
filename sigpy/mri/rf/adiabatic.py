"""Adiabatic Pulse Design functions.

"""
import numpy as np

__all__ = ['bir4', 'hypsec', 'wurst']


def bir4(n, beta, kappa, theta, dw0):
    """Generate a BIR-4 pulse

    Args:
        n (int): number of samples (should be a multiple of 4).
        beta (float): AM waveform parameter.
        kappa (float): FM waveform parameter.
        theta (float): flip angle in radians.
        dw0 (float): FM waveform scaling (radians/s)

    Returns:
        array: AM waveform
        array: FM waveform (radians/s)

    """
    dphi = np.pi + theta / 2

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


def hypsec(n=512, beta=800, mu=4.9, T=0.012):
    t = np.arange(-n // 2, n // 2) / n * T

    a = np.cosh(beta * t) ** (-1)
    om = -mu * beta * np.tanh(beta * t)

    return a, om


def wurst(n=512, N_fac=40, bw=40e3, T=2e-3):
    t = np.arange(0, n) * T / n

    a = 1 - np.power(np.abs(np.cos(np.pi * t / T)), N_fac)
    om = np.linspace(-bw / 2, bw / 2, n) * 2 * np.pi

    return a, om
