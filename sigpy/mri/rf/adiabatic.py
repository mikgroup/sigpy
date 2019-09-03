"""Adiabatic Pulse Design functions.

"""
import numpy as np

__all__ = ['bir4', 'hypsec']


def bir4(n, beta, kappa, theta, dw0):

    # genbir4: generate a BIR-4 pulse
    # In:
    #     n: number of samples (should be multiple of 4)
    #     beta: AM waveform parameter
    #     kappa: FM waveform parameter
    #     theta: flip angle in radians
    #     dw0: FM waveform scaling (radians/s)
    #
    # Out:
    #     a: AM waveform
    #     om: FM waveform (radians/s)

    dphi = np.pi + theta/2

    t = np.arange(0,n)/n

    a1 = np.tanh(beta*(1-4*t[:n//4]))
    a2 = np.tanh(beta*(4*t[n//4:n//2]-1))
    a3 = np.tanh(beta*(3-4*t[n//2:3*n//4]))
    a4 = np.tanh(beta*(4*t[3*n//4:]-3))

    a = np.concatenate((a1, a2, a3, a4)).astype(complex)
    a[n//4:3*n//4] = a[n//4:3*n//4]*np.exp(1j*dphi)

    om1 = dw0*np.tan(kappa*4*t[:n//4])/np.tan(kappa)
    om2 = dw0*np.tan(kappa*(4*t[n//4:n//2]-2))/np.tan(kappa)
    om3 = dw0*np.tan(kappa*(4*t[n//2:3*n//4]-2))/np.tan(kappa)
    om4 = dw0*np.tan(kappa*(4*t[3*n//4:]-4))/np.tan(kappa)

    om = np.concatenate((om1, om2, om3, om4))

    return a, om

def hypsec(n = 512, beta = 800, mu = 4.9, T = 0.012):

    t = np.arange(-n//2,n//2)/n*T

    a = np.cosh(beta*t)**(-1)
    om = -mu*beta*np.tanh(beta*t)

    return a, om