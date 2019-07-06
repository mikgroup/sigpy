# -*- coding: utf-8 -*-
"""MRI RF excitation pulse design functions,
    including SLR and small tip spatial design
"""

import numpy as np
import sigpy as sp
import scipy.signal as signal

from sigpy.mri import linop

__all__ = ['stspa', 'dinf', 'dzrf', 'dzls', 'msinc', 'dzmp', 'fmp', 'dzlp',
           'b2rf', 'b2a', 'mag2mp', 'ab2rf', 'abrm', 'abrmnd']


def stspa(target, sens, coord=None, mask=None, pinst=float('inf'),
          pavg=float('inf'), max_iter=1000, tol=1E-6):
    """Small tip spatial domain method for multicoil parallel excitation.
       Allows for constrained or unconstrained designs.

    Args:
        target (array): desired magnetization profile.
        sens (array): sensitivity maps.
        coord (array): coordinates for noncartesian trajectories
        mask (array): kspace sampling mask for cartesian patterns only
        pinst (float): maximum instantaneous power
        pavg (float): maximum average power
        max_iter (int): max number of iterations
        tol (float): allowable error

    References:
        Grissom, W., Yip, C., Zhang, Z., Stenger, V. A., Fessler, J. A.
        & Noll, D. C.(2006).
        Spatial Domain Method for the Design of RF Pulses in Multicoil
        Parallel Excitation. Magnetic resonance in medicine, 56, 620-629.
    """

    A = linop.Sense(sens, coord, mask, ishape=target.shape).H

    if coord is not None:
        # Nc*Nt pulses
        pulses = np.zeros((sens.shape[0], coord.shape[0]), np.complex)
    else:
        pulses = np.zeros(sens.shape, np.complex)

    u = np.zeros(target.shape, np.complex)

    lipschitz = np.linalg.svd(A * A.H * np.ones(target.shape, np.complex),
                              compute_uv=False)[0]
    tau = 1.0 / lipschitz
    sigma = 0.01
    lamda = 0.01

    # build proxg, includes all constraints:
    def proxg(alpha, pulses):
        # instantaneous power constraint
        func = (pulses / (1 + lamda * alpha)) * \
               np.minimum(pinst/np.abs(pulses) ** 2, 1)
        # avg power constraint for each of Nc channels
        for i in range(pulses.shape[0]):
            func[i] *= np.minimum(pavg/((np.linalg.norm(func[i], 2, axis=0) ** 2)/len(pulses[i])), 1)

        return func

    alg_method = sp.alg.PrimalDualHybridGradient(
        lambda alpha, u: (u - alpha * target) / (1 + alpha),
        lambda alpha, pulses: proxg(alpha, pulses),
        lambda pulses: A * pulses,
        lambda pulses: A.H * pulses,
        pulses, u, tau, sigma, max_iter=max_iter, tol=tol)

    while not alg_method.done():
        alg_method.update()

    return pulses


""" Functions for use in SLR pulse design
    SLR algorithm simplifies the solution of the Bloch equations
    to the design of 2 polynomials
    Code from William Grissom, 2019
"""


def dinf(d1=0.01, d2=0.01):
    a1 = 5.309e-3
    a2 = 7.114e-2
    a3 = -4.761e-1
    a4 = -2.66e-3
    a5 = -5.941e-1
    a6 = -4.278e-1

    l10d1 = np.log10(d1)
    l10d2 = np.log10(d2)

    d = (a1*l10d1*l10d1+a2*l10d1+a3)*l10d2+(a4*l10d1*l10d1+a5*l10d1+a6)

    return d


def dzrf(N=64, tb=4, ptype='st', ftype='ls', d1=0.01, d2=0.01):
    """Primary function for design of pulses using the SLR algorithm.
        Following functions are to support dzrf

    Args:
        N (int): number of time points.
        tb (int): pulse time bandwidth product.
        ptype (string): type of pulse to be designed.
        ftype (string): type of filter to use in pulse design
        d1 (float): maximum instantaneous power
        d2 (float): maximum average power

    References:
        Pauly, J., Le Roux, Patrick., Nishimura, D., and Macovski, A.(1991).
        Parameter Relations for the Shinnar-LeRoux Selective Excitation
        Pulse Design Algorithm.
        IEEE Transactions on Medical Imaging, Vol 10, No 1, 53-65.
    """

    if ptype == 'st':
        bsf = 1
    elif ptype == 'ex':
        bsf = np.sqrt(1/2)
        d1 = np.sqrt(d1/2)
        d2 = d2/np.sqrt(2)
    elif ptype == 'se':
        bsf = 1
        d1 = d1/4
        d2 = np.sqrt(d2)
    elif ptype == 'inv':
        bsf = 1
        d1 = d1/8
        d2 = np.sqrt(d2/2)
    elif ptype == 'sat':
        bsf = np.sqrt(1/2)
        d1 = d1/2
        d2 = np.sqrt(d2)
    else:
        raise Exception('The pulse type ("{}") cannot be identified.'.format(ptype))

    if ftype == 'ms':
        b = msinc(N, tb/4)
    elif ftype == 'pm':
        b = dzlp(N, tb, d1, d2)
    elif ftype == 'min':
        b = dzmp(N, tb, d1, d2)
        b = b[::-1]
    elif ftype == 'max':
        b = dzmp(N, tb, d1, d2)
    elif ftype == 'ls':
        b = dzls(N, tb, d1, d2)
    else:
        raise Exception('The filter type ("{}") cannot be identified.'.format(ftype))

    if ptype == 'st':
        rf = b
    else:
        b = bsf*b
        rf = b2rf(b)

    return rf


def dzls(N=64, tb=4, d1=0.01, d2=0.01):

    di = dinf(d1, d2)
    w = di/tb
    f = np.asarray([0, (1-w)*(tb/2), (1+w)*(tb/2), (N/2)])/(N/2)
    m = [1, 1, 0, 0]
    w = [1, d1/d2]

    h = signal.firls(N+1, f, m, w)
    # shift the filter half a sample to make it symmetric, like in MATLAB
    H = np.fft.fft(h)
    H = np.multiply(H, np.exp(1j*2*np.pi/(2*(N+1)) *
                              np.concatenate([np.arange(0, N/2+1, 1), np.arange(-N/2, 0, 1)])))
    h = np.fft.ifft(H)
    # lop off extra sample
    h = np.real(h[:N])

    return h


def dzmp(N=64, tb=4, d1=0.01, d2=0.01):

    n2 = 2*N-1
    di = 0.5*dinf(2*d1, 0.5*d2*d2)
    w = di/tb
    f = np.asarray([0, (1-w)*(tb/2), (1+w)*(tb/2), (N/2)])/N
    m = [1, 0]
    w = [1, 2*d1/(0.5*d2*d2)]

    hl = signal.remez(n2, f, m, w)

    h = fmp(hl)

    return h


def fmp(h):

    l = np.size(h)
    lp = 128*np.exp(np.ceil(np.log(l)/np.log(2))*np.log(2))
    padwidths = np.array([np.ceil((lp-l)/2), np.floor((lp-l)/2)])
    hp = np.pad(h, padwidths.astype(int), 'constant')
    hpf = np.fft.fftshift(np.fft.fft(np.fft.fftshift(hp)))
    hpfs = hpf - np.min(np.real(hpf))*1.000001
    hpfmp = mag2mp(np.sqrt(np.abs(hpfs)))
    hpmp = np.fft.ifft(np.fft.fftshift(np.conj(hpfmp)))
    hmp = hpmp[:np.int((l+1)/2)]

    return hmp


def dzlp(N=64, tb=4, d1=0.01, d2=0.01):

    di = dinf(d1, d2)
    w = di/tb
    f = np.asarray([0, (1-w)*(tb/2), (1+w)*(tb/2), (N/2)])/(N)
    m = [1, 0]
    w = [1, d1/d2]

    h = signal.remez(N, f, m, w)

    return h


def msinc(N=64, m=1):
    x = np.arange(-N/2, N/2, 1)/(N/2)
    snc = np.divide(np.sin(m*2*np.pi*x+0.00001), (m*2*np.pi*x+0.00001))
    ms = np.multiply(snc, 0.54+0.46*np.cos(np.pi*x))
    ms = ms*4*m/N

    return ms


def b2rf(b):

    a = b2a(b)
    rf = ab2rf(a, b)
    return rf


def b2a(b):

    N = np.size(b)

    Npad = N*16
    bcp = np.zeros(Npad, dtype=complex)
    bcp[0:N:1] = b
    bf = np.fft.fft(bcp)
    bfmax = np.max(np.abs(bf))
    if bfmax >= 1:
        bf = bf/(1e-7 + bfmax)
    afa = mag2mp(np.sqrt(1-np.abs(bf)**2))
    a = np.fft.fft(afa)/Npad
    a = a[0:N:1]
    a = a[::-1]

    return a


def mag2mp(x):

    N = np.size(x)
    xl = np.log(np.abs(x))  # Log of mag spectrum
    xlf = np.fft.fft(xl)
    xlfp = xlf
    xlfp[0] = xlf[0]        # Keep DC the same
    xlfp[1:(N//2):1] = 2*xlf[1:(N//2):1]  # Double positive frequencies
    xlfp[N//2] = xlf[N//2]    # keep half Nyquist the same
    xlfp[N//2+1:N:1] = 0     # zero negative frequencies
    xlaf = np.fft.ifft(xlfp)
    a = np.exp(xlaf)        # complex exponentiation

    return a


def ab2rf(a, b):

    N = np.size(a)
    rf = np.zeros(N, dtype=complex)

    a = a.astype(complex)
    b = b.astype(complex)

    for ii in range(N-1, -1, -1):

        Cj = np.sqrt(1/(1+np.abs(b[ii]/a[ii])**2))
        Sj = np.conj(Cj*b[ii]/a[ii])
        theta = np.arctan2(np.abs(Sj), Cj)
        psi = np.angle(Sj)
        rf[ii] = 2*theta*np.exp(1j*psi)

        # remove this rotation from polynomials
        if ii > 0:
            at = Cj*a + Sj*b
            bt = -np.conj(Sj)*a + Cj*b
            a = at[1:ii+1:1]
            b = bt[0:ii:1]

    return rf


def abrm(rf, x):

    # Simulation of the RF pulse, with simultaneous RF + gradient rotations
    g = np.ones(np.size(rf))*2*np.pi/np.size(rf)

    a = np.ones(np.size(x), dtype=complex)
    b = np.zeros(np.size(x), dtype=complex)
    for mm in range(0, np.size(rf), 1):
        om = x*g[mm]
        phi = np.sqrt(np.abs(rf[mm])**2 + om**2)
        n = np.column_stack((np.real(rf[mm])/phi, np.imag(rf[mm])/phi, om/phi))
        av = np.cos(phi/2) - 1j*n[:, 2]*np.sin(phi/2)
        bv = -1j*(n[:, 0] + 1j*n[:, 1])*np.sin(phi/2)
        at = av*a - np.conj(bv)*b
        bt = bv*a + np.conj(av)*b
        a = at
        b = bt

    return a, b


def abrmnd(rf, x, g):

    # assume x has inverse spatial units of g, and that g has gamma*dt already applied
    # assume x = [...,Ndim], g = [Ndim,Nt]

    a = np.ones(np.shape(x)[0], dtype=complex)
    b = np.zeros(np.shape(x)[0], dtype=complex)
    for mm in range(0, np.size(rf), 1):
        om = x@g[mm, :]
        phi = np.sqrt(np.abs(rf[mm])**2 + om**2)
        n = np.column_stack((np.real(rf[mm])/phi, np.imag(rf[mm])/phi, om/phi))
        av = np.cos(phi/2) - 1j*n[:, 2]*np.sin(phi/2)
        bv = -1j*(n[:, 0] + 1j*n[:, 1])*np.sin(phi/2)
        at = av*a - np.conj(bv)*b
        bt = bv*a + np.conj(av)*b
        a = at
        b = bt

    return a, b
