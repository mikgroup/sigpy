# -*- coding: utf-8 -*-
"""MRI RF excitation pulse design functions,
    including SLR and small tip spatial design
"""

import numpy as np
import scipy.signal as signal

__all__ = ['dinf', 'dzrf', 'dzls', 'msinc', 'dzmp', 'fmp', 'dzlp',
           'b2rf', 'b2a', 'mag2mp', 'ab2rf', 'dzgSliderB', 'dzgSliderrf',
           'dzb1rf']


""" Functions for SLR pulse design
    SLR algorithm simplifies the solution of the Bloch equations
    to the design of 2 polynomials
    Code from William Grissom, 2019, based on John Pauly's rf_tools
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

    [bsf, d1, d2] = calcRipples(ptype, d1, d2)

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
        raise Exception('Filter type ("{}") is not recognized.'.format(ftype))

    if ptype == 'st':
        rf = b
    else:
        b = bsf*b
        rf = b2rf(b)

    return rf

def calcRipples(ptype = 'st', d1 = 0.01, d2 = 0.01):

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
        raise Exception('Pulse type ("{}") is not recognized.'.format(ptype))

    return bsf, d1, d2

def dzls(N=64, tb=4, d1=0.01, d2=0.01):

    di = dinf(d1, d2)
    w = di/tb
    f = np.asarray([0, (1-w)*(tb/2), (1+w)*(tb/2), (N/2)])/(N/2)
    m = [1, 1, 0, 0]
    w = [1, d1/d2]

    h = signal.firls(N+1, f, m, w)
    # shift the filter half a sample to make it symmetric, like in MATLAB
    c = np.exp(1j*2*np.pi/(2*(N+1)) *
               np.concatenate([np.arange(0, N/2+1, 1),
                               np.arange(-N/2, 0, 1)]))
    h = np.real(np.fft.ifft(np.multiply(np.fft.fft(h), c)))
    # lop off extra sample
    h = h[:N]

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


def dzgSliderB(N=128, G=5, Gind=1, tb=4, d1=0.01, d2=0.01, phi=np.pi,
               shift=32, demod=True):

    ftw = dinf(d1, d2)/tb  # fractional transition width of the slab profile

    if np.fmod(G, 2) and Gind == int(np.ceil(G/2)):  # centered sub-slice
        if G == 1:  # no sub-slices, as a sanity check
            b = dzls(N, tb, d1, d2)
        else:
            # Design 2 filters, to allow arbitrary phases on the subslice the
            # first is a wider notch filter with '0's where it the subslice
            # appears, and the second is the subslice. Multiply the subslice by
            # its phase and add the filters.
            f = np.asarray([0, (1/G-ftw)*(tb/2), (1/G+ftw)*(tb/2),
                            (1-ftw)*(tb/2), (1+ftw)*(tb/2), (N/2)])/(N/2)
            mNotch = [0, 0, 1, 1, 0, 0]
            mSub = [1, 1, 0, 0, 0, 0]
            w = [1, 1, d1/d2]

            bNotch = signal.firls(N+1, f, mNotch, w)  # the notched filter
            bSub = signal.firls(N+1, f, mSub, w)  # the subslice filter
            # add them with the subslice phase
            b = np.add(bNotch, np.multiply(np.exp(1j*phi), bSub))
            # shift the filter half a sample to make it symmetric,
            # like in MATLAB
            c = np.exp(1j*2*np.pi/(2*(N+1)) *
                       np.concatenate([np.arange(0, N/2+1, 1),
                                       np.arange(-N/2, 0, 1)]))
            b = np.fft.ifft(np.multiply(np.fft.fft(b), c))
            # lop off extra sample
            b = b[:N]

    else:
        # design two shifted filters that we can add to kill off the left band,
        # then demodulate the result back to DC
        Gcent = shift+(Gind-G/2-1/2)*tb/G
        if Gind > 1 and Gind < G:
            # separate transition bands for slab+slice
            f = np.asarray([0, shift-(1+ftw)*(tb/2), shift-(1-ftw)*(tb/2),
                Gcent-(tb/G/2+ftw*(tb/2)), Gcent-(tb/G/2-ftw*(tb/2)),
                Gcent+(tb/G/2-ftw*(tb/2)), Gcent+(tb/G/2+ftw*(tb/2)),
                shift+(1-ftw)*(tb/2), shift+(1+ftw)*(tb/2), (N/2)])/(N/2)
            mNotch = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
            mSub = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
            w = [d1/d2, 1, 1, 1, d1/d2]
        elif Gind == 1:
            # the slab and slice share a left transition band
            f = np.asarray([0, shift-(1+ftw)*(tb/2), shift-(1-ftw)*(tb/2),
                           Gcent+(tb/G/2-ftw*(tb/2)), Gcent+(tb/G/2+ftw*(tb/2)),
                           shift+(1-ftw)*(tb/2), shift+(1+ftw)*(tb/2), (N/2)])/(N/2)
            mNotch = [0, 0, 0, 0, 1, 1, 0, 0]
            mSub = [0, 0, 1, 1, 0, 0, 0, 0]
            w = [d1/d2, 1, 1, d1/d2]
        elif Gind == G:
            # the slab and slice share a right transition band
            f = np.asarray([0, shift-(1+ftw)*(tb/2), shift-(1-ftw)*(tb/2),
                           Gcent-(tb/G/2+ftw*(tb/2)), Gcent-(tb/G/2-ftw*(tb/2)),
                           shift+(1-ftw)*(tb/2), shift+(1+ftw)*(tb/2), (N/2)])/(N/2)
            mNotch = [0, 0, 1, 1, 0, 0, 0, 0]
            mSub = [0, 0, 0, 0, 1, 1, 0, 0]
            w = [d1/d2, 1, 1, d1/d2]

        c = np.exp(1j*2*np.pi/(2*(N+1))
                   * np.concatenate([np.arange(0,N/2+1,1), np.arange(-N/2,0,1)]))

        bNotch = signal.firls(N+1, f, mNotch, w)  # the notched filter
        bNotch = np.fft.ifft(np.multiply(np.fft.fft(bNotch), c))
        bNotch = np.real(bNotch[:N])
        if demod == True:
            # hilbert transform to suppress negative passband
            bNotch = signal.hilbert(bNotch)

        bSub = signal.firls(N+1, f, mSub, w)  # the sub-band filter
        bSub = np.fft.ifft(np.multiply(np.fft.fft(bSub), c))
        bSub = np.real(bSub[:N])
        if demod == True:
            # hilbert transform to suppress negative passband
            bSub = signal.hilbert(bSub)

        # add them with the subslice phase
        b = bNotch + np.exp(1j*phi)*bSub

        if demod == True:
            # demodulate to DC
            cShift = np.exp(-1j*2*np.pi/N*shift*np.arange(0, N, 1))/2 \
                * np.exp(-1j*np.pi/N*shift)
            b = np.multiply(b, cShift)

    return b


def dzgSliderrf(N = 256, G = 5, flip = np.pi/2, phi = np.pi, tb = 12,
    d1 = 0.01, d2 = 0.01):

    bsf = np.sin(flip/2) # beta scaling factor

    rf = np.zeros((N, G), dtype = 'complex')
    for Gind in range(1,G+1):
        b = bsf*dzgSliderB(N, G, Gind, tb, d1, d2, phi)
        a = b2a(b)
        rf[:, Gind-1] = ab2rf(a, b)

    return rf

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


def dzb1rf(dt = 2e-6, tb = 4, ptype = 'st', flip = np.pi/6, pbw = 0.3, pbc = 2,
    d1 = 0.01, d2 = 0.01, os = 8):

    # design a B1-selective excitation pulse, following Grissom JMR 2014
    # pbw = width of passband in Gauss
    # pbc = center of passband in Gauss

    # calculate beta filter ripple
    [bsf, d1, d2] = calcRipples(ptype, d1, d2)

    # calculate pulse duration
    B = 4257*pbw
    T = tb/B

    # calculate number of samples in pulse
    n = np.int(np.ceil(T/dt/2)*2)

    if pbc == 0: # we want passband as close to zero as possible.
                 # do my own dual-band filter design to minimize interaction
                 # between the left and right bands

        # build system matrix
        A = np.exp(1j*2*np.pi*np.outer(np.arange(-n*os/2, n*os/2),
            np.arange(-n/2, n/2))/(n*os))

        # build target pattern
        ii = np.arange(-n*os/2, n*os/2)/(n*os)*2
        w = dinf(d1,d2)/tb
        f = np.asarray([0, (1-w)*(tb/2), (1+w)*(tb/2), n/2])/(n/2)
        d = np.double(np.abs(ii) < f[1])
        ds = np.double(np.abs(ii) > f[2])

        # shift the target pattern to minimum center position
        pbc = np.int(np.ceil((f[2]-f[1])*n*os/2 + f[1]*n*os/2))
        dl = np.roll(d, pbc)
        dr = np.roll(d, -pbc)
        dsl = np.roll(ds, pbc)
        dsr = np.roll(ds, -pbc)

        # build error weight vector
        w = dl + dr + d1/d2*np.multiply(dsl, dsr)

        # solve for the dual-band filter
        AtA = A.conj().T @ np.multiply(np.reshape(w,(np.size(w), 1)), A)
        Atd = A.conj().T @ np.multiply(w, dr-dl)
        h = np.imag(np.linalg.pinv(AtA) @ Atd)

    else: # normal design

        # design filter
        h = dzls(n, tb, d1, d2)

        # dual-band-modulate the filter
        om = 2*np.pi*4257*pbc # modulation frequency
        t = np.arange(0, n)*T/n - T/2
        h = 2*h*np.sin(om*t)

    # split and flip fm waveform to improve large-tip accuracy
    dom = np.concatenate((h[n//2::-1], h, h[n:n//2:-1]))/2

    # scale to target flip, convert to Hz
    dom = dom*flip/(2*np.pi*dt)

    # build am waveform
    om1 = np.concatenate((-np.ones(n//2), np.ones(n), -np.ones(n//2)))

    return om1, dom
