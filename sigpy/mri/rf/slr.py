# -*- coding: utf-8 -*-
"""MRI RF excitation pulse design functions,
    including SLR and small tip spatial design
"""

import numpy as np
import scipy.linalg as linalg
import scipy.signal as signal

import sigpy as sp
from sigpy.mri.rf.util import dinf

__all__ = ['dzrf', 'dzls', 'msinc', 'dzmp', 'fmp', 'dzlp',
           'b2rf', 'b2a', 'mag2mp', 'ab2rf', 'dzgSliderB', 'dzgSliderrf',
           'rootFlip', 'dzRecursiveRF']

""" Functions for SLR pulse design
    SLR algorithm simplifies the solution of the Bloch equations
    to the design of 2 polynomials
    Code from William Grissom, 2019, based on John Pauly's rf_tools
"""


def dzrf(N=64, tb=4, ptype='st', ftype='ls', d1=0.01, d2=0.01,
         cancelAlphaPhs=False):
    """Primary function for design of pulses using the SLR algorithm.
        Following functions are to support dzrf

    Args:
        N (int): number of time points.
        tb (int): pulse time bandwidth product.
        ptype (string): type of pulse to be designed.
        ftype (string): type of filter to use in pulse design
        d1 (float): maximum instantaneous power
        d2 (float): maximum average power
        cancelAlphaPhs (bool): For 'ex' pulses, absorb the alpha phase profile
            from beta's profile, so they cancel for a flatter total phase

    References:
        Pauly, J., Le Roux, Patrick., Nishimura, D., and Macovski, A.(1991).
        Parameter Relations for the Shinnar-LeRoux Selective Excitation
        Pulse Design Algorithm.
        IEEE Transactions on Medical Imaging, Vol 10, No 1, 53-65.
    """

    [bsf, d1, d2] = calcRipples(ptype, d1, d2)

    if ftype == 'ms':
        b = msinc(N, tb / 4)
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
    elif ptype == 'ex':
        b = bsf * b
        rf = b2rf(b, cancelAlphaPhs)
    else:
        b = bsf * b
        rf = b2rf(b)

    return rf


def calcRipples(ptype='st', d1=0.01, d2=0.01):
    if ptype == 'st':
        bsf = 1
    elif ptype == 'ex':
        bsf = np.sqrt(1 / 2)
        d1 = np.sqrt(d1 / 2)
        d2 = d2 / np.sqrt(2)
    elif ptype == 'se':
        bsf = 1
        d1 = d1 / 4
        d2 = np.sqrt(d2)
    elif ptype == 'inv':
        bsf = 1
        d1 = d1 / 8
        d2 = np.sqrt(d2 / 2)
    elif ptype == 'sat':
        bsf = np.sqrt(1 / 2)
        d1 = d1 / 2
        d2 = np.sqrt(d2)
    else:
        raise Exception('Pulse type ("{}") is not recognized.'.format(ptype))

    return bsf, d1, d2


def dzls(N=64, tb=4, d1=0.01, d2=0.01):
    di = dinf(d1, d2)
    w = di / tb
    f = np.asarray([0, (1 - w) * (tb / 2), (1 + w) * (tb / 2), (N / 2)])
    f = f / (N / 2)
    m = [1, 1, 0, 0]
    w = [1, d1 / d2]

    h = signal.firls(N + 1, f, m, w)
    # shift the filter half a sample to make it symmetric, like in MATLAB
    c = np.exp(1j * 2 * np.pi / (2 * (N + 1)) *
               np.concatenate([np.arange(0, N / 2 + 1, 1),
                               np.arange(-N / 2, 0, 1)]))
    h = np.real(np.fft.ifft(np.multiply(np.fft.fft(h), c)))
    # lop off extra sample
    h = h[:N]

    return h


def dzmp(N=64, tb=4, d1=0.01, d2=0.01):
    n2 = 2 * N - 1
    di = 0.5 * dinf(2 * d1, 0.5 * d2 * d2)
    w = di / tb
    f = np.asarray([0, (1 - w) * (tb / 2), (1 + w) * (tb / 2), (N / 2)]) / N
    m = [1, 0]
    w = [1, 2 * d1 / (0.5 * d2 * d2)]

    hl = signal.remez(n2, f, m, w)

    h = fmp(hl)

    return h


def fmp(h):
    l = np.size(h)
    lp = 128 * np.exp(np.ceil(np.log(l) / np.log(2)) * np.log(2))
    padwidths = np.array([np.ceil((lp - l) / 2), np.floor((lp - l) / 2)])
    hp = np.pad(h, padwidths.astype(int), 'constant')
    hpf = np.fft.fftshift(np.fft.fft(np.fft.fftshift(hp)))
    hpfs = hpf - np.min(np.real(hpf)) * 1.000001
    hpfmp = mag2mp(np.sqrt(np.abs(hpfs)))
    hpmp = np.fft.ifft(np.fft.fftshift(np.conj(hpfmp)))
    hmp = hpmp[:np.int((l + 1) / 2)]

    return hmp


def dzlp(N=64, tb=4, d1=0.01, d2=0.01):
    di = dinf(d1, d2)
    w = di / tb
    f = np.asarray([0, (1 - w) * (tb / 2), (1 + w) * (tb / 2), (N / 2)]) / N
    m = [1, 0]
    w = [1, d1 / d2]

    h = signal.remez(N, f, m, w)

    return h


def msinc(N=64, m=1):
    x = np.arange(-N / 2, N / 2, 1) / (N / 2)
    snc = np.divide(np.sin(m * 2 * np.pi * x + 0.00001),
                    (m * 2 * np.pi * x + 0.00001))
    ms = np.multiply(snc, 0.54 + 0.46 * np.cos(np.pi * x))
    ms = ms * 4 * m / N

    return ms


def dzgSliderB(N=128, G=5, Gind=1, tb=4, d1=0.01, d2=0.01,
               phi=np.pi, shift=32):
    ftw = dinf(d1, d2) / tb  # fractional transition width of the slab profile

    if np.fmod(G, 2) and Gind == int(np.ceil(G / 2)):  # centered sub-slice
        if G == 1:  # no sub-slices, as a sanity check
            b = dzls(N, tb, d1, d2)
        else:
            # Design 2 filters, to allow arbitrary phases on the subslice the
            # first is a wider notch filter with '0's where it the subslice
            # appears, and the second is the subslice. Multiply the subslice by
            # its phase and add the filters.
            f = np.asarray([0, (1 / G - ftw) * (tb / 2),
                            (1 / G + ftw) * (tb / 2),
                            (1 - ftw) * (tb / 2),
                            (1 + ftw) * (tb / 2),
                            (N / 2)])
            f = f / (N / 2)
            mNotch = [0, 0, 1, 1, 0, 0]
            mSub = [1, 1, 0, 0, 0, 0]
            w = [1, 1, d1 / d2]

            bNotch = signal.firls(N + 1, f, mNotch, w)  # the notched filter
            bSub = signal.firls(N + 1, f, mSub, w)  # the subslice filter
            # add them with the subslice phase
            b = np.add(bNotch, np.multiply(np.exp(1j * phi), bSub))
            # shift the filter half a sample to make it symmetric,
            # like in MATLAB
            c = np.exp(1j * 2 * np.pi / (2 * (N + 1)) *
                       np.concatenate([np.arange(0, N / 2 + 1, 1),
                                       np.arange(-N / 2, 0, 1)]))
            b = np.fft.ifft(np.multiply(np.fft.fft(b), c))
            # lop off extra sample
            b = b[:N]

    else:
        # design filters for the slab and the subslice, hilbert xform them
        # to suppress their left bands,
        # then demodulate the result back to DC
        Gcent = shift + (Gind - G / 2 - 1 / 2) * tb / G
        if Gind > 1 and Gind < G:
            # separate transition bands for slab+slice
            f = np.asarray([0, shift - (1 + ftw) * (tb / 2),
                            shift - (1 - ftw) * (tb / 2),
                            Gcent - (tb / G / 2 + ftw * (tb / 2)),
                            Gcent - (tb / G / 2 - ftw * (tb / 2)),
                            Gcent + (tb / G / 2 - ftw * (tb / 2)),
                            Gcent + (tb / G / 2 + ftw * (tb / 2)),
                            shift + (1 - ftw) * (tb / 2),
                            shift + (1 + ftw) * (tb / 2), (N / 2)])
            f = f / (N / 2)
            mNotch = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
            mSub = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
            w = [d1 / d2, 1, 1, 1, d1 / d2]
        elif Gind == 1:
            # the slab and slice share a left transition band
            f = np.asarray([0, shift - (1 + ftw) * (tb / 2),
                            shift - (1 - ftw) * (tb / 2),
                            Gcent + (tb / G / 2 - ftw * (tb / 2)),
                            Gcent + (tb / G / 2 + ftw * (tb / 2)),
                            shift + (1 - ftw) * (tb / 2),
                            shift + (1 + ftw) * (tb / 2),
                            (N / 2)])
            f = f / (N / 2)
            mNotch = [0, 0, 0, 0, 1, 1, 0, 0]
            mSub = [0, 0, 1, 1, 0, 0, 0, 0]
            w = [d1 / d2, 1, 1, d1 / d2]
        elif Gind == G:
            # the slab and slice share a right transition band
            f = np.asarray([0, shift - (1 + ftw) * (tb / 2),
                            shift - (1 - ftw) * (tb / 2),
                            Gcent - (tb / G / 2 + ftw * (tb / 2)),
                            Gcent - (tb / G / 2 - ftw * (tb / 2)),
                            shift + (1 - ftw) * (tb / 2),
                            shift + (1 + ftw) * (tb / 2),
                            (N / 2)])
            f = f / (N / 2)
            mNotch = [0, 0, 1, 1, 0, 0, 0, 0]
            mSub = [0, 0, 0, 0, 1, 1, 0, 0]
            w = [d1 / d2, 1, 1, d1 / d2]

        c = np.exp(1j * 2 * np.pi / (2 * (N + 1))
                   * np.concatenate([np.arange(0, N / 2 + 1, 1),
                                     np.arange(-N / 2, 0, 1)]))

        bNotch = signal.firls(N + 1, f, mNotch, w)  # the notched filter
        bNotch = np.fft.ifft(np.multiply(np.fft.fft(bNotch), c))
        bNotch = np.real(bNotch[:N])
        # hilbert transform to suppress negative passband
        bNotch = signal.hilbert(bNotch)

        bSub = signal.firls(N + 1, f, mSub, w)  # the sub-band filter
        bSub = np.fft.ifft(np.multiply(np.fft.fft(bSub), c))
        bSub = np.real(bSub[:N])
        # hilbert transform to suppress negative passband
        bSub = signal.hilbert(bSub)

        # add them with the subslice phase
        b = bNotch + np.exp(1j * phi) * bSub

        # demodulate to DC
        cShift = np.exp(-1j * 2 * np.pi / N * shift * np.arange(0, N, 1)) / 2 \
            * np.exp(-1j * np.pi / N * shift)
        b = np.multiply(b, cShift)

    return b


def dzHadamardB(N=128, G=5, Gind=1, tb=4, d1=0.01, d2=0.01, shift=32):
    H = linalg.hadamard(G)
    encode = H[Gind - 1, :]

    ftw = dinf(d1, d2) / tb  # fractional transition width of the slab profile

    if Gind == 1:  # no sub-slices
        b = dzls(N, tb, d1, d2)
    else:
        # left stopband
        f = np.asarray([0, shift - (1 + ftw) * (tb / 2)])
        m = np.asarray([0, 0])
        w = np.asarray([d1 / d2])
        # first sub-band
        ii = 1
        Gcent = shift + (ii - G / 2 - 1 / 2) * tb / G  # first band center
        # first band left edge
        f = np.append(f, Gcent - (tb / G / 2 - ftw * (tb / 2)))
        m = np.append(m, encode[ii - 1])
        if encode[ii - 1] != encode[ii]:
            # add the first band's right edge and its amplitude, and a weight
            f = np.append(f, Gcent + (tb / G / 2 - ftw * (tb / 2)))
            m = np.append(m, encode[ii - 1])
            w = np.append(w, 1)
        # middle sub-bands
        for ii in range(2, G):
            Gcent = shift + (ii - G / 2 - 1 / 2) * tb / G  # center of band
            if encode[ii - 1] != encode[ii - 2]:
                # add a left edge and amp for this band
                f = np.append(f, Gcent - (tb / G / 2 - ftw * (tb / 2)))
                m = np.append(m, encode[ii - 1])
            if encode[ii - 1] != encode[ii]:
                # add a right edge and its amp, and a weight for this band
                f = np.append(f, Gcent + (tb / G / 2 - ftw * (tb / 2)))
                m = np.append(m, encode[ii - 1])
                w = np.append(w, 1)
        # last sub-band
        ii = G
        Gcent = shift + (ii - G / 2 - 1 / 2) * tb / G  # center of last band
        if encode[ii - 1] != encode[ii - 2]:
            # add a left edge and amp for the last band
            f = np.append(f, Gcent - (tb / G / 2 - ftw * (tb / 2)))
            m = np.append(m, encode[ii - 1])
        # add a right edge and its amp, and a weight for the last band
        f = np.append(f, Gcent + (tb / G / 2 - ftw * (tb / 2)))
        m = np.append(m, encode[ii - 1])
        w = np.append(w, 1)
        # right stop-band
        f = np.append(f, (shift + (1 + ftw) * (tb / 2), (N / 2))) / (N / 2)
        m = np.append(m, [0, 0])
        w = np.append(w, d1 / d2)

        # separate the positive and negative bands
        mp = (m > 0).astype(float)
        mn = (m < 0).astype(float)

        # design the positive and negative filters
        c = np.exp(1j * 2 * np.pi / (2 * (N + 1))
                   * np.concatenate([np.arange(0, N / 2 + 1, 1),
                                     np.arange(-N / 2, 0, 1)]))
        bp = signal.firls(N + 1, f, mp, w)  # the positive filter
        bn = signal.firls(N + 1, f, mn, w)  # the negative filter

        # combine the filters and demodulate
        b = np.fft.ifft(np.multiply(np.fft.fft(bp - bn), c))
        b = np.real(b[:N])
        # hilbert transform to suppress negative passband
        b = signal.hilbert(b)
        # demodulate to DC
        cShift = np.exp(-1j * 2 * np.pi / N * shift * np.arange(0, N, 1)) / 2 \
            * np.exp(-1j * np.pi / N * shift)
        b = np.multiply(b, cShift)

    return b


def dzgSliderrf(N=256, G=5, flip=np.pi / 2, phi=np.pi, tb=12,
                d1=0.01, d2=0.01, cancelAlphaPhs=True):
    bsf = np.sin(flip / 2)  # beta scaling factor

    rf = np.zeros((N, G), dtype='complex')
    for Gind in range(1, G + 1):
        b = bsf * dzgSliderB(N, G, Gind, tb, d1, d2, phi)
        rf[:, Gind - 1] = b2rf(b, cancelAlphaPhs)

    return rf


def b2rf(b, cancelAlphaPhs=False):
    a = b2a(b)
    if cancelAlphaPhs:
        b = np.fft.ifft(np.fft.fft(b) *
                        np.exp(-1j * np.angle(np.fft.fft(a[np.size(a)::-1]))))
    rf = ab2rf(a, b)

    return rf


def b2a(b):
    N = np.size(b)

    Npad = N * 16
    bcp = np.zeros(Npad, dtype=complex)
    bcp[0:N:1] = b
    bf = np.fft.fft(bcp)
    bfmax = np.max(np.abs(bf))
    if bfmax >= 1:
        bf = bf / (1e-7 + bfmax)
    afa = mag2mp(np.sqrt(1 - np.abs(bf) ** 2))
    a = np.fft.fft(afa) / Npad
    a = a[0:N:1]
    a = a[::-1]

    return a


def mag2mp(x):
    N = np.size(x)
    xl = np.log(np.abs(x))  # Log of mag spectrum
    xlf = np.fft.fft(xl)
    xlfp = xlf
    xlfp[0] = xlf[0]  # Keep DC the same
    xlfp[1:(N // 2):1] = 2 * xlf[1:(N // 2):1]  # Double positive frequencies
    xlfp[N // 2] = xlf[N // 2]  # keep half Nyquist the same
    xlfp[N // 2 + 1:N:1] = 0  # zero negative frequencies
    xlaf = np.fft.ifft(xlfp)
    a = np.exp(xlaf)  # complex exponentiation

    return a


def ab2rf(a, b):
    N = np.size(a)
    rf = np.zeros(N, dtype=complex)

    a = a.astype(complex)
    b = b.astype(complex)

    for ii in range(N - 1, -1, -1):

        Cj = np.sqrt(1 / (1 + np.abs(b[ii] / a[ii]) ** 2))
        Sj = np.conj(Cj * b[ii] / a[ii])
        theta = np.arctan2(np.abs(Sj), Cj)
        psi = np.angle(Sj)
        rf[ii] = 2 * theta * np.exp(1j * psi)

        # remove this rotation from polynomials
        if ii > 0:
            at = Cj * a + Sj * b
            bt = -np.conj(Sj) * a + Cj * b
            a = at[1:ii + 1:1]
            b = bt[0:ii:1]

    return rf


def rootFlip(b, d1, flip, tb):
    # exhaustive root-flip pattern search for min-peak b1

    n = np.size(b)
    [w, bResp] = signal.freqz(b)
    b /= np.max(np.abs(bResp))  # normalize beta
    b *= np.sin(flip / 2 + np.arctan(d1 * 2) / 2)  # scale to target flip
    r = sp.util.leja_fast(np.roots(b))

    candidates = np.logical_and(np.abs(1 - np.abs(r)) > 0.004,
                                np.abs(np.angle(r)) < tb / n * np.pi)

    iiMin = 0
    iiMax = 2 ** np.sum(candidates)

    maxRF = np.max(np.abs(b2rf(b)))

    for ii in range(iiMin, iiMax):

        if ii % 20 == 0:
            print('Evaluating root-flip pattern ' + str(ii) + ' out of ' +
                  str(iiMax - iiMin))

        # get a binary flipping pattern
        doFlipStr = format(ii, 'b')
        doFlip = np.zeros(np.sum(candidates), dtype=bool)
        for jj in range(0, len(doFlipStr)):
            doFlip[jj] = bool(int(doFlipStr[jj]))

        # embed the pattern in an all-roots vector
        tmp = np.zeros(n - 1, dtype=bool)
        tmp[candidates] = doFlip
        doFlip = tmp

        # flip those indices
        rFlip = np.zeros(np.shape(r), dtype=complex)
        rFlip[:] = r[:]
        rFlip[doFlip] = np.conj(1 / rFlip[doFlip])

        bTmp = np.poly(rFlip)
        [w, bTmpResp] = signal.freqz(bTmp)
        bTmp /= np.max(np.abs(bTmpResp))  # normalize beta
        bTmp *= np.sin(flip / 2 + np.arctan(d1 * 2) / 2)  # scale to targ. flip
        rfTmp = b2rf(bTmp)

        if np.max(np.abs(rfTmp)) < maxRF:
            maxRF = np.max(np.abs(rfTmp))
            rf_out = rfTmp
            b_out = bTmp

    return rf_out, b_out


def dzRecursiveRF(Nseg, tb, N, seSeq=False, tbRef=8, zPadFact=4,
                  winFact=1.75, cancelAlphaPhs=True, T1=np.inf, TRseg=60,
                  useMz=True, d1=0.01, d2=0.01, d1se=0.01, d2se=0.01):
    # get refocusing pulse and its rotation parameters
    if seSeq is True:
        [bsf, d1se, d2se] = calcRipples('se', d1se, d2se)
        bRef = bsf * dzls(N, tbRef, d1se, d2se)
        bRef = np.concatenate((np.zeros(int(zPadFact * N / 2 - N / 2)), bRef,
                               np.zeros(int(zPadFact * N / 2 - N / 2))))
        rfRef = b2rf(bRef)
        Bref = ft(bRef)
        Bref /= np.max(np.abs(Bref))
        BrefMag = np.abs(Bref)
        ArefMag = np.abs(np.sqrt(1 - BrefMag ** 2))
        flipRef = 2 * np.arcsin(BrefMag[int(zPadFact * N / 2)]) * 180 / np.pi

    # get flip angles
    flip = np.zeros(Nseg)
    flip[Nseg - 1] = 90
    for jj in range(Nseg - 2, -1, -1):
        if seSeq is False:
            flip[jj] = np.arctan(np.sin(flip[jj + 1] * np.pi / 180))
            flip[jj] = flip[jj] * 180 / np.pi  # deg
        else:
            flip[jj] = np.arctan(np.cos(flipRef * np.pi / 180) *
                                 np.sin(flip[jj + 1] * np.pi / 180))
            flip[jj] = flip[jj] * 180 / np.pi  # deg

    # design first RF pulse
    b = np.zeros((int(zPadFact * N), Nseg), dtype=complex)
    b[int(zPadFact * N / 2 - N / 2):int(zPadFact * N / 2 + N / 2), 0] = \
        dzls(N, tb, d1, d2)
    # b = np.concatenate((np.zeros(int(zPadFact*N/2-N/2)), b,
    #    np.zeros(int(zPadFact*N/2-N/2))))
    B = ft(b[:, 0])
    c = np.exp(-1j * 2 * np.pi / (N * zPadFact) / 2 *
               np.arange(-N * zPadFact / 2, N * zPadFact / 2, 1))
    B = np.multiply(B, c)
    b[:, 0] = ift(B / np.max(np.abs(B)))
    b[:, 0] *= np.sin(flip[0] * (np.pi / 180) / 2)
    rf = np.zeros((zPadFact * N, Nseg), dtype=complex)
    a = b2a(b[:, 0])
    if cancelAlphaPhs:
        # cancel a phase by absorbing into b
        # Note that this is the only time we need to do it
        b[:, 0] = np.fft.ifft(np.fft.fft(b[:, 0]) *
                              np.exp(-1j *
                                     np.angle(np.fft.fft(a[np.size(a)::-1]))))
    rf[:, 0] = b2rf(b[:, 0])

    # get min-phase alpha and its response
    # a = b2a(b[:, 0])
    A = sp.fft(a)

    # calculate beta filter response
    B = ft(b[:, 0])

    if winFact < zPadFact:
        winLen = (winFact - 1) * N
        Npad = N * zPadFact - winFact * N
        # blackman window?
        window = signal.blackman(int((winFact - 1) * N))
        # split in half; stick N ones in the middle
        window = np.concatenate((window[0:int(winLen / 2)], np.ones(N),
                                 window[int(winLen / 2):]))
        window = np.concatenate((np.zeros(int(Npad / 2)), window,
                                 np.zeros(int(Npad / 2))))
        # apply windowing to first pulse for consistency
        b[:, 0] = np.multiply(b[:, 0], window)
        rf[:, 0] = b2rf(b[:, 0])
        # recalculate B and A
        B = ft(b[:, 0])
        A = ft(b2a(b[:, 0]))

    # use A and B to get Mxy
    # Mxy = np.zeros((zPadFact*N, Nseg), dtype = complex)
    if seSeq is False:
        Mxy0 = 2 * np.conj(A) * B
    else:
        Mxy0 = 2 * A * np.conj(B) * Bref ** 2

    # Amplitude of next pulse's Mxy profile will be
    #               |Mz*2*a*b| = |Mz*2*sqrt(1-abs(B).^2)*B|.
    # If we set this = |Mxy_1|, we can solve for |B| via solving quadratic
    # equation 4*Mz^2*(1-B^2)*B^2 = |Mxy_1|^2.
    # Subsequently solve for |A|, and get phase of A via min-phase, and
    # then get phase of B by dividing phase of A from first pulse's Mxy phase.
    Mz = np.ones((zPadFact * N), dtype=complex)
    for jj in range(1, Nseg):

        # calculate Mz profile after previous pulse
        if seSeq is False:
            Mz = Mz * (1 - 2 * np.abs(B) ** 2) * np.exp(-TRseg / T1) + \
                 (1 - np.exp(-TRseg / T1))
        else:
            Mz = Mz * (1 - 2 * (np.abs(A * BrefMag) ** 2
                                + np.abs(ArefMag * B) ** 2))
            # (second term is about 1%)

        if useMz is True:  # design the pulses accounting for the
            # actual Mz profile (the full method)
            # set up quadratic equation to get |B|
            cq = -np.abs(Mxy0) ** 2
            if seSeq is False:
                bq = 4 * Mz ** 2
                aq = -4 * Mz ** 2
            else:
                bq = 4 * (BrefMag ** 4) * Mz ** 2
                aq = -4 * (BrefMag ** 4) * Mz ** 2
            Bmag = np.sqrt((-bq + np.real(np.sqrt(bq ** 2 - 4 * aq * cq)))
                           / (2 * aq))
            Bmag[np.isnan(Bmag)] = 0
            # get A - easier to get complex A than complex B since |A| is
            # determined by |B|, and phase is gotten by min-phase relationship
            # Phase of B doesn't matter here since only profile mag is used by
            # b2a
            A = ft(b2a(ift(Bmag)))
            # trick: now we can get complex B from ratio of Mxy and A
            B = Mxy0 / (2 * np.conj(A) * Mz)

        else:  # design assuming ideal Mz (conventional VFA)

            B *= np.sin(np.pi / 180 * flip[jj] / 2) \
                 / np.sin(np.pi / 180 * flip[jj - 1] / 2)
            A = ft(b2a(ift(B)))

        # get polynomial
        b[:, jj] = ift(B)

        if winFact < zPadFact:
            b[:, jj] *= window
            # recalculate B and A
            B = ft(b[:, jj])
            A = ft(b2a(b[:, jj]))

        rf[:, jj] = b2rf(b[:, jj])

    # truncate the RF
    if winFact < zPadFact:
        pulseLen = int(winFact * N)
        rf = rf[int(Npad / 2):int(Npad / 2 + pulseLen), :]

    if seSeq is False:
        return rf
    else:
        return rf, rfRef


def ft(x):
    X = np.fft.fftshift(np.fft.fft(np.fft.fftshift(x)))

    return X


def ift(X):
    x = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(X)))

    return x
