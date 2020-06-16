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
           'b2rf', 'b2a', 'mag2mp', 'ab2rf', 'dz_gslider_b', 'dz_gslider_rf',
           'root_flip', 'dz_recursive_rf', 'dz_hadamard_b', 'calc_ripples']

""" Functions for SLR pulse design
    SLR algorithm simplifies the solution of the Bloch equations
    to the design of 2 polynomials
"""


def dzrf(n=64, tb=4, ptype='st', ftype='ls', d1=0.01, d2=0.01,
         cancel_alpha_phs=False):
    r"""Primary function for design of pulses using the SLR algorithm.

    Args:
        n (int): number of time points.
        tb (int): pulse time bandwidth product.
        ptype (string): pulse type, 'st' (small-tip excitation), 'ex' (pi/2
            excitation pulse), 'se' (spin-echo pulse), 'inv' (inversion), or
            'sat' (pi/2 saturation pulse).
        ftype (string): type of filter to use: 'ms' (sinc), 'pm'
            (Parks-McClellan equal-ripple), 'min' (minphase using factored pm),
            'max' (maxphase using factored pm), 'ls' (least squares).
        d1 (float): passband ripple level in :math:'M_0^{-1}'.
        d2 (float): stopband ripple level in :math:'M_0^{-1}'.
        cancel_alpha_phs (bool): For 'ex' pulses, absorb the alpha phase
            profile from beta's profile, so they cancel for a flatter
            total phase

    Returns:
        rf (array): designed RF pulse.

    References:
        Pauly, J., Le Roux, Patrick., Nishimura, D., and Macovski, A.(1991).
        Parameter Relations for the Shinnar-LeRoux Selective Excitation
        Pulse Design Algorithm.
        IEEE Transactions on Medical Imaging, Vol 10, No 1, 53-65.
    """

    [bsf, d1, d2] = calc_ripples(ptype, d1, d2)

    if ftype == 'ms':  # sinc
        b = msinc(n, tb / 4)
    elif ftype == 'pm':  # linphase
        b = dzlp(n, tb, d1, d2)
    elif ftype == 'min':  # minphase
        b = dzmp(n, tb, d1, d2)
        b = b[::-1]
    elif ftype == 'max':  # maxphase
        b = dzmp(n, tb, d1, d2)
    elif ftype == 'ls':  # least squares
        b = dzls(n, tb, d1, d2)
    else:
        raise Exception('Filter type ("{}") is not recognized.'.format(ftype))

    if ptype == 'st':
        rf = b
    elif ptype == 'ex':
        b = bsf * b
        rf = b2rf(b, cancel_alpha_phs)
    else:
        b = bsf * b
        rf = b2rf(b)

    return rf


def calc_ripples(ptype='st', d1=0.01, d2=0.01):
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

# following functions are used to support dzrf


def dzls(n=64, tb=4, d1=0.01, d2=0.01):
    di = dinf(d1, d2)
    w = di / tb
    f = np.asarray([0, (1 - w) * (tb / 2), (1 + w) * (tb / 2), (n / 2)])
    f = f / (n / 2)
    m = [1, 1, 0, 0]
    w = [1, d1 / d2]

    h = signal.firls(n + 1, f, m, w)
    # shift the filter half a sample to make it symmetric, like in MATLAB
    c = np.exp(1j * 2 * np.pi / (2 * (n + 1)) *
               np.concatenate([np.arange(0, n / 2 + 1, 1),
                               np.arange(-n / 2, 0, 1)]))
    h = np.real(sp.ifft(np.multiply(sp.fft(h, center=False), c), center=False))

    # lop off extra sample
    h = h[:n]

    return h


def dzmp(n=64, tb=4, d1=0.01, d2=0.01):
    n2 = 2 * n - 1
    di = 0.5 * dinf(2 * d1, 0.5 * d2 * d2)
    w = di / tb
    f = np.asarray([0, (1 - w) * (tb / 2), (1 + w) * (tb / 2), (n / 2)]) / n
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
    hpf = sp.fft(hp, norm=None)
    hpfs = hpf - np.min(np.real(hpf)) * 1.000001
    hpfmp = mag2mp(np.sqrt(np.abs(hpfs)))
    hpmp = sp.ifft(np.fft.ifftshift(np.conj(hpfmp)), center=False, norm=None)
    hmp = hpmp[:np.int((l + 1) / 2)]

    return hmp


def dzlp(n=64, tb=4, d1=0.01, d2=0.01):
    di = dinf(d1, d2)
    w = di / tb
    f = np.asarray([0, (1 - w) * (tb / 2), (1 + w) * (tb / 2), (n / 2)]) / n
    m = [1, 0]
    w = [1, d1 / d2]

    h = signal.remez(n, f, m, w)

    return h


def msinc(n=64, m=1):
    x = np.arange(-n / 2, n / 2, 1) / (n / 2)
    snc = np.divide(np.sin(m * 2 * np.pi * x + 0.00001),
                    (m * 2 * np.pi * x + 0.00001))
    ms = np.multiply(snc, 0.54 + 0.46 * np.cos(np.pi * x))
    ms = ms * 4 * m / n

    return ms


def dz_gslider_b(n=128, g=5, gind=1, tb=4, d1=0.01, d2=0.01,
                 phi=np.pi, shift=32):
    r"""Design a g-slider pulse b

    Args:
        n (int): number of time points.
        g (int): number of sub-slices.
        gind (int): subslice index.
        tb (int): time bandwidth product.
        d1 (float): passband ripple level in :math:'M_0^{-1}'.
        d2 (float): stopband ripple level in :math:'M_0^{-1}'.
        phi (float): subslice phase.
        shift (int): n time points shift of pulse.

    Returns:
        b (array): SLR beta parameter.

    References:
        Setsompop, K. et al. 'High-resolution in vivo diffusion imaging of the
        human brain with generalized slice dithered enhanced resolution:
        Simultaneous multislice (gSlider-SMS). Magn. Reson. Med.79, 141–151
        (2018).
    """
    ftw = dinf(d1, d2) / tb  # fractional transition width of the slab profile

    if np.fmod(g, 2) and gind == int(np.ceil(g / 2)):  # centered sub-slice
        if g == 1:  # no sub-slices, as a sanity check
            b = dzls(n, tb, d1, d2)
        else:
            # Design 2 filters, to allow arbitrary phases on the subslice the
            # first is a wider notch filter with '0's where it the subslice
            # appears, and the second is the subslice. Multiply the subslice by
            # its phase and add the filters.
            f = np.asarray([0, (1 / g - ftw) * (tb / 2),
                            (1 / g + ftw) * (tb / 2),
                            (1 - ftw) * (tb / 2),
                            (1 + ftw) * (tb / 2),
                            (n / 2)])
            f = f / (n / 2)
            m_notch = [0, 0, 1, 1, 0, 0]
            m_sub = [1, 1, 0, 0, 0, 0]
            w = [1, 1, d1 / d2]

            b_notch = signal.firls(n + 1, f, m_notch, w)  # the notched filter
            b_sub = signal.firls(n + 1, f, m_sub, w)  # the subslice filter
            # add them with the subslice phase
            b = np.add(b_notch, np.multiply(np.exp(1j * phi), b_sub))
            # shift the filter half a sample to make it symmetric,
            # like in MATLAB
            c = np.exp(1j * 2 * np.pi / (2 * (n + 1)) *
                       np.concatenate([np.arange(0, n / 2 + 1, 1),
                                       np.arange(-n / 2, 0, 1)]))
            b = sp.ifft(np.multiply(sp.fft(b, center=False), c), center=False)
            # lop off extra sample
            b = b[:n]

    else:
        # design filters for the slab and the subslice, hilbert xform them
        # to suppress their left bands,
        # then demodulate the result back to DC
        gcent = shift + (gind - g / 2 - 1 / 2) * tb / g
        if gind > 1 and gind < g:
            # separate transition bands for slab+slice
            f = np.asarray([0, shift - (1 + ftw) * (tb / 2),
                            shift - (1 - ftw) * (tb / 2),
                            gcent - (tb / g / 2 + ftw * (tb / 2)),
                            gcent - (tb / g / 2 - ftw * (tb / 2)),
                            gcent + (tb / g / 2 - ftw * (tb / 2)),
                            gcent + (tb / g / 2 + ftw * (tb / 2)),
                            shift + (1 - ftw) * (tb / 2),
                            shift + (1 + ftw) * (tb / 2), (n / 2)])
            f = f / (n / 2)
            m_notch = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
            m_sub = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
            w = [d1 / d2, 1, 1, 1, d1 / d2]
        elif gind == 1:
            # the slab and slice share a left transition band
            f = np.asarray([0, shift - (1 + ftw) * (tb / 2),
                            shift - (1 - ftw) * (tb / 2),
                            gcent + (tb / g / 2 - ftw * (tb / 2)),
                            gcent + (tb / g / 2 + ftw * (tb / 2)),
                            shift + (1 - ftw) * (tb / 2),
                            shift + (1 + ftw) * (tb / 2),
                            (n / 2)])
            f = f / (n / 2)
            m_notch = [0, 0, 0, 0, 1, 1, 0, 0]
            m_sub = [0, 0, 1, 1, 0, 0, 0, 0]
            w = [d1 / d2, 1, 1, d1 / d2]
        elif gind == g:
            # the slab and slice share a right transition band
            f = np.asarray([0, shift - (1 + ftw) * (tb / 2),
                            shift - (1 - ftw) * (tb / 2),
                            gcent - (tb / g / 2 + ftw * (tb / 2)),
                            gcent - (tb / g / 2 - ftw * (tb / 2)),
                            shift + (1 - ftw) * (tb / 2),
                            shift + (1 + ftw) * (tb / 2),
                            (n / 2)])
            f = f / (n / 2)
            m_notch = [0, 0, 1, 1, 0, 0, 0, 0]
            m_sub = [0, 0, 0, 0, 1, 1, 0, 0]
            w = [d1 / d2, 1, 1, d1 / d2]

        c = np.exp(1j * 2 * np.pi / (2 * (n + 1))
                   * np.concatenate([np.arange(0, n / 2 + 1, 1),
                                     np.arange(-n / 2, 0, 1)]))

        b_notch = signal.firls(n + 1, f, m_notch, w)  # the notched filter
        b_notch = sp.ifft(np.multiply(sp.fft(b_notch, center=False), c),
                          center=False)
        b_notch = np.real(b_notch[:n])
        # hilbert transform to suppress negative passband
        b_notch = signal.hilbert(b_notch)

        b_sub = signal.firls(n + 1, f, m_sub, w)  # the sub-band filter
        b_sub = sp.ifft(np.multiply(sp.fft(b_sub, center=False), c),
                        center=False)

        b_sub = np.real(b_sub[:n])
        # hilbert transform to suppress negative passband
        b_sub = signal.hilbert(b_sub)

        # add them with the subslice phase
        b = b_notch + np.exp(1j * phi) * b_sub

        # demodulate to DC
        c_shift = np.exp(-1j * 2 * np.pi / n * shift * np.arange(0, n, 1)) / 2
        c_shift *= np.exp(-1j * np.pi / n * shift)

        b = np.multiply(b, c_shift)

    return b


def dz_hadamard_b(n=128, g=5, gind=1, tb=4, d1=0.01, d2=0.01, shift=32):
    r"""Design a pulse with hadamard encoding

    Args:
        n (int): number of time points.
        g (int): order of the Hadamard matrix.
        gind (int): index of vector to use from Hadamard matrix for encoding.
        tb (int): time bandwidth product.
        d1 (float): passband ripple level in :math:'M_0^{-1}'.
        d2 (float): stopband ripple level in :math:'M_0^{-1}'.
        shift (int): n time points shift of pulse.

    Returns:
        b (array): SLR beta parameter.

    References:
            Souza, S.P., Szumowski, J., Dumoulin, C.L., Plewes, D.P. &
            Glover, G. 'Sima: Simultaneous multislice acquisition of MR images
            by hadamard - encoded excitation. J.Comput.Assist.Tomogr. 12,
            1026–1030(1988).
    """

    H = linalg.hadamard(g)
    encode = H[gind - 1, :]

    ftw = dinf(d1, d2) / tb  # fractional transition width of the slab profile

    if gind == 1:  # no sub-slices
        b = dzls(n, tb, d1, d2)
    else:
        # left stopband
        f = np.asarray([0, shift - (1 + ftw) * (tb / 2)])
        m = np.asarray([0, 0])
        w = np.asarray([d1 / d2])
        # first sub-band
        ii = 1
        gcent = shift + (ii - g / 2 - 1 / 2) * tb / g  # first band center
        # first band left edge
        f = np.append(f, gcent - (tb / g / 2 - ftw * (tb / 2)))
        m = np.append(m, encode[ii - 1])
        if encode[ii - 1] != encode[ii]:
            # add the first band's right edge and its amplitude, and a weight
            f = np.append(f, gcent + (tb / g / 2 - ftw * (tb / 2)))
            m = np.append(m, encode[ii - 1])
            w = np.append(w, 1)
        # middle sub-bands
        for ii in range(2, g):
            gcent = shift + (ii - g / 2 - 1 / 2) * tb / g  # center of band
            if encode[ii - 1] != encode[ii - 2]:
                # add a left edge and amp for this band
                f = np.append(f, gcent - (tb / g / 2 - ftw * (tb / 2)))
                m = np.append(m, encode[ii - 1])
            if encode[ii - 1] != encode[ii]:
                # add a right edge and its amp, and a weight for this band
                f = np.append(f, gcent + (tb / g / 2 - ftw * (tb / 2)))
                m = np.append(m, encode[ii - 1])
                w = np.append(w, 1)
        # last sub-band
        ii = g
        gcent = shift + (ii - g / 2 - 1 / 2) * tb / g  # center of last band
        if encode[ii - 1] != encode[ii - 2]:
            # add a left edge and amp for the last band
            f = np.append(f, gcent - (tb / g / 2 - ftw * (tb / 2)))
            m = np.append(m, encode[ii - 1])
        # add a right edge and its amp, and a weight for the last band
        f = np.append(f, gcent + (tb / g / 2 - ftw * (tb / 2)))
        m = np.append(m, encode[ii - 1])
        w = np.append(w, 1)
        # right stop-band
        f = np.append(f, (shift + (1 + ftw) * (tb / 2), (n / 2))) / (n / 2)
        m = np.append(m, [0, 0])
        w = np.append(w, d1 / d2)

        # separate the positive and negative bands
        mp = (m > 0).astype(float)
        mn = (m < 0).astype(float)

        # design the positive and negative filters
        c = np.exp(1j * 2 * np.pi / (2 * (n + 1))
                   * np.concatenate([np.arange(0, n / 2 + 1, 1),
                                     np.arange(-n / 2, 0, 1)]))
        bp = signal.firls(n + 1, f, mp, w)  # the positive filter
        bn = signal.firls(n + 1, f, mn, w)  # the negative filter

        # combine the filters and demodulate
        b = sp.ifft(np.multiply(sp.fft(bp - bn, center=False), c),
                    center=False)

        b = np.real(b[:n])
        # hilbert transform to suppress negative passband
        b = signal.hilbert(b)
        # demodulate to DC
        c_shift = np.exp(-1j * 2 * np.pi / n * shift * np.arange(0, n, 1)) / 2
        c_shift *= np.exp(-1j * np.pi / n * shift)
        b = np.multiply(b, c_shift)

    return b


def dz_gslider_rf(n=256, g=5, flip=np.pi / 2, phi=np.pi, tb=12,
                  d1=0.01, d2=0.01, cancel_alpha_phs=True):
    r"""Design a g-slider pulse rf

    Args:
        n (int): number of time points.
        g (int): number of sub-slices.
        flip (float): flip angle.
        phi (float): subslice phase.
        tb (int): time bandwidth product.
        d1 (float): passband ripple level in :math:'M_0^{-1}'.
        d2 (float): stopband ripple level in :math:'M_0^{-1}'.
        cancel_alpha_phs (bool): absorb the alpha phase
            profile from beta's profile, so they cancel for a flatter
            total phase

    Returns:
        rf (array): rf pulse out.

    References:
        Setsompop, K. et al. 'High-resolution in vivo diffusion imaging of the
        human brain with generalized slice dithered enhanced resolution:
        Simultaneous multislice (gSlider-SMS). Magn. Reson. Med.79, 141–151
        (2018).
    """
    bsf = np.sin(flip / 2)  # beta scaling factor

    rf = np.zeros((n, g), dtype='complex')
    for gind in range(1, g + 1):
        b = bsf * dz_gslider_b(n, g, gind, tb, d1, d2, phi)
        rf[:, gind - 1] = b2rf(b, cancel_alpha_phs)

    return rf


def b2rf(b, cancel_alpha_phs=False):
    a = b2a(b)
    if cancel_alpha_phs:
        b_a_phase = sp.fft(b, center=False, norm=None) * \
            np.exp(-1j * np.angle(sp.fft(a[np.size(a)::-1],
                                         center=False, norm=None)))
        b = sp.ifft(b_a_phase, center=False, norm=None)
    rf = ab2rf(a, b)

    return rf


def b2a(b):
    n = np.size(b)

    npad = n * 16
    bcp = np.zeros(npad, dtype=complex)
    bcp[0:n:1] = b
    bf = sp.fft(bcp, center=False, norm=None)
    bfmax = np.max(np.abs(bf))
    if bfmax >= 1:
        bf = bf / (1e-7 + bfmax)
    afa = mag2mp(np.sqrt(1 - np.abs(bf) ** 2))
    a = sp.fft(afa, center=False, norm=None) / npad
    a = a[0:n:1]
    a = a[::-1]

    return a


def mag2mp(x):
    n = np.size(x)
    xl = np.log(np.abs(x))  # Log of mag spectrum
    xlf = sp.fft(xl, center=False, norm=None)
    xlfp = xlf
    xlfp[0] = xlf[0]  # Keep DC the same
    xlfp[1:(n // 2):1] = 2 * xlf[1:(n // 2):1]  # Double positive frequencies
    xlfp[n // 2] = xlf[n // 2]  # keep half Nyquist the same
    xlfp[n // 2 + 1:n:1] = 0  # zero negative frequencies
    xlaf = sp.ifft(xlfp, center=False, norm=None)
    a = np.exp(xlaf)  # complex exponentiation

    return a


def ab2rf(a, b):
    n = np.size(a)
    rf = np.zeros(n, dtype=complex)

    a = a.astype(complex)
    b = b.astype(complex)

    for ii in range(n - 1, -1, -1):

        cj = np.sqrt(1 / (1 + np.abs(b[ii] / a[ii]) ** 2))
        sj = np.conj(cj * b[ii] / a[ii])
        theta = np.arctan2(np.abs(sj), cj)
        psi = np.angle(sj)
        rf[ii] = 2 * theta * np.exp(1j * psi)

        # remove this rotation from polynomials
        if ii > 0:
            at = cj * a + sj * b
            bt = -np.conj(sj) * a + cj * b
            a = at[1:ii + 1:1]
            b = bt[0:ii:1]

    return rf


def root_flip(b, d1, flip, tb, verbose=False):
    r"""Exhaustive root-flip pattern search for min-peak b1

    Args:
        b (array): SLR beta parameter.
        d1 (float): passband ripple level.
        flip (array): target flip angle.
        tb (int): pulse time bandwidth product.
        verbose (bool): print feeback on iterations.

    Returns:
        2-element tuple containing

        - **rf_out** (*array*): rf pulse out.
        - **b_out** (*array*): SLR beta parameter.

    References:
        Sharma, A. Lustig, M. and Grissom, W. (2016).
        Root-flipped multiband refocusing pulses.
        Magn Reson Med. 2016 Jan; 75(1): 227-237.
     """

    n = np.size(b)
    [_, b_resp] = signal.freqz(b)
    b /= np.max(np.abs(b_resp))  # normalize beta
    b *= np.sin(flip / 2 + np.arctan(d1 * 2) / 2)  # scale to target flip
    r = sp.util.leja(np.roots(b))

    candidates = np.logical_and(np.abs(1 - np.abs(r)) > 0.004,
                                np.abs(np.angle(r)) < tb / n * np.pi)

    ii_min = 0
    ii_max = 2 ** np.sum(candidates)

    max_rf = np.max(np.abs(b2rf(b)))

    for ii in range(ii_min, ii_max):

        if ii % 20 == 0:
            if verbose:
                print('Evaluating root-flip pattern ' + str(ii) + ' out of ' +
                      str(ii_max - ii_min))

        # get a binary flipping pattern
        do_flip_str = format(ii, 'b')
        do_flip = np.zeros(np.sum(candidates), dtype=bool)
        for jj in range(0, len(do_flip_str)):
            do_flip[jj] = bool(int(do_flip_str[jj]))

        # embed the pattern in an all-roots vector
        tmp = np.zeros(n - 1, dtype=bool)
        tmp[candidates] = do_flip
        do_flip = tmp

        # flip those indices
        r_flip = np.zeros(np.shape(r), dtype=complex)
        r_flip[:] = r[:]
        r_flip[do_flip] = np.conj(1 / r_flip[do_flip])

        b_tmp = np.poly(r_flip)
        [_, b_tmp_resp] = signal.freqz(b_tmp)
        b_tmp /= np.max(np.abs(b_tmp_resp))  # normalize beta
        b_tmp *= np.sin(flip / 2 + np.arctan(d1 * 2) / 2)  # scale to targ flip
        rf_tmp = b2rf(b_tmp)

        if np.max(np.abs(rf_tmp)) < max_rf:
            max_rf = np.max(np.abs(rf_tmp))
            rf_out = rf_tmp
            b_out = b_tmp

    return rf_out, b_out


def dz_recursive_rf(n_seg, tb, n, se_seq=False, tb_ref=8, z_pad_fact=4,
                    win_fact=1.75, cancel_alpha_phs=True, t1=np.inf, tr_seg=60,
                    use_mz=True, d1=0.01, d2=0.01, d1se=0.01, d2se=0.01):
    r"""Recursive SLR pulse design.

    Args:
        n_seg (int): number of segments designed by recursion.
        tb (int): time bandwidth product.
        n (int): pulse length.
        se_seq (bool): spin echo sequence.
        tb_ref (int): time bandwidth product of refocusing pulse.
        z_pad_fact (float): zero padding factor.
        win_fact (float): applied window factor.
        cancel_alpha_phs (bool): absorb the alpha phase
            profile from beta's profile, so they cancel for a flatter
            total phase
        t1 (float): t1
        tr_seg (int): length of tr
        use_mz (bool): design the pulses accounting for the actual Mz profile
        d1 (float): passband ripple level in :math:'M_0^{-1}'.
        d2 (float): stopband ripple level in :math:'M_0^{-1}'.
        d1se (float): passband ripple level for se
        d2se (float): stopband ripple level for se

    Returns:
        If se_seq=True, 2-element tuple containing

        - **rf** (*array*): rf pulse out.
        - **rf_ref** (*array*): rf refocusing pulse out.
    """

    # get refocusing pulse and its rotation parameters
    if se_seq is True:
        [bsf, d1se, d2se] = calc_ripples('se', d1se, d2se)
        b_ref = bsf * dzls(n, tb_ref, d1se, d2se)
        b_ref = np.concatenate((np.zeros(int(z_pad_fact * n / 2 - n / 2)),
                                b_ref,
                                np.zeros(int(z_pad_fact * n / 2 - n / 2))))
        rf_ref = b2rf(b_ref)
        bref = sp.fft(b_ref, norm=None)
        bref /= np.max(np.abs(bref))
        bref_mag = np.abs(bref)
        aref_mag = np.abs(np.sqrt(1 - bref_mag ** 2))
        flip_ref = 2 * np.arcsin(bref_mag[int(z_pad_fact * n / 2)]) \
            * 180 / np.pi

    # get flip angles
    flip = np.zeros(n_seg)
    flip[n_seg - 1] = 90
    for jj in range(n_seg - 2, -1, -1):
        if se_seq is False:
            flip[jj] = np.arctan(np.sin(flip[jj + 1] * np.pi / 180))
            flip[jj] = flip[jj] * 180 / np.pi  # deg
        else:
            flip[jj] = np.arctan(np.cos(flip_ref * np.pi / 180) *
                                 np.sin(flip[jj + 1] * np.pi / 180))
            flip[jj] = flip[jj] * 180 / np.pi  # deg

    # design first RF pulse
    b = np.zeros((int(z_pad_fact * n), n_seg), dtype=complex)
    b[int(z_pad_fact * n / 2 - n / 2):int(z_pad_fact * n / 2 + n / 2), 0] = \
        dzls(n, tb, d1, d2)
    # b = np.concatenate((np.zeros(int(zPadFact*N/2-N/2)), b,
    #    np.zeros(int(zPadFact*N/2-N/2))))
    B = sp.fft(b[:, 0], norm=None)

    c = np.exp(-1j * 2 * np.pi / (n * z_pad_fact) / 2 *
               np.arange(-n * z_pad_fact / 2, n * z_pad_fact / 2, 1))
    B = np.multiply(B, c)
    b[:, 0] = sp.ifft(B / np.max(np.abs(B)), norm=None)
    b[:, 0] *= np.sin(flip[0] * (np.pi / 180) / 2)
    rf = np.zeros((z_pad_fact * n, n_seg), dtype=complex)
    a = b2a(b[:, 0])
    if cancel_alpha_phs:
        # cancel a phase by absorbing into b
        # Note that this is the only time we need to do it
        b_a_phase = sp.fft(b[:, 0], center=False, norm=None) * \
            np.exp(-1j * np.angle(sp.fft(a[np.size(a)::-1],
                                         center=False, norm=None)))
        b[:, 0] = sp.ifft(b_a_phase, center=False, norm=None)
    rf[:, 0] = b2rf(b[:, 0])

    # get min-phase alpha and its response
    # a = b2a(b[:, 0])
    A = sp.fft(a)

    # calculate beta filter response
    B = sp.fft(b[:, 0], norm=None)

    if win_fact < z_pad_fact:
        win_len = (win_fact - 1) * n
        npad = n * z_pad_fact - win_fact * n
        # blackman window?
        window = signal.blackman(int((win_fact - 1) * n))
        # split in half; stick N ones in the middle
        window = np.concatenate((window[0:int(win_len / 2)], np.ones(n),
                                 window[int(win_len / 2):]))
        window = np.concatenate((np.zeros(int(npad / 2)), window,
                                 np.zeros(int(npad / 2))))
        # apply windowing to first pulse for consistency
        b[:, 0] = np.multiply(b[:, 0], window)
        rf[:, 0] = b2rf(b[:, 0])
        # recalculate B and A
        B = sp.fft(b[:, 0], norm=None)
        A = sp.fft(b2a(b[:, 0]), norm=None)

    # use A and B to get Mxy
    # Mxy = np.zeros((zPadFact*N, Nseg), dtype = complex)
    if se_seq is False:
        mxy0 = 2 * np.conj(A) * B
    else:
        mxy0 = 2 * A * np.conj(B) * bref ** 2

    # Amplitude of next pulse's Mxy profile will be
    #               |Mz*2*a*b| = |Mz*2*sqrt(1-abs(B).^2)*B|.
    # If we set this = |Mxy_1|, we can solve for |B| via solving quadratic
    # equation 4*Mz^2*(1-B^2)*B^2 = |Mxy_1|^2.
    # Subsequently solve for |A|, and get phase of A via min-phase, and
    # then get phase of B by dividing phase of A from first pulse's Mxy phase.
    mz = np.ones((z_pad_fact * n), dtype=complex)
    for jj in range(1, n_seg):

        # calculate Mz profile after previous pulse
        if se_seq is False:
            mz = mz * (1 - 2 * np.abs(B) ** 2) * np.exp(-tr_seg / t1) + \
                 (1 - np.exp(-tr_seg / t1))
        else:
            mz = mz * (1 - 2 * (np.abs(A * bref_mag) ** 2
                                + np.abs(aref_mag * B) ** 2))
            # (second term is about 1%)

        if use_mz is True:  # design the pulses accounting for the
            # actual Mz profile (the full method)
            # set up quadratic equation to get |B|
            cq = -np.abs(mxy0) ** 2
            if se_seq is False:
                bq = 4 * mz ** 2
                aq = -4 * mz ** 2
            else:
                bq = 4 * (bref_mag ** 4) * mz ** 2
                aq = -4 * (bref_mag ** 4) * mz ** 2
            bmag = np.sqrt((-bq + np.real(np.sqrt(bq ** 2 - 4 * aq * cq)))
                           / (2 * aq))
            bmag[np.isnan(bmag)] = 0
            # get A - easier to get complex A than complex B since |A| is
            # determined by |B|, and phase is gotten by min-phase relationship
            # Phase of B doesn't matter here since only profile mag is used by
            # b2a
            A = sp.fft(b2a(sp.ifft(bmag, norm=None)), norm=None)
            # trick: now we can get complex B from ratio of Mxy and A
            B = mxy0 / (2 * np.conj(A) * mz)

        else:  # design assuming ideal Mz (conventional VFA)

            B *= np.sin(np.pi / 180 * flip[jj] / 2) \
                 / np.sin(np.pi / 180 * flip[jj - 1] / 2)
            A = sp.fft(b2a(sp.ifft(B, norm=None)), norm=None)

        # get polynomial
        b[:, jj] = sp.ifft(B, norm=None)

        if win_fact < z_pad_fact:
            b[:, jj] *= window
            # recalculate B and A
            B = sp.fft(b[:, jj], norm=None)
            A = sp.fft(b2a(b[:, jj]), norm=None)

        rf[:, jj] = b2rf(b[:, jj])

    # truncate the RF
    if win_fact < z_pad_fact:
        pulse_len = int(win_fact * n)
        rf = rf[int(npad / 2):int(npad / 2 + pulse_len), :]

    if se_seq is False:
        return rf
    else:
        return rf, rf_ref
