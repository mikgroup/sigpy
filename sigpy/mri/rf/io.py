# -*- coding: utf-8 -*-
"""MRI waveform import/export files.
"""

import numpy as np
import struct

__all__ = ['signa', 'ge_rf_params', 'philips_rf_params']


def signa(wav, filename, scale=-1):
    """Write a binary waveform in the GE format.

    Args:
        wav (array): waveform (gradient or RF), may be complex-valued.
        filename (string): filename to write to.
        scale (float): scaling factor to apply (default = waveform's max)

    Adapted from John Pauly's RF Tools signa() MATLAB function

    """

    wmax = int('7ffe', 16)

    if not np.iscomplexobj(wav):

        if scale == -1:
            scale = 1 / np.max(np.abs(wav))

        # scale up to fit in a short integer
        wav = wav * scale * wmax

        # mask off low bit, since it would be an EOS otherwise
        wav = 2 * np.round(wav / 2)

        fid = open(filename, 'wb')

        for x in np.nditer(wav):
            fid.write(struct.pack('>h', int(x.item())))

        fid.close()

    else:

        if scale == -1:
            scale = 1 / np.max(
                (np.max(np.abs(np.real(wav))), np.max(np.abs(np.imag(wav)))))

        # scale up to fit in a short integer
        wav = wav * scale * wmax

        # mask off low bit, since it would be an EOS otherwise
        wav = 2 * np.round(wav / 2)

        fid = open(filename + '.r', 'wb')

        for x in np.nditer(wav):
            fid.write(struct.pack('>h', np.real(x)))

        fid.close()

        fid = open(filename + '.i', 'wb')

        for x in np.nditer(wav):
            fid.write(struct.pack('>h', np.imag(x)))

        fid.close()


def ge_rf_params(rf, dt=4e-6):
    """Calculate RF pulse parameters for deployment
    on a GE scanner.

    Args:
        rf (array): RF pulse samples
        dt (scalar): RF dwell time (seconds)

    Adapted from Adam Kerr's rf_save() MATLAB function

    """

    print('GE RF Pulse Parameters:')

    n = len(rf)
    rfn = rf / np.max(np.abs(rf))

    abswidth = np.sum(np.abs(rfn)) / n
    print('abswidth = ', abswidth)

    effwidth = np.sum(np.abs(rfn) ** 2) / n
    print('effwidth = ', effwidth)

    print('area = ', abswidth)

    pon = np.abs(rfn) > 0
    temp_pw = 0
    max_pw = 0
    for i in range(0, len(rfn)):
        if pon[i] == 0 & temp_pw > 0:
            max_pw = np.max(max_pw, temp_pw)
            temp_pw = 0
    max_pw = max_pw / n

    dty_cyc = np.sum(np.abs(rfn) > 0.2236) / n
    if dty_cyc < max_pw:
        dty_cyc = max_pw
    print('dtycyc = ', dty_cyc)
    print('maxpw = ', max_pw)

    max_b1 = np.max(np.abs(rf))
    print('max_b1 = ', max_b1)

    int_b1_sqr = np.sum(np.abs(rf) ** 2) * dt * 1e3
    print('int_b1_sqr = ', int_b1_sqr)

    rms_b1 = np.sqrt(np.sum(np.abs(rf) ** 2)) / n
    print('max_rms_b1 = ', rms_b1)


def philips_rf_params(rf):
    """Calculate RF pulse parameters for deployment
    on a Philips scanner.

    Args:
        rf (array): RF pulse samples (assumed real-valued)

    """

    print('Philips RF Pulse Parameters')

    n = len(rf)
    rfn = rf / np.max(np.abs(rf))

    am_c_teff = np.sum(rfn * 32767) / (32767 * n)
    print('am_c_teff = ', am_c_teff)

    am_c_trms = np.sum((rfn * 32767) ** 2) / (32767 ** 2 * n)
    print('am_c_trms = ', am_c_trms)

    am_c_tabs = np.sum(np.abs(rfn) * 32767) / (32767 * n)
    print('am_c_tabs = ', am_c_tabs)

    # assume that the isodelay point is at the peak
    am_c_sym = np.argmax(np.abs(rfn)) / n
    print('am_c_sym = ', am_c_sym)
