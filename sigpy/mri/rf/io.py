# -*- coding: utf-8 -*-
"""MRI waveform import/export files.
"""

import numpy as np
import struct

__all__ = ['signa', 'ge_rf_params', 'philips_rf_params', 'siemens_rf']


def siemens_rf(pulse, rfbw, rfdurms, pulsename, minslice=0.5, maxslice=320.0,
               comment=None):
    """Write a .pta text file for Siemens PulseTool.

    Args:
        pulse (array): complex-valued RF pulse array with maximum of 4096
            points.
        rfbw (float): bandwidth of RF pulse in Hz
        rfdurms (float): duration of RF pulse in ms
        pulsename (string): '<FamilyName>.<PulseName>', e.g. 'Sigpy.SincPulse'
        minslice (float): minimum slice thickness [mm]
        maxslice (float): maximum slice thickness [mm]
        comment (string): a comment that can be seen in Siemens PulseTool

    Note this has only been tested on MAGNETOM Verio running (VB17)

    Open pulsetool from the IDEA command line. Open the extrf.dat file and add
    this .pta file using the import function

    Recommended to make a copy and renaming extrf.dat prior to making changes.

    After saving a new pulse to <myUniqueFileName>_extrf.dat and copying it to
    the scanner, you will need to re-boot the host for it to load changes.

    """

    # get the number of points in RF waveform
    npts = pulse.size
    assert npts <= 4096, ('RF pulse must have less than 4096 points for'
                          ' Siemens VB17')

    if comment is None:
        comment = ''

    # Calculate reference gradient value.
    # This is necessary for proper calculation of slice-select gradient
    # amplitude using the .getGSAmplitude() method for the external RF class.
    # See the IDEA documentation for more details on this.
    refgrad = 1000.0 * rfbw * (rfdurms/5.12) / (42.577E06 * (10.0/1000.0))

    rffile = open(pulsename+'.pta', 'w')
    rffile.write('PULSENAME: {}\n'.format(pulsename))
    rffile.write('COMMENT: {}\n'.format(comment))
    rffile.write('REFGRAD: {:6.5f}\n'.format(refgrad))
    rffile.write('MINSLICE: {:6.5f}\n'.format(minslice))
    rffile.write('MAXSLICE: {:6.5f}\n'.format(maxslice))

    # the following are related to SAR calcs and will be calculated by
    # PulseTool upon loading the pulse
    rffile.write('AMPINT: \n')
    rffile.write('POWERINT: \n')
    rffile.write('ABSINT: \n\n')

    # magnitude must be between 0 and 1
    mxmag = np.max(np.abs(pulse))
    for n in range(npts):
        mag = np.abs(pulse[n]) / mxmag  # magnitude at current point
        mag = np.squeeze(mag)
        pha = np.angle(pulse[n])     # phase at current point
        pha = np.squeeze(pha)
        rffile.write('{:10.9f}\t{:10.9f}\t; ({:d})\n'.format(mag, pha, n))
    rffile.close()


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
