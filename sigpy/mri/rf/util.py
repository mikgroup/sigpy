# -*- coding: utf-8 -*-
"""MRI RF utilities.
"""

import numpy as np
import struct

__all__ = ['dinf']


def dinf(d1=0.01, d2=0.01):
    """Calculate D infinity for a linear phase filter.

    Args:
        d1 (float): passband ripple level in M0**-1.
        d2 (float): stopband ripple level in M0**-1.

    Returns:
        float: D infinity.

    References:
        Pauly J, Le Roux P, Nishimra D, Macovski A. Parameter relations for the
        Shinnar-Le Roux selective excitation pulse design algorithm.
        IEEE Tr Medical Imaging 1991; 10(1):53-65.

    """

    a1 = 5.309e-3
    a2 = 7.114e-2
    a3 = -4.761e-1
    a4 = -2.66e-3
    a5 = -5.941e-1
    a6 = -4.278e-1

    l10d1 = np.log10(d1)
    l10d2 = np.log10(d2)

    d = (a1 * l10d1 * l10d1 + a2 * l10d1 + a3) * l10d2 \
        + (a4 * l10d1 * l10d1 + a5 * l10d1 + a6)

    return d


def signa(wav, filename, scale = -1):
    """Write a binary waveform in the GE format.

    Args:
        wav (array): waveform, may be complex-valued.
        filename (string): filename to write to.
        scale (float): scaling factor to apply (default = waveform's max)

    Adapted from John Pauly's RF Tools signa() MATLAB function

    """
        
    wmax = int('7ffe', 16)
    
    if np.iscomplexobj(wav) == False:
    
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
            scale = 1 / np.max((np.max(np.abs(np.real(wav))), np.max(np.abs(np.imag(wav)))))
            
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
        
        
    