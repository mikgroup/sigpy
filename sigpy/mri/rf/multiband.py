# -*- coding: utf-8 -*-
"""Multiband RF Pulse Design functions.
"""
import numpy as np

__all__ = ['mb_phs_tab', 'mb_rf', 'dz_pins']

from sigpy.mri.rf.trajgrad import trap_grad
from sigpy.mri.rf import slr as slr


def mb_rf(pulse_in, n_bands=3, band_sep=5, phs_0_pt='None'):
    r"""Multiband an input RF pulse.

     Args:
         pulse_in (array): samples of single-band RF pulse.
         n_bands (int): number of bands.
         band_sep (float): normalized slice separation.
         phs_0_pt (string): set of phases to use. Can be 'phs_mod' (Wong),
            'amp_mod' (Malik), 'quad_mod' (Grissom), or 'None'

     band_sep = slice_sep/slice_thick*tb, where tb is time-bandwidth product
     of the single-band pulse

     Returns:
         array: multibanded pulse out

     References:
         Wong, E. (2012). 'Optimized Phase Schedules for Minimizing Peak RF
         Power in Simultaneous Multi-Slice RF Excitation Pulses'. Proc. Intl.
         Soc. Mag. Reson. Med., 20 p. 2209.
         Malik, S. J., Price, A. N., and Hajnal, J. V. (2015). 'Optimized
         Amplitude Modulated Multi-Band RF pulses'. Proc. Intl. Soc. Mag.
         Reson. Med., 23 p. 2398.
     """

    if phs_0_pt != 'None':
        phs = mb_phs_tab(n_bands, phs_0_pt)
    else:
        phs = np.zeros(n_bands)

    # build multiband modulation function
    n = np.size(pulse_in)
    b = np.zeros(n, dtype='complex')
    for ii in range(0, n_bands):
        b += np.exp(1j * 2 * np.pi / n * band_sep * np.arange(-n / 2, n / 2, 1)
                    * (ii - (n_bands - 1) / 2)) * np.exp(1j * phs[ii])

    pulse_out = b * pulse_in

    return pulse_out


def mb_phs_tab(n_bands, phs_type='phs_mod'):
    # Return phases to minimize peak b1 amplitude of an MB pulse

    if phs_type == 'phs_mod':

        if n_bands < 3 or n_bands > 16:
            raise Exception('Wongs phases valid for 2 < nBands < 17.')

        # Eric Wong's phases: From E C Wong, ISMRM 2012, p. 2209
        p = np.zeros((14, 16))
        p[0, 1:3] = np.array([0.73, 4.602])
        p[1, 1:4] = np.array([3.875, 5.94, 6.197])
        p[2, 1:5] = np.array([3.778, 5.335, 0.872, 0.471])
        p[3, 1:6] = np.array([2.005, 1.674, 5.012, 5.736, 4.123])
        p[4, 1:7] = np.array([3.002, 5.998, 5.909, 2.624, 2.528, 2.440])
        p[5, 1:8] = np.array([1.036, 3.414, 3.778, 3.215, 1.756, 4.555, 2.467])
        p[6, 1:9] = np.array([1.250, 1.783, 3.558, 0.739, 3.319, 1.296,
                              0.521, 5.332])
        p[7, 1:10] = np.array([4.418, 2.360, 0.677, 2.253, 3.472, 3.040,
                               3.974, 1.192, 2.510])
        p[8, 1:11] = np.array([5.041, 4.285, 3.001, 5.765, 4.295, 0.056,
                               4.213, 6.040, 1.078, 2.759])
        p[9, 1:12] = np.array([2.755, 5.491, 4.447, 0.231, 2.499, 3.539,
                               2.931, 2.759, 5.376, 4.554, 3.479])
        p[10, 1:13] = np.array([0.603, 0.009, 4.179, 4.361, 4.837, 0.816,
                                5.995, 4.150, 0.417, 1.520, 4.517, 1.729])
        p[11, 1:14] = np.array([3.997, 0.830, 5.712, 3.838, 0.084, 1.685,
                                5.328, 0.237, 0.506, 1.356, 4.025, 4.483,
                                4.084])
        p[12, 1:15] = np.array([4.126, 2.266, 0.957, 4.603, 0.815, 3.475,
                                0.977, 1.449, 1.192, 0.148, 0.939, 2.531,
                                3.612, 4.801])
        p[13, 1:16] = np.array([4.359, 3.510, 4.410, 1.750, 3.357, 2.061,
                                5.948, 3.000, 2.822, 0.627, 2.768, 3.875,
                                4.173, 4.224, 5.941])

        out = p[n_bands - 3, 0:n_bands]

    elif phs_type == 'amp_mod':

        # Malik's Hermitian phases: From S J Malik, ISMRM 2015, p. 2398
        if n_bands < 4 or n_bands > 12:
            raise Exception('Maliks phases valid for 3 < nBands < 13.')

        p = np.zeros((9, 12))
        p[0, 0:4] = np.array([0, np.pi, np.pi, 0])
        p[1, 0:5] = np.array([0, 0, np.pi, 0, 0])
        p[2, 0:6] = np.array([1.691, 2.812, 1.157, -1.157, -2.812, -1.691])
        p[3, 0:7] = np.array([2.582, -0.562, 0.102, 0, -0.102, 0.562, -2.582])
        p[4, 0:8] = np.array([2.112, 0.220, 1.464, 1.992, -1.992, -1.464,
                              -0.220, -2.112])
        p[5, 0:9] = np.array([0.479, -2.667, -0.646, -0.419, 0, 0.419, 0.646,
                              2.667, -0.479])
        p[6, 0:10] = np.array([1.683, -2.395, 2.913, 0.304, 0.737, -0.737,
                               -0.304, -2.913, 2.395, -1.683])
        p[7, 0:11] = np.array([1.405, 0.887, -1.854, 0.070, -1.494, 0, 1.494,
                               -0.070, 1.854, -0.887, -1.405])
        p[8, 0:12] = np.array([1.729, 0.444, 0.722, 2.190, -2.196, 0.984,
                               -0.984, 2.196, -2.190, -0.722, -0.444, -1.729])

        out = p[n_bands - 4, 0:n_bands]

    elif phs_type == 'quad_mod':

        # Grissom's quadratic phases (unpublished)
        k = 3.4 / n_bands  # quadratic phase coefficient
        out = k * (np.arange(0, n_bands, 1) - (n_bands - 1) / 2) ** 2

    else:
        raise Exception('phase type ("{}") not recognized.'.format(phs_type))

    return out


def dz_pins(tb, sl_sep, sl_thick, g_max, g_slew, dt, b1_max=0.18,
            ptype='ex', ftype='ls', d1=0.01, d2=0.01, gambar=4258):

    r"""PINS multiband pulse design.

    Args:
        tb (float): time-bandwidth product.
        sl_sep (float): slice separation in cm.
        sl_thick (float): slice thickness in cm.
        g_max (float): max gradient amplitude in gauss/cm
        g_slew (float): max gradient sliew in gauss/cm/s
        dt (float): RF + gradient dwell time in s.
        b1_max (float): Maximum RF amplitude
        ptype (string): pulse type, 'st' (small-tip excitation), 'ex' (pi/2
            excitation pulse), 'se' (spin-echo pulse), 'inv' (inversion), or
            'sat' (pi/2 saturation pulse).
        ftype (string): type of filter to use in pulse design
        d1 (float): passband ripple level in :math:'M_0^{-1}'.
        d2 (float): stopband ripple level in :math:'M_0^{-1}'.
        gambar (float): Appropriate gyromagnetic ratio in Hz/gauss.

    Returns:
        2-element tuple containing:

        - **rf** (*array*): RF Pulse in Gauss
        - **g** (*array*): Gradient waveform in Gauss/cm

    References:
        Norris, D.G. and Koopmans, P.J. and Boyacioglu, R and Barth, M (2011).
        'Power independent of number of slices (PINS) radiofrequency Pulses
        for low-power simultaneous multislice excitation'.
        Magn. Reson. Med., 66(5):1234-1240.
    """

    kz_width = tb / sl_thick  # 1/cm, width in k-space we must go
    # calcualte number of subpulses (odd)
    n_pulses = int(2 * np.floor(np.ceil(kz_width / (1 / sl_sep)) / 2))
    # call SLR to get envelope
    rf_soft = slr.dzrf(n_pulses, tb, ptype, ftype, d1, d2)

    # design the blip trapezoid
    g_area = 1 / sl_sep / gambar
    [gz_blip, _] = trap_grad(g_area, g_max, g_slew, dt)

    # Calculate the block/hard RF pulse width based on
    b1_scaled = 2 * np.pi * gambar * b1_max * dt
    hpw = int(np.ceil(np.max(np.abs(rf_soft)) / b1_scaled))

    # interleave RF subpusles with gradient subpulses to form full pulses
    rf = np.kron(rf_soft[:-1], np.concatenate((np.ones(hpw),
                 np.zeros((np.size(gz_blip))))))
    rf = np.concatenate((rf, rf_soft[-1] * np.ones(hpw)))
    rf = rf / (np.sum(rf) * 2 * np.pi * gambar * dt) * np.sum(rf_soft)

    g = np.concatenate([np.zeros(hpw), np.squeeze(gz_blip)])
    g = np.tile(g, n_pulses - 1)
    g = np.concatenate((g, np.zeros(hpw)))

    return rf, g
