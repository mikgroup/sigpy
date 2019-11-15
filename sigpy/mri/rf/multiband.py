# -*- coding: utf-8 -*-
"""Multiband RF Pulse Design functions.
"""
import numpy as np

__all__ = ['mb_phs_tab', 'mb_rf']


def mb_rf(pulse_in, n_bands=3, band_sep=5, phs_0_pt='None'):
    r"""Multiband an input RF pulse.

     Args:
         pulse_in (array): samples of single-band RF pulse.
         n_bands (int): number of bands.
         band_sep (float): normalized slice separation.
         phs_0_pt (string): set of phases to use. Can be 'phs_mod' (Wong),
            'amp_mod' (Malik), 'quad_mod' (Grissom), or 'None'

     bandSep = sliceSep/sliceThick*tb, where tb is time-bandwidth product
     of the single-band pulse

     Returns:
         array: multibanded pulse out
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
    # return phases to minimize peak b1 amplitude of an MB pulse

    if phs_type == 'phs_mod':

        if n_bands < 3 or n_bands > 16:
            raise Exception('Wongs phases valid for 2 < nBands < 17.')

        # Eric Wong's phases: From E C Wong, ISMRM 2012, p. 2209
        P = np.zeros((14, 16))
        P[0, 1:3] = np.array([0.73, 4.602])
        P[1, 1:4] = np.array([3.875, 5.94, 6.197])
        P[2, 1:5] = np.array([3.778, 5.335, 0.872, 0.471])
        P[3, 1:6] = np.array([2.005, 1.674, 5.012, 5.736, 4.123])
        P[4, 1:7] = np.array([3.002, 5.998, 5.909, 2.624, 2.528, 2.440])
        P[5, 1:8] = np.array([1.036, 3.414, 3.778, 3.215, 1.756, 4.555, 2.467])
        P[6, 1:9] = np.array([1.250, 1.783, 3.558, 0.739, 3.319, 1.296,
                              0.521, 5.332])
        P[7, 1:10] = np.array([4.418, 2.360, 0.677, 2.253, 3.472, 3.040,
                               3.974, 1.192, 2.510])
        P[8, 1:11] = np.array([5.041, 4.285, 3.001, 5.765, 4.295, 0.056,
                               4.213, 6.040, 1.078, 2.759])
        P[9, 1:12] = np.array([2.755, 5.491, 4.447, 0.231, 2.499, 3.539,
                               2.931, 2.759, 5.376, 4.554, 3.479])
        P[10, 1:13] = np.array([0.603, 0.009, 4.179, 4.361, 4.837, 0.816,
                                5.995, 4.150, 0.417, 1.520, 4.517, 1.729])
        P[11, 1:14] = np.array([3.997, 0.830, 5.712, 3.838, 0.084, 1.685,
                                5.328, 0.237, 0.506, 1.356, 4.025, 4.483,
                                4.084])
        P[12, 1:15] = np.array([4.126, 2.266, 0.957, 4.603, 0.815, 3.475,
                                0.977, 1.449, 1.192, 0.148, 0.939, 2.531,
                                3.612, 4.801])
        P[13, 1:16] = np.array([4.359, 3.510, 4.410, 1.750, 3.357, 2.061,
                                5.948, 3.000, 2.822, 0.627, 2.768, 3.875,
                                4.173, 4.224, 5.941])

        out = P[n_bands - 3, 0:n_bands]

    elif phs_type == 'amp_mod':

        # Malik's Hermitian phases: From S J Malik, ISMRM 2015, p. 2398
        if n_bands < 4 or n_bands > 12:
            raise Exception('Maliks phases valid for 3 < nBands < 13.')

        P = np.zeros((9, 12))
        P[0, 0:4] = np.array([0, np.pi, np.pi, 0])
        P[1, 0:5] = np.array([0, 0, np.pi, 0, 0])
        P[2, 0:6] = np.array([1.691, 2.812, 1.157, -1.157, -2.812, -1.691])
        P[3, 0:7] = np.array([2.582, -0.562, 0.102, 0, -0.102, 0.562, -2.582])
        P[4, 0:8] = np.array([2.112, 0.220, 1.464, 1.992, -1.992, -1.464,
                              -0.220, -2.112])
        P[5, 0:9] = np.array([0.479, -2.667, -0.646, -0.419, 0, 0.419, 0.646,
                              2.667, -0.479])
        P[6, 0:10] = np.array([1.683, -2.395, 2.913, 0.304, 0.737, -0.737,
                               -0.304, -2.913, 2.395, -1.683])
        P[7, 0:11] = np.array([1.405, 0.887, -1.854, 0.070, -1.494, 0, 1.494,
                               -0.070, 1.854, -0.887, -1.405])
        P[8, 0:12] = np.array([1.729, 0.444, 0.722, 2.190, -2.196, 0.984,
                               -0.984, 2.196, -2.190, -0.722, -0.444, -1.729])

        out = P[n_bands - 4, 0:n_bands]

    elif phs_type == 'quad_mod':

        # Grissom's quadratic phases (unpublished)
        k = 3.4 / n_bands  # quadratic phase coefficient
        out = k * (np.arange(0, n_bands, 1) - (n_bands - 1) / 2) ** 2

    else:
        raise Exception('phase type ("{}") is not recognized.'.format(phs_type))

    return out
