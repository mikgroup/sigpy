import unittest

import numpy as np
import numpy.testing as npt

import sigpy as sp
import sigpy.mri.rf as rf

if __name__ == '__main__':
    unittest.main()


class TestSlr(unittest.TestCase):

    def test_st(self):
        #  check to make sure profile roughly matches anticipated within d1, d2
        N = 128
        tb = 16
        filts = ['ls', 'ms', 'pm', 'min', 'max']
        for idx, filt in enumerate(filts):
            pulse = sp.mri.rf.dzrf(N, tb, ptype='st', ftype=filt,
                                   d1=0.01, d2=0.01)

            m = np.abs(sp.fft(pulse, norm=None))

            pts = np.array([m[int(N / 2 - 10)], m[int(N / 2)],
                            m[int(N / 2 + 10)]])
            npt.assert_almost_equal(pts, np.array([0, 1, 0]), decimal=2)

    def test_inv(self):
        #  also provides testing of sim. Check inv profile.
        tb = 8
        N = 128
        d1 = 0.01
        d2 = 0.01
        ptype = 'ex'
        filts = ['min', 'max']  # filts produce inconsistent inversions

        for idx, filt in enumerate(filts):
            pulse = rf.slr.dzrf(N, tb, ptype, filt, d1, d2)

            [_, b] = rf.sim.abrm(pulse, np.arange(-2 * tb, 2 * tb, 0.01))
            mz = 1 - 2 * np.abs(b) ** 2

            pts = np.array([mz[int(len(mz) / 2 - len(mz)/3)],
                            mz[int(len(mz) / 2)],
                            mz[int(len(mz) / 2 + len(mz)/3)]])

            npt.assert_almost_equal(pts, np.array([1, -0.2, 1]), decimal=1)

    def test_root_flipped(self):
        tb = 12
        N = 128
        d1 = 0.01
        d2 = 0.001
        flip = np.pi / 2
        ptype = 'ex'
        [bsf, d1, d2] = rf.slr.calc_ripples(ptype, d1, d2)
        b = bsf * rf.slr.dzmp(N, tb, d1, d2)
        b = b[::-1]
        [pulse, _] = rf.slr.root_flip(b, d1, flip, tb,
                                      verbose=False)

        [_, b] = rf.sim.abrm(pulse, np.arange(-2 * tb, 2 * tb, 0.01))
        mz = 1 - 2 * np.abs(b) ** 2

        pts = np.array([mz[int(len(mz) / 2 - len(mz) / 3)],
                        mz[int(len(mz) / 2)],
                        mz[int(len(mz) / 2 + len(mz) / 3)]])

        npt.assert_almost_equal(pts, np.array([1, 0.2, 1]), decimal=1)

    def test_recursive(self):
        # Design the pulses
        nseg = 3  # number of EPI segments/RF Pulses
        tb = 4
        n = 200
        se_seq = True
        tb_ref = 8  # time-bandwidth of ref pulse
        [pulses, _] = rf.slr.dz_recursive_rf(nseg, tb, n, se_seq, tb_ref)

        mz = np.ones(np.size(np.arange(-4 * tb, 4 * tb, 0.01)))
        for ii in range(0, nseg):
            [a, b] = rf.sim.abrm(pulses[:, ii],
                                 np.arange(-4 * tb, 4 * tb, 0.01), True)
            mxy = 2 * mz * np.multiply(np.conj(a), b)
            mz = mz * (1 - 2 * np.abs(b) ** 2)

            pts = np.array([mxy[int(len(mxy) / 2 - len(mxy) / 3)],
                            mxy[int(len(mxy) / 2)],
                            mxy[int(len(mxy) / 2 + len(mxy) / 3)]])

            npt.assert_almost_equal(abs(pts), np.array([0, 0.5, 0]), decimal=1)

    def test_gslider(self):
        n = 512
        g = 5
        ex_flip = 90 * np.pi / 180
        tb = 12
        d1 = 0.01
        d2 = 0.01
        phi = np.pi

        pulses = rf.slr.dz_gslider_rf(n, g, ex_flip, phi, tb, d1, d2,
                                      cancel_alpha_phs=True)

        for gind in range(1, g + 1):
            [a, b] = rf.sim.abrm(pulses[:, gind - 1],
                                 np.arange(-2 * tb, 2 * tb, 0.01), True)
            mxy = 2 * np.multiply(np.conj(a), b)

            pts = np.array([mxy[int(len(mxy) / 2 - len(mxy) / 3)],
                            mxy[int(len(mxy) / 2)],
                            mxy[int(len(mxy) / 2 + len(mxy) / 3)]])

            npt.assert_almost_equal(abs(pts), np.array([0, 1, 0]), decimal=2)
