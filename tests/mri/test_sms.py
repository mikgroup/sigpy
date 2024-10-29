import unittest
import numpy as np
import numpy.testing as npt

from sigpy.mri import sms

if __name__ == '__main__':
    unittest.main()


class TestSms(unittest.TestCase):

    def test_slice_order(self):
        list_NS = [94, 114, 60, 159, 141, 117]
        list_MB = [2, 3, 2, 3, 3, 3]

        for N_slices_uncollap, MB in zip(list_NS, list_MB):

            print('****** total slices: ' + str(N_slices_uncollap)
                  + ', multi-band: ' + str(MB) + ' ******')

            N_slices_collap = N_slices_uncollap // MB

            slice_mb_idx = []
            for s in range(N_slices_collap):

                slice_mb_idx += sms.map_acquire_to_ordered_slice_idx(s, N_slices_uncollap, MB, verbose=True)

            slice_mb_idx.sort()

            npt.assert_allclose(slice_mb_idx,
                                range(N_slices_uncollap),
                                err_msg = ['error in slice ordering for MB = '
                                           + str(MB) + ' slices = '
                                           + str(N_slices_uncollap).zfill(3)]
                                )

    def test_reorder(self):
        N_slices = 114
        N_band = 3

        N_slices_collap = N_slices // N_band

        img_shape = [4, 4]

        I = np.zeros([N_slices_collap, N_band] + img_shape)

        for s in range(N_slices_collap):
            slice_mb_idx = sms.map_acquire_to_ordered_slice_idx(s, N_slices, N_band, verbose=True)

            for b in range(N_band):
                idx = slice_mb_idx[b]
                I[s, b, :, :] = idx

        O = sms.reorder_slices_mbx(I, N_band, N_slices)

        for s in range(N_slices):
            print('slice idx ' + str(s) + '; value ' + str(O[s, 0, 0]))
