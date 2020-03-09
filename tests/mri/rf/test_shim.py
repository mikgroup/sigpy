import unittest
import numpy as np
import sigpy as sp
import numpy.testing as npt
import scipy.ndimage.filters as filt

from sigpy.mri import rf, linop, sim

if __name__ == '__main__':
    unittest.main()


class TestPtx(unittest.TestCase):

    def test_calc_shims(self):
        dim = 8
        dt =4e-6
        img_shape = [dim, dim]
        sens_shape = [8, dim, dim]

        # target - slightly blurred circle
        x, y = np.ogrid[-img_shape[0] / 2: img_shape[0] - img_shape[0] / 2,
                        -img_shape[1] / 2: img_shape[1] - img_shape[1] / 2]
        circle = x * x + y * y <= int(img_shape[0] / 6) ** 2
        target = np.zeros(img_shape)
        target[circle] = 1

        sens = sim.birdcage_maps(sens_shape)

        shims = sp.mri.rf.calc_shims(target, sens, dt, lamb=0, max_iter=50,
                                     minibatch=False, batchsize=1)
