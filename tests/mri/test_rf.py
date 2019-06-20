import unittest
import numpy as np
import sigpy as sp
import numpy.testing as npt

from sigpy.mri import rf, linop, sim

if __name__ == '__main__':
    unittest.main()

class TestRf(unittest.TestCase):

    def test_stspa_cartesian(self):

        img_shape = [32, 32]
        sens_shape = [8, 32, 32]

        x, y = np.ogrid[-img_shape[0] / 2: img_shape[0] - img_shape[0] / 2,
               -img_shape[1] / 2: img_shape[1] - img_shape[1] / 2]
        circle = x * x + y * y <= int(img_shape[0] / 6) ** 2
        target = np.zeros(img_shape, np.complex)
        target[circle] = 1

        sens = sim.birdcage_maps(sens_shape)

        mask = np.zeros(sens_shape)
        mask[:, ::2, :] = 1.0

        A = linop.Sense(sens, coord=None, weights=mask, ishape=target.shape).H

        pulses = rf.stspa(target,sens,mask,coord=None,max_iter=50,tol=1E-3)

        npt.assert_array_almost_equal(A*pulses,target,1E-3)

    def test_stspa_radial(self):

        img_shape = [32, 32]
        sens_shape = [8, 32, 32]

        x, y = np.ogrid[-img_shape[0] / 2: img_shape[0] - img_shape[0] / 2,
               -img_shape[1] / 2: img_shape[1] - img_shape[1] / 2]
        circle = x * x + y * y <= int(img_shape[0] / 6) ** 2
        target = np.zeros(img_shape, np.complex)
        target[circle] = 1

        sens = sim.birdcage_maps(sens_shape)

        traj = sp.mri.radial((sens.shape[1],sens.shape[1],2),img_shape,golden=True,dtype=np.float)

        mask = np.zeros(img_shape)

        x,y = 0,0
        for i in range(img_shape[1]):
            for j in range(img_shape[1]):
                x = traj[i, j, 0] + img_shape[1] / 2
                y = traj[i, j, 1] + img_shape[1] / 2

                mask[int(y), int(x)] = 1

        fullmask = np.repeat(mask[np.newaxis, :, :], 8, axis=0)

        A = linop.Sense(sens, coord=traj, weights=fullmask, ishape=target.shape).H

        pulses = rf.stspa(target,sens,fullmask,coord=traj,max_iter=25,tol=1E-3)

        npt.assert_array_almost_equal(A*pulses,target,1E-3)

    def test_stspa_spiral(self):

        img_shape = [32, 32]
        sens_shape = [8, 32, 32]

        x, y = np.ogrid[-img_shape[0] / 2: img_shape[0] - img_shape[0] / 2,
               -img_shape[1] / 2: img_shape[1] - img_shape[1] / 2]
        circle = x * x + y * y <= int(img_shape[0] / 6) ** 2
        target = np.zeros(img_shape, np.complex)
        target[circle] = 1

        sens = sim.birdcage_maps(sens_shape)

        traj = sp.mri.spiral(fov=1, img_shape=img_shape, f_sampling=1, R=1, ninterleaves=5, alpha=1.5, gm=0.03, sm=200)

        mask = np.zeros(img_shape)

        x,y = 0,0
        for i in range(img_shape[1]):
            for j in range(img_shape[1]):
                x = traj[i, j, 0] + img_shape[1] / 2
                y = traj[i, j, 1] + img_shape[1] / 2

                mask[int(y), int(x)] = 1

        fullmask = np.repeat(mask[np.newaxis, :, :], 8, axis=0)

        A = linop.Sense(sens, coord=traj, weights=fullmask, ishape=target.shape).H

        pulses = rf.stspa(target,sens,fullmask,coord=traj,max_iter=25,tol=1E-3)

        npt.assert_array_almost_equal(A*pulses,target,1E-3)