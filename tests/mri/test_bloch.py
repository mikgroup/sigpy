import unittest
import numpy as np
import numpy.testing as npt

from sigpy.mri import bloch

if __name__ == '__main__':
    unittest.main()


class TestBloch(unittest.TestCase):

    def test_to_density_matrix(self):
        m = np.array([0, 0, 1])
        npt.assert_allclose(bloch.to_density_matrix(m),
                            [[1, 0],
                             [0, 0]])

        m = np.array([0, 1, 0])
        npt.assert_allclose(bloch.to_density_matrix(m),
                            [[0.5, -0.5j],
                             [0.5j, 0.5]])

        m = np.array([1, 0, 0])
        npt.assert_allclose(bloch.to_density_matrix(m),
                            [[0.5, 0.5],
                             [0.5, 0.5]])

    def test_to_bloch_vector(self):
        p = np.array([[1, 0],
                      [0, 0]])
        npt.assert_allclose(bloch.to_bloch_vector(p),
                            [0, 0, 1])

        p = np.array([[0.5, -0.5j],
                      [0.5j, 0.5]])
        npt.assert_allclose(bloch.to_bloch_vector(p),
                            [0, 1, 0])

        p = np.array([[0.5, 0.5],
                      [0.5, 0.5]])
        npt.assert_allclose(bloch.to_bloch_vector(p),
                            [1, 0, 0])

    def test_free_induction_decay(self):
        f0 = 0
        t1 = 100e-3
        t2 = 50e-3
        dt = 4e-6

        p = bloch.init_density_matrix()
        pN = bloch.free_induction_decay(p, f0, t1, t2, dt)
        npt.assert_allclose(pN, [[1, 0],
                                 [0, 0]])

        p = np.array([[1 / 2, 1 / 2],
                      [1 / 2, 1 / 2]], np.complex)
        pN = bloch.free_induction_decay(p, f0, t1, t2, dt)
        e1 = np.exp(-dt / t1)
        e2 = np.exp(-dt / t2)
        npt.assert_allclose(pN, [[1 - e1 + 0.5 * e1, 0.5 * e2],
                                 [0.5 * e2, 0.5 * e1]])

    def test_hard_pulse_rotation(self):

        p = bloch.init_density_matrix()

        # 90 degree along x
        b1 = np.pi / 2
        pN = bloch.hard_pulse_rotation(p, b1)
        npt.assert_allclose(pN, [[1 / 2, 1j / 2],
                                 [-1j / 2, 1 / 2]])

        # 90 degree along y
        b1 = 1j * np.pi / 2
        pN = bloch.hard_pulse_rotation(p, b1)
        npt.assert_allclose(pN, [[1 / 2, 1 / 2],
                                 [1 / 2, 1 / 2]])

        # 180 degree along x
        b1 = np.pi
        pN = bloch.hard_pulse_rotation(p, b1)
        npt.assert_allclose(pN, [[0, 0],
                                 [0, 1]], atol=1e-10)

        # 180 degree along y
        b1 = 1j * np.pi
        pN = bloch.hard_pulse_rotation(p, b1)
        npt.assert_allclose(pN, [[0, 0],
                                 [0, 1]], atol=1e-10)

    def test_bloch_forward(self):
        dt = 1
        N = 1000
        b1 = np.full(N, np.pi / 2 / N)

        f0 = 0
        t1 = np.infty
        t2 = np.infty

        p0 = bloch.init_density_matrix()
        pN = bloch.bloch_forward(p0, b1, f0, t1, t2, dt)

        npt.assert_allclose(pN, [[0.5, 0.5j],
                                 [-0.5j, 0.5]], atol=1e-5)
