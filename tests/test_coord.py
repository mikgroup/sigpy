import unittest
import numpy as np
import numpy.testing as npt
from sigpy import util, coord

if __name__ == '__main__':
    unittest.main()

class TestCoord(unittest.TestCase):

    def test_normal(self):

        x = util.randn((10, 1))
        y = util.randn((10, 1))
        z = util.randn((10, 1))

        r, theta, phi = coord.cartes_to_spheri(x, y, z)
        xn, yn, zn = coord.spheri_to_cartes(r, theta, phi)

        npt.assert_allclose(x, xn)
        npt.assert_allclose(y, yn)
        npt.assert_allclose(z, zn)