import unittest
import numpy as np
import numpy.testing as npt

from sigpy.mri import samp

if __name__ == '__main__':
    unittest.main()


class TestPoisson(unittest.TestCase):
    """Test poisson undersampling defined in `sigpy.mri.samp.poisson`."""

    def test_numpy_random_state(self):
        """Verify that random state is unchanged when seed is specified."""
        np.random.seed(0)
        expected_state = np.random.get_state()

        _ = samp.poisson((120, 120), accel=6, seed=80)

        state = np.random.get_state()
        assert (expected_state[1] == state[1]).all()

    def test_reproducibility(self):
        """Verify that poisson is reproducible."""
        np.random.seed(45)
        mask1 = samp.poisson((120, 120), accel=6, seed=80)

        # Changing internal numpy state should not affect mask.
        np.random.seed(20)
        mask2 = samp.poisson((120, 120), accel=6, seed=80)

        npt.assert_allclose(mask2, mask1)

    def test_poisson_accel(self):
        """Verify that poisson generates the correct acceleration."""
        for x in [60, 120]:
            for y in [60, 120]:
                for tol in [0.1, 0.2]:
                    for accel in [4, 5, 6, 7, 8]:
                        mask = samp.poisson(
                            (x, y), accel=accel, seed=80, tol=tol)
                        assert abs(mask.size / np.sum(mask) - accel) < tol
