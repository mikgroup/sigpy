import unittest
import numpy as np
import numpy.testing as npt

from sigpy.mri import sim

if __name__ == '__main__':
    unittest.main()


class TestSim(unittest.TestCase):

    def test_multi_gradient_echoes(self):
        # echo time
        TE0 = 1.70
        ESP = 1.52
        N_eco = 35

        TE = (TE0 + np.arange(N_eco) * ESP) * 1E-3

        # dictionary simulation
        dict = sim.gradient_echoes(TE)
        U = sim.get_subspace(dict, num_coeffs=22, prior_err=False)

        # simulate a tri-exponential signal
        rho = [0.30, 0.30, 0.40]
        T2 = [0.02, 0.01, 0.10]  # second
        B0 = [50, 100, -20]  # Hz

        sig = np.zeros_like(TE, dtype=complex)

        for a, b, c in zip(rho, T2, B0):
            sig += a * np.exp(-TE / b) * np.exp(1j*2*np.pi * c * TE)

        recon_sig = U @ U.T @ sig

        npt.assert_allclose(recon_sig, sig, atol=1e-3, rtol=1e-3)
