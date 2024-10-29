import unittest
import numpy.testing as npt
from sigpy import backend, config, util, linop, prox

from sigpy.mri import nlop, epi, nlrecon

if __name__ == '__main__':
    unittest.main()

devices = [backend.cpu_device]
if config.cupy_enabled:
    devices.append(backend.Device(0))

class TestNlls(unittest.TestCase):

    def test_diff_nlls(self):
        for device in devices:
            xp = device.xp

            b = xp.load('diff_b.npy')
            g = xp.load('diff_g.npy')

            for model in ['dti', 'dki']:
                if model == 'dti':
                    B = epi.get_B(b, g)
                elif model == 'dki':
                    B = epi.get_B2(b, g)

                num_params = B.shape[1]

                D = util.randn((num_params, 16, 16),
                            dtype=float,
                            device=device) / 1E8

                D = D + 0. * 1j  # real numbers

                coils = util.randn((8, 1, 16, 16),
                                dtype=complex,
                                device=device)

                F = nlop.Diffusion(D.shape, B, coil=coils)
                y = F(D) + util.randn(F.oshape) * 1e-5

                x = nlrecon.kinv(y,
                                D.shape, coils.shape,
                                coil=coils,
                                outer_iter=8, redu=3,
                                inner_iter=100,
                                model='Diffusion',
                                sample_time=B
                                ).run()

                npt.assert_allclose(x, D,
                                    rtol=1e-5, atol=1e-7,
                                    err_msg='Diffusion Nlls failed!')
