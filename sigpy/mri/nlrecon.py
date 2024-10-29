# -*- coding: utf-8 -*-
"""Non-Linear MRI reconstruction applications.
"""
import numpy as np

import sigpy as sp
from sigpy import backend, nlls, fourier
from sigpy.mri import nlop, app, epi


class kinv(nlls.NonLinearLeastSquares):
    r"""non-linear mri reconstruction.

    Args:
    """
    def __init__(self, y, image_shape, coil_shape,
                 coil=None, W_coil=True, upd_coil=True,
                 coord=None, x=None, x0=None,
                 rvc=False, dwi_phase=None, weights=None,
                 model='Diffusion', sample_time=None,
                 device=backend.cpu_device,
                 outer_iter=6, alpha=1., redu=2.,
                 trafos=None, proxf=None,
                 inner_iter=100, inner_tol=0.01,
                 scaling=False,
                 **kwargs):

        y = sp.to_device(y, device=device)

        xp = device.xp

        if model == 'Nlinv':
            A = nlop.Nlinv(image_shape, coil_shape,
                           coord=coord, coil=coil, W_coil=W_coil,
                           upd_coil=upd_coil)

            if x is None:
                with device:
                    x = xp.ones(A.ishape) * 0.1

        elif model == 'Diffusion':
            # estimate coil
            if coil is None:
                None

            with device:
                weights = app._estimate_weights(y, weights, coord)

            A = nlop.Diffusion(image_shape, sample_time, coil,
                               rvc=rvc, dwi_phase=dwi_phase,
                               weights=weights)

            if x is None:
                with device:
                    if image_shape[0] == 6 or image_shape[0] == 21:
                        x = xp.ones(A.ishape, dtype=y.dtype) * 0.
                    else:
                        x_b0 = xp.ones([1] + list(image_shape[1:]), dtype=y.dtype) * 1E-5
                        x_D = xp.zeros([image_shape[0]-1] + list(image_shape[1:]), dtype=y.dtype)

                        x = xp.concatenate((x_b0, x_D))

            if x0 is None:
                with device:
                    x0 = 0.9 * x
                    # x0 = xp.zeros(A.ishape, dtype=y.dtype)

        super().__init__(A, y, x=x, x0=x0,
                         outer_iter=outer_iter,
                         alpha=alpha, redu=redu,
                         trafos=trafos, proxf=proxf,
                         inner_iter=inner_iter,
                         inner_tol=inner_tol,
                         **kwargs)

    def _estimate_scaling(self):
        """Estimate scaling of unknowns.
        """
        return None

    def run(self):
        """Run non-linear reconstruction.
        """
        while not self.done():
            self.update()

        return self.x
