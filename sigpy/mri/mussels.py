# -*- coding: utf-8 -*-
"""
MUSSELS Diffusion MRI Reconstruction.

Author: Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import numpy as np
import sigpy as sp

from sigpy import backend
from sigpy.mri import app, sms


def MusselsRecon(y, coils, MB=1,
                 lamda=0.05, max_iter=50, rho=0.1,
                 regu='SLR', regu_kspace=True, thresh='hard',
                 blk_shape=[1, 7, 7], blk_strides=[1, 1, 1],
                 use_readout_extend_fov=False, ro_extend_fold=1,
                 yshift=None,
                 verbose=True,
                 device=sp.cpu_device):
    """MUSSELS Reconstruction.

    Args:
        y (array): measured k-space data.
        coils (array): coil sensitivity maps.
        MB (int): multi-band factor.
        lamda (float): regularization strength.
        max_iter (int): maximal iteration steps.
        rho (float): ADMM rho.
        regu (string): regularization type ('SLR', 'TIK', 'LLR', 'TV').
        regu_kspace (boolean): regularization performed in k-space.
        thresh (string): thresholding type ('hard', 'soft').
        blk_shape (tuple or list of int): block shape (default: [7, 7]).
        blk_strides (tuple or list of int): stride shape (default: [1, 1]).
        use_readout_extend_fov (boolean): use the method
        readout extended fov for image reconstruction.
        ro_extend_fold (int): unextend readout extended FOV
        for regularization (1 or MB).
        verbose (boolean): output ADMM iteration info.
        device (int): cpu or gpu device to run the recon.

    References:
        * Mani M, Jacob M, Kelley D, Magnotta V.
          Multi-shot sensitivity-encoded diffusion data recovery using
          structured low-rank matrix completion (MUSSELS).
          Magn Reson Med 2017;78:494-507.

        * Bilgic B, Chatnuntawech I, Manhard MK, Tian Q, Liao C, Iyer SS,
          Cauley SF, Huang SY, Polimeni JR, Wald LL, Setsompop K.
          High accelerated multishot echo planar imaging through
          synergistic machine learning and joint reconstruction.
          Magn Reson Med 2019;82:1343-1358.

        * Mani M, Aggarwal HK, Magnotta V, Jacob M.
          Improved MUSSELS reconstruction for high-resolution
          multi-shot diffusion weighted imaging.
          Magn Reson Med 2020;83:2253-2263.

        * Dai E, Mani M, McNab JA.
          Multi-band multi-shot diffusion MRI reconstruction
          with joint usage of structured low-rank constraints
          and explicit phase mapping.
          Magn Reson Med 2023.
    """
    Ndiff, Nshot, Ncoil, Nz_collap, Ny, Nx = y.shape
    assert (Nshot > 1)  # MUSSELS is a multi-shot technique

    _Ncoil, Nz, _Ny, _Nx = coils.shape

    assert ((Ncoil == _Ncoil) and (Ny == _Ny) and (Nx == _Nx))
    assert ((Nz_collap == Nz / MB))

    output = []
    for z in range(Nz_collap):  # loop over collapsed k-space

        slice_idx = sms.get_uncollap_slice_idx(Nz, MB, z)
        mps = coils[:, slice_idx, ...]

        ksp_slice = y[..., z, :, :]  # 5

        for d in range(Ndiff):  # loop over diffusion encodings (DWI)

            print('>> mussels on slice ' + str(z).zfill(2)
                  + ' diff ' + str(d).zfill(3))

            ksp = ksp_slice[d, ...]  # 4

            # readout extended FOV
            if use_readout_extend_fov is True:

                ksp_ext, mps_ext, msk_ext = sms.readout_extended_fov(ksp,
                                                                     mps,
                                                                     MB)

                ksp_ext = np.expand_dims(ksp_ext, axis=(-3, 0))  # 6
                msk_ext = np.expand_dims(msk_ext, axis=(-3, 0))  # 6
                mps_ext = np.expand_dims(mps_ext, axis=(-3, ))

                sms_phase = sms.get_sms_phase_shift([1, Ny, Nx * MB],
                                                    MB=1,
                                                    yshift=[0])

            else:

                ksp_ext = ksp[None, :, :, None, :, :]  # 6 dim
                mps_ext = mps.copy()
                msk_ext = None

                sms_phase = sms.get_sms_phase_shift([MB, Ny, Nx],
                                                    MB=MB,
                                                    yshift=yshift)

            # structured low rank matrix completion (SLRMC) as a regularizer
            img_ext = app.CompressedSenseRecon(
                ksp_ext, mps_ext, lamda, max_iter=max_iter,
                combine_echo=False, weights=msk_ext,
                phase_sms=sms_phase,
                regu=regu, regu_kspace=regu_kspace,
                blk_shape=blk_shape, blk_strides=blk_strides,
                thresh=thresh, solver='ADMM', rho=rho,
                ro_extend_fold=ro_extend_fold,
                verbose=verbose,
                device=device).run()

            img = backend.to_device(img_ext)

            if use_readout_extend_fov is True:
                img = sms.readout_unextend_fov(img, MB)
                phi = sms.get_sms_phase_shift(img.shape, MB, yshift=yshift)
                img = sp.ifft(np.conj(phi) * sp.fft(img, axes=[-2, -1]),
                              axes=[-2, -1])

            output.append(img)

    output = np.array(output)

    return output
