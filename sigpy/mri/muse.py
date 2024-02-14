"""
MUSE Reconstruction.

Author: Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import numpy as np
import sigpy as sp

from sigpy import backend
from sigpy.mri import app, sms
from sigpy.mri.dims import *


# %%
def _denoising(input, full_img_shape=None, use_iter=True, max_iter=5):
    """
    Args:
        input: acs images
        full_img_shape: shape of full-FOV images
    """

    # # Hanning
    if full_img_shape is None:
        full_img_shape = input.shape[-2:]

    device = backend.get_device(input)
    xp = device.xp

    with device:

        H = sp.hanning(input.shape[-2:], dtype=complex, symm=True,
                       device=device)

        H_full = sp.resize(H, full_img_shape)

        k_full = sp.resize(sp.fft(input, axes=[-2, -1]),
                           oshape=list(input.shape[:-2])
                           + list(full_img_shape))


        if use_iter:
            for m in range(max_iter):
                k_full = H_full * k_full
        else:
            k_full = H_full**max_iter * k_full


        img = sp.ifft(k_full, axes=[-2, -1])

        idx = abs(img) > 0
        phs = xp.zeros_like(img)
        phs[idx] = img[idx] / abs(img[idx])

    return img, phs


def sms_sense_linop(kdat, coils, yshift, phase_echo=None):

    device = backend.get_device(kdat)
    assert (device == backend.get_device(coils))

    Ncoil, Nz, Ny, Nx = coils.shape

    assert (Nz == len(yshift))

    phase_sms = sms.get_sms_phase_shift([Nz, Ny, Nx], Nz, yshift=yshift)

    img_shape = [1, Nz, Ny, Nx]

    if phase_echo is not None:

        P = sp.linop.Multiply(img_shape,
                              sp.to_device(phase_echo, device=device))

    else:

        P = sp.linop.Identity(img_shape)

    # coils
    S = sp.linop.Multiply(P.oshape, coils)

    # FFT
    F = sp.linop.FFT(S.oshape, axes=range(-2, 0))

    # SMS
    PHI = sp.linop.Multiply(F.oshape, sp.to_device(phase_sms, device=device))
    SUM = sp.linop.Sum(PHI.oshape, axes=(DIM_Z, ), keepdims=True)

    M = SUM * PHI

    weights = app._estimate_weights(kdat, None, None, coil_dim=DIM_COIL)

    W = sp.linop.Multiply(M.oshape, weights**0.5)

    return W * M * F * S * P


def sms_sense_solve(A, y, lamda=0.01, tol=0, max_iter=30, verbose=False):

    device = backend.get_device((y))
    xp = device.xp

    AHA = lambda x: A.N(x) + lamda * x
    AHy = A.H(y)

    img = xp.zeros(A.ishape, dtype=y.dtype)
    alg_method = sp.alg.ConjugateGradient(AHA, AHy, img,
                            tol=tol,
                            max_iter=max_iter, verbose=verbose)

    while (not alg_method.done()):
        alg_method.update()

    return img


# %%
def MuseRecon(y, coils, MB=1, acs_shape=[64, 64],
              lamda=0.001, max_iter=80, tol=0,
              use_readout_extend_fov=False, yshift=None,
              device=sp.cpu_device, verbose=False):
    """
    MUSE is a novel method to reconstruct one diffusion-weighted image (DWI)
    from multi-shot EPI acquisition. It consists of the following steps:
        1. shot-by-shot SENSE recon;
        2. phase estimation from every shot image;
        3. incorporate phase into phase-informed SENSE recon to obtain one DWI.

    Args:
        y (array): zero-filled k-space data with shape:
            [Nshot, Ncoil, Nz_collap, Ny, Nx], where
            - Nshot: # of shots per DWI,
            - Ncoil: # of coils,
            - Nz_collap: # of collapsed slices,
            - Ny: # of phase-encoding lines,
            - Nx: # of readout lines.

        coils (array): coil sensitivity maps with shape:
            [Ncoil, Nz, Ny, Nx], where
            - Nz: # of un-collapsed slices.

        MB (int): multi-band factor
            MB = Nz / Nz_collap.

        acs_shape (tuple of ints): shape of the auto-calibration signal (ACS),
            which is used for the shot-by-shot SENSE recon.

    References:
        * Liu C, Moseley ME, Bammer R.
          Simultaneous phase correction and SENSE reconstruction for navigated multi-shot DWI with non-Cartesian k-space sampling.
          Magn Reson Med 2005;54:1412-1422.

        * Chen NK, Guidon A, Chang HC, Song AW.
          A robust multi-shot strategy for high-resolution diffusion weighted MRI enabled by multiplexed sensitivity-encoding (MUSE).
          NeuroImage 2013;72:41-47.
    """
    Ndiff, Nshot, Ncoil, Nz_collap, Ny, Nx = y.shape
    assert(Nshot > 1)  # MUSE is a multi-shot technique

    _Ncoil, Nz, _Ny, _Nx = coils.shape

    assert ((Ncoil == _Ncoil) and (Ny == _Ny) and (Nx == _Nx))
    assert ((Nz_collap == Nz / MB))

    phi = sms.get_sms_phase_shift([MB, Ny, Nx], MB, yshift=yshift)

    if acs_shape is None:

        ksp_acs = y.copy()
        mps_acs = coils.copy()

    else:

        ksp_acs = sp.resize(y, oshape=list(y.shape[:-2]) + list(acs_shape))

        import torchvision.transforms as T

        coils_tensor = sp.to_pytorch(coils)
        TR = T.Resize(acs_shape)
        mps_acs_r = TR(coils_tensor[..., 0]).cpu().detach().numpy()
        mps_acs_i = TR(coils_tensor[..., 1]).cpu().detach().numpy()
        mps_acs = mps_acs_r + 1j * mps_acs_i

    print('**** MUSE - ksp_acs shape ', ksp_acs.shape)
    print('**** MUSE - mps_acs shape ', mps_acs.shape)

    R_muse = []
    R_shot = []
    for z in range(Nz_collap):  # loop over collapsed k-space

        slice_idx = sms.get_uncollap_slice_idx(Nz, MB, z)
        mps_acs_slice = mps_acs[:, slice_idx, ...]

        for d in range(Ndiff):

            print('>> muse on slice ' + str(z).zfill(2) + ' diff ' + str(d).zfill(3))

            if use_readout_extend_fov:

                # 1. perform shot-by-shot ACS SENSE recon to estimate phase
                img_ext_shots = []
                for s in range(Nshot):  # loop over every shot

                    ksp = ksp_acs[d, s, :, z, ...]

                    ksp_ext, mps_ext, _ = sms.readout_extended_fov(ksp, mps_acs_slice, MB)

                    img_ext = app.SenseRecon(ksp_ext, mps_ext, 5E-5,
                                max_iter=90, tol=0,
                                device=device).run()

                    img_ext_shots.append(backend.to_device(img_ext))

                img_ext_shots = np.array(img_ext_shots)
                R_shot.append(img_ext_shots)

                # 2. phase estimation from shot images
                img_ext_shots_den, phs_ext_shots = _denoising(img_ext_shots, full_img_shape=[Ny, Nx * MB])

                img_ini = abs(np.mean(img_ext_shots_den * np.conj(phs_ext_shots), axis=0)).astype(phs_ext_shots.dtype)

                # 3. perform phase-informed SENSE recon
                # to estimate shot-combined DWI
                ksp = y[d, :, :, z, ...]
                mps = coils[:, slice_idx, ...]
                ksp_ext, mps_ext, _ = sms.readout_extended_fov(ksp, mps, MB)

                phs_ext_shots = np.expand_dims(phs_ext_shots, axis=1)
                phs_ext_shots_mps = phs_ext_shots * mps_ext
                phs_ext_shots_mps = phs_ext_shots_mps.reshape((-1, Ny, Nx * MB))

                # -- calculate weights for k-space
                arr1 = np.ones((Ncoil, Ny, Nx * MB))
                weights = (sp.rss(ksp_ext, axes=(1, ), keepdims=True) > 0.).astype(ksp_ext.dtype)
                weights = arr1 * weights

                ksp_ext = ksp_ext.reshape((-1, Ny, Nx * MB))
                weights = weights.reshape((-1, Ny, Nx * MB))

                img_ext = app.SenseRecon(ksp_ext, phs_ext_shots_mps,
                                         lamda, max_iter=max_iter, tol=tol,
                                         weights=weights,
                                         x=img_ini,
                                         device=device).run()

                img = sms.readout_unextend_fov(backend.to_device(img_ext), MB)
                # img = sp.ifft(np.conj(phi) * sp.fft(img, axes=[-2, -1]), axes=[-2, -1])
                R_muse.append(img)

            else:

                xp = device.xp
                ksp_acs = sp.to_device(ksp_acs, device=device)
                mps_acs_slice = sp.to_device(mps_acs_slice, device=device)

                y = sp.to_device(y, device=device)
                coils = sp.to_device(coils, device=device)

                # 1. perform shot-by-shot ACS SENSE recon to estimate phase
                img_acs_shots = []
                for s in range(Nshot):

                    ksp = ksp_acs[d, s, :, z, :, :]
                    ksp = ksp[..., None, :, :]

                    A = sms_sense_linop(ksp, mps_acs_slice, yshift)

                    img = sms_sense_solve(A, ksp, lamda=5E-5, tol=0,
                                          max_iter=max_iter, verbose=verbose)

                    img_acs_shots.append(img)

                img_acs_shots = xp.array(img_acs_shots)
                R_shot.append(sp.to_device(img_acs_shots))

                # 2. phase estimation from shot images
                _, phs_shots = _denoising(img_acs_shots,
                                          full_img_shape=[Ny, Nx])

                # 3. perform phase-informed SENSE recon
                # to estimate shot-combined DWI
                ksp = y[d, :, :, z, ...]
                ksp = ksp[..., None, :, :]
                mps = coils[:, slice_idx, ...]

                A = sms_sense_linop(ksp, mps, yshift, phase_echo=phs_shots)

                img = sms_sense_solve(A, ksp, lamda=lamda, tol=tol,
                                      max_iter=max_iter, verbose=verbose)

                R_muse.append(sp.to_device(img))

    R_muse = np.array(R_muse)
    R_shot = np.array(R_shot)

    return R_muse, R_shot
