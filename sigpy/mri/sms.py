"""Functions for Simultaneous Multi-Slice (SMS) Acquisition

Author: Zhengguo Tan <zhengguo.tan@gmail.com>
"""
import numpy as np
import sigpy as sp

__all__ = ['get_uncollap_slice_idx',
           'get_ordered_slice_idx',
           'reorder_slices',
           'get_sms_phase_shift',
           'readout_extended_fov',
           'readout_unextend_fov']


def is_even(input):
    return (input % 2 ==0)


def map_acquire_to_ordered_slice_idx(acq_slice_idx,
                                     N_slices_uncollap, N_band,
                                     verbose=False):
    """
    Map the acquired slice index to ordered uncollapsed slice indices.

    Args:
        acq_slice_idx (int):
            acquired slice index
        N_slices_uncollap (int):
            total number of uncollapsed slices
        N_band (int):
            multi-band factor

    Output:
        ordered uncollapsed slice indices (list of int)
    """
    N_slices_collap = N_slices_uncollap // N_band
    N_slices_collap_half = N_slices_collap // 2

    ord_slice_idx = []
    for b in range(N_band):

        # interleaved slice order
        if (acq_slice_idx >= N_slices_collap_half) and \
            (is_even(N_slices_collap)):
            so = acq_slice_idx * 2
        else:
            so = acq_slice_idx * 2 + 1

        so = (so + b * N_slices_collap) % N_slices_uncollap
        ord_slice_idx.append(so)

    # need this when MB=3 and N_slices_uncollap=141
    # if (acq_slice_idx != N_slices_collap_half) or (N_band % 2 == 0):
        # and (N_band % 2) and (N_slices_uncollap % 2)
    ord_slice_idx.sort()

    if verbose is True:
        print('acquired slice: ' + str(acq_slice_idx).zfill(3)
              + ' --> ordered slices: ' + str(ord_slice_idx))

    return ord_slice_idx


def get_uncollap_slice_idx(N_slices_uncollap, MB, collap_slice_idx):
    """
    Get uncollapsed slice indices for the collapsed slice "collap_slice_idx".

    Args:
        N_slices_uncollap (int): total number of uncollapsed slices.
        MB (int): multi-band factor.
        collap_slice_idx (int): collapsed slice index.

    e.g.:
        slice_idx = get_uncollap_slice_idx(30, 2, 0)
        returns [0, 7]
    """
    slice_idx = []
    N_slices_collap = N_slices_uncollap // MB

    if (collap_slice_idx < 0) or (collap_slice_idx >= N_slices_collap):
        raise ValueError('collap_slice_idx must be in the range: [0, '
                         + str(N_slices_collap) + ').')

    N_slices_collap_half = N_slices_collap // 2

    N_slices_1 = N_slices_collap_half * MB
    N_slices_2 = N_slices_uncollap - N_slices_1

    for b in range(MB):

        if collap_slice_idx < N_slices_collap_half:

            N_0 = collap_slice_idx
            N_step = N_slices_1 // MB

        else:

            N_0 = N_slices_1 + collap_slice_idx - N_slices_collap_half
            N_step = N_slices_2 // MB

        slice_idx.append(N_0 + b * N_step)

    return list(slice_idx)


def get_ordered_slice_idx(acq_slice_idx, N_slices):
    """
    Get the ordered (geometrically correct) slice indices
    from the acquisition slice order.

    Args:
        acq_slice_idx (int or tuple of ints): indices for acquired slices.
        N_slices (int): total number of slices.

    Output:
        ordered slice indices (list).
    """
    ordered_slice_idx = list(range(1, N_slices, 2)) \
        + list(range(0, N_slices, 2))

    if isinstance(acq_slice_idx, int):
        return ordered_slice_idx[acq_slice_idx]
    else:
        return [ordered_slice_idx[i] for i in list(acq_slice_idx)]


def reorder_slices_mb1(input, N_slices, slice_axis=-3):

    input = np.swapaxes(input, 0, slice_axis)

    img_shape = input.shape[1:]

    output = np.zeros_like(input, shape=[N_slices] + list(img_shape))

    for s in range(N_slices):
        ord_slice_idx = get_ordered_slice_idx(s, N_slices)

        print('acquired slice: ' + str(s).zfill(3)
              + ' --> geometric slice: ' + str(ord_slice_idx).zfill(3))

        output[ord_slice_idx, ...] = input[s, ...]

    output = np.swapaxes(output, 0, slice_axis)

    return output


def reorder_slices_mbx(input, N_band, N_slices, band_axis=-3, slice_axis=0):

    assert (N_band == input.shape[band_axis])

    N_slices_collap = N_slices // N_band

    assert (N_slices_collap == input.shape[slice_axis])

    # swap axes such that slice stored in axis 0, and band in axis 1
    input = np.swapaxes(input, 1, band_axis)
    input = np.swapaxes(input, 0, slice_axis)

    img_shape = input.shape[2:]  # image shape excluding slices and bands

    output = np.zeros_like(input, shape=[N_slices] + list(img_shape))

    for s in range(N_slices_collap):

        slice_mb_idx = map_acquire_to_ordered_slice_idx(s, N_slices, N_band,
                                                        verbose=True)

        output[slice_mb_idx, ...] = input[s, ...]

    output = np.swapaxes(output, 0, -3)

    return output


def reorder_slices(input, N_band, N_slices, band_axis=-3, slice_axis=0):
    """
    reorder slices after SMS image reconstruction.

    Requirement:
        band and slice must be stored in different axes

    Args:

    Output:

    """
    assert (N_band == input.shape[band_axis])

    N_slices_collap = N_slices // N_band

    assert (N_slices_collap == input.shape[slice_axis])

    # swap axes such that slice stored in axis 0, and band in axis 1
    input = np.swapaxes(input, 1, band_axis)
    input = np.swapaxes(input, 0, slice_axis)

    img_shape = input.shape[2:]  # image shape excluding slices and bands

    output = np.zeros_like(input, shape=[N_slices] + list(img_shape))

    for s in range(N_slices_collap):
        acq_slice_idx = get_uncollap_slice_idx(N_slices, N_band, s)
        ord_slice_idx = get_ordered_slice_idx(acq_slice_idx, N_slices)

        print('collapsed slice: ' + str(s).zfill(3)
              + ' --> acquired slices: '
              + str([str(sid).zfill(3) for sid in acq_slice_idx])
              + ' --> reordered slices: '
              + str([str(sid).zfill(3) for sid in ord_slice_idx]))

        output[ord_slice_idx, ...] = input[s, ...]

    # slice stored in axis -3
    output = np.swapaxes(output, 0, -3)

    return output


def get_sms_phase_shift(ishape, MB, yshift=None):
    """
    Args:
        ishape (tuple or list): input shape of [..., Nz, Ny, Nx].
        MB (int): multi-band factor.
        yshift (tuple or list): use custom yshift.

    References:
        * Breuer FA, Blaimer M, Heidemann RM, Mueller MF,
          Griswold MA, Jakob PM.
          Controlled aliasing in parallel imagin results in
          higher acceleration (CAIPIRINHA) for multi-slice imaging.
          Magn. Reson. Med. 53:684-691 (2005).
    """
    Nz, Ny, Nx = ishape[-3:]

    phi = np.ones(ishape, dtype=complex)

    bas = 2 * np.pi / 2

    if yshift is None:
        yshift = (np.arange(Nz)) / MB
    else:
        assert (len(yshift) == Nz)

    print(' > sms: yshift ', yshift)

    lx = np.arange(Nx) - Nx // 2
    ly = np.arange(Ny) - Ny // 2
    mx, my = np.meshgrid(lx, ly)

    for z in range(Nz):
        slice_yshift = bas * yshift[z]

        phi[..., z, :, :] = np.exp(1j * my * slice_yshift)

    return phi


def readout_extended_fov(ksp, mps, MB):
    """
    References:
        * Koopmans PJ, Poser BA, Breuer FA.
          2D-SENSE-GRAPPA for fast, ghosting-robust reconstruction of
          in-plane and slice accelerated blipped-CAIPI-EPI.
          Proc. Intl. Soc. Magn. Reson. Med. 2015, page 2410.
    """
    Ncoil, Ny, Nx = ksp.shape[-3:]
    _Ncoil, Nz, _Ny, _Nx = mps.shape[-4:]

    assert ((Ncoil == _Ncoil) and (Nz == MB) and (Ny == _Ny) and (Nx == _Nx))

    msk = (sp.rss(ksp, axes=(-3, ), keepdims=True) > 0.).astype(ksp.dtype)

    ksp_ext = np.zeros(list(ksp.shape[:-3]) + [Ncoil] + [Ny] + [Nx * MB],
                       dtype=ksp.dtype)
    msk_ext = np.zeros(list(ksp.shape[:-3]) + [1] + [Ny] + [Nx * MB],
                       dtype=ksp.dtype)
    mps_ext = np.zeros(list(mps.shape[:-3]) + [Ny] + [Nx * MB],
                       dtype=mps.dtype)

    for b in range(MB):  # loop over multi bands

        x_idx = np.arange(b*Nx, (b+1)*Nx, 1)
        mps_ext[..., x_idx] = mps[..., b, :, :]
        ksp_ext[..., x_idx] = sp.ifft(ksp, axes=[-2, -1])

        k_idx = np.arange(0, Nx * MB, MB)
        msk_ext[..., k_idx] = msk
        # ksp_ext[..., k_idx] = ksp

    ksp_ext = msk_ext * sp.fft(ksp_ext, axes=[-2, -1])

    return ksp_ext, mps_ext, msk_ext


def readout_unextend_fov(input, MB):
    """unextend readout extended FOV images.

    Args:
        input (array): input image of shape [..., Ny, Nx * MB].
        MB (int): multi-band factor.

    Output:
        image array of shape [..., MB, Ny, Nx].
    """
    Ny, Nx_ext = input.shape[-2:]

    Nx = int(Nx_ext // MB)

    output = np.zeros(list(input.shape[:-2]) + [MB] + [Ny] + [Nx],
                      dtype=input.dtype)

    for b in range(MB):
        x_idx = np.arange(b*Nx, (b+1)*Nx, 1)
        output[..., b, :, :] = input[..., x_idx]

    return output
