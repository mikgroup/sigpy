# -*- coding: utf-8 -*-
"""MRI linear operators.

This module mainly contains the Sense linear operator,
which integrates multi-channel coil sensitivity maps and
discrete Fourier transform.

"""
import sigpy as sp


def Sense(mps, coord=None, weights=None, tseg=None, ishape=None,
          coil_batch_size=None, comm=None, transp_nufft=False):
    """Sense linear operator.

    Args:
        mps (array): sensitivity maps of length = number of channels.
        coord (None or array): coordinates.
        weights (None or array): k-space weights.
            Useful for soft-gating or density compensation.
        tseg (None or Dictionary): parameters for time-segmented off-resonance
            correction. Parameters are 'b0' (array), 'dt' (float),
            'lseg' (int), and 'n_bins' (int). Lseg is the number of
            time segments used, and n_bins is the number of histogram bins.
        ishape (None or tuple): image shape.
        coil_batch_size (None or int): batch size for processing multi-channel.
            When None, process all coils at the same time.
            Useful for saving memory.
        comm (None or `sigpy.Communicator`): communicator
            for distributed computing.

    """
    # Get image shape and dimension.
    num_coils = len(mps)
    if ishape is None:
        ishape = mps.shape[1:]
        img_ndim = mps.ndim - 1
    else:
        img_ndim = len(ishape)

    # Serialize linop if coil_batch_size is smaller than num_coils.
    num_coils = len(mps)
    if coil_batch_size is None:
        coil_batch_size = num_coils

    if coil_batch_size < len(mps):
        num_coil_batches = (num_coils + coil_batch_size - 1) // coil_batch_size
        A = sp.linop.Vstack([Sense(mps[c::num_coil_batches], coord=coord,
                                   weights=weights, ishape=ishape)
                             for c in range(num_coil_batches)], axis=0)

        if comm is not None:
            C = sp.linop.AllReduceAdjoint(ishape, comm, in_place=True)
            A = A * C

        return A

    # Create Sense linear operator
    S = sp.linop.Multiply(ishape, mps)
    if tseg is None:
        if coord is None:
            F = sp.linop.FFT(S.oshape, axes=range(-img_ndim, 0))
        else:
            if transp_nufft is False:
                F = sp.linop.NUFFT(S.oshape, coord)
            else:
                F = sp.linop.NUFFT(S.oshape, -coord).H

        A = F * S

    # If B0 provided, perform time-segmented off-resonance compensation
    else:
        if transp_nufft is False:
            F = sp.linop.NUFFT(S.oshape, coord)
        else:
            F = sp.linop.NUFFT(S.oshape, -coord).H
        time = len(coord) * tseg['dt']
        b, ct = sp.mri.util.tseg_off_res_b_ct(tseg['b0'], tseg['n_bins'],
                                              tseg['lseg'], tseg['dt'], time)
        for ii in range(tseg['lseg']):
            Bi = sp.linop.Multiply(F.oshape, b[:, ii])
            Cti = sp.linop.Multiply(S.ishape, ct[:, ii].reshape(S.ishape))

            # operation below is effectively A = A + Bi * F(Cti * S)
            if ii == 0:
                A = Bi * F * S * Cti
            else:
                A = A + Bi * F * S * Cti

    if weights is not None:
        with sp.get_device(weights):
            P = sp.linop.Multiply(F.oshape, weights**0.5)

        A = P * A

    if comm is not None:
        C = sp.linop.AllReduceAdjoint(ishape, comm, in_place=True)
        A = A * C

    A.repr_str = 'Sense'
    return A


def ConvSense(img_ker_shape, mps_ker, coord=None, weights=None, grd_shape=None,
              comm=None):
    """Convolution linear operator with sensitivity maps kernel in k-space.

    Args:
        img_ker_shape (tuple of ints): image kernel shape.
        mps_ker (array): sensitivity maps kernel.
        coord (array): coordinates.
        grd_shape (None or list): Shape of grid.

    """
    ndim = len(img_ker_shape)
    num_coils = mps_ker.shape[0]
    mps_ker = mps_ker.reshape((num_coils, 1) + mps_ker.shape[1:])
    R = sp.linop.Reshape((1, ) + tuple(img_ker_shape), img_ker_shape)
    C = sp.linop.ConvolveData(R.oshape, mps_ker,
                              mode='valid', multi_channel=True)
    A = C * R

    if coord is not None:
        if grd_shape is None:
            grd_shape = sp.estimate_shape(coord)
        else:
            grd_shape = list(grd_shape)

        grd_shape = [num_coils] + grd_shape
        iF = sp.linop.IFFT(grd_shape, axes=range(-ndim, 0))
        N = sp.linop.NUFFT(grd_shape, coord)
        A = N * iF * A

    if weights is not None:
        with sp.get_device(weights):
            P = sp.linop.Multiply(A.oshape, weights**0.5)

        A = P * A

    if comm is not None:
        C = sp.linop.AllReduceAdjoint(img_ker_shape, comm, in_place=True)
        A = A * C

    return A


def ConvImage(mps_ker_shape, img_ker, coord=None, weights=None,
              grd_shape=None):
    """Convolution linear operator with image kernel in k-space.

    Args:
        mps_ker_shape (tuple of ints): sensitivity maps kernel shape.
        img_ker (array): image kernel.
        coord (array): coordinates.
        grd_shape (None or list): Shape of grid.

    """
    ndim = img_ker.ndim
    num_coils = mps_ker_shape[0]
    img_ker = img_ker.reshape((1, ) + img_ker.shape)
    R = sp.linop.Reshape((num_coils, 1) + tuple(mps_ker_shape[1:]),
                         mps_ker_shape)
    C = sp.linop.ConvolveFilter(R.oshape, img_ker,
                                mode='valid', multi_channel=True)
    A = C * R

    if coord is not None:
        num_coils = mps_ker_shape[0]
        if grd_shape is None:
            grd_shape = sp.estimate_shape(coord)
        else:
            grd_shape = list(grd_shape)

        grd_shape = [num_coils] + grd_shape
        iF = sp.linop.IFFT(grd_shape, axes=range(-ndim, 0))
        N = sp.linop.NUFFT(grd_shape, coord)
        A = N * iF * A

    if weights is not None:
        with sp.get_device(weights):
            P = sp.linop.Multiply(A.oshape, weights**0.5)
        A = P * A

    return A
