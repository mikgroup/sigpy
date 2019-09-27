# -*- coding: utf-8 -*-
"""MRI linear operators.

This module mainly contains the Sense linear operator,
which integrates multi-channel coil sensitivity maps and
discrete Fourier transform.

"""
import sigpy as sp


def Sense(mps, coord=None, weights=None, ishape=None,
          coil_batch_size=None, comm=None):
    """Sense linear operator.

    Args:
        mps (array): sensitivity maps of length = number of channels.
        coord (None or array): coordinates.
        weights (None or array): k-space weights.
        Useful for soft-gating or density compensation.
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
    if coord is None:
        F = sp.linop.FFT(S.oshape, axes=range(-img_ndim, 0))
    else:
        F = sp.linop.NUFFT(S.oshape, coord)

    A = F * S

    if weights is not None:
        with sp.get_device(weights):
            P = sp.linop.Multiply(F.oshape, weights**0.5)

        A = P * A

    if comm is not None:
        C = sp.linop.AllReduceAdjoint(ishape, comm, in_place=True)
        A = A * C

    A.repr_str = 'Sense'
    return A


def ConvSense(img_ker_shape, mps_ker, coord=None, weights=None, comm=None):
    """Convolution linear operator with sensitivity maps kernel in k-space.

    Args:
        img_ker_shape (tuple of ints): image kernel shape.
        mps_ker (array): sensitivity maps kernel.
        coord (array): coordinates.

    """
    ndim = len(img_ker_shape)
    num_coils = mps_ker.shape[0]
    mps_ker = mps_ker.reshape((num_coils, 1) + mps_ker.shape[1:])
    R = sp.linop.Reshape((1, ) + tuple(img_ker_shape), img_ker_shape)
    C = sp.linop.ConvolveData(R.oshape, mps_ker,
                              mode='valid', multi_channel=True)
    A = C * R

    if coord is not None:
        grd_shape = [num_coils] + sp.estimate_shape(coord)
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


def ConvImage(mps_ker_shape, img_ker, coord=None, weights=None):
    """Convolution linear operator with image kernel in k-space.

    Args:
        mps_ker_shape (tuple of ints): sensitivity maps kernel shape.
        img_ker (array): image kernel.
        coord (array): coordinates.

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
        grd_shape = [num_coils] + sp.estimate_shape(coord)
        iF = sp.linop.IFFT(grd_shape, axes=range(-ndim, 0))
        N = sp.linop.NUFFT(grd_shape, coord)
        A = N * iF * A

    if weights is not None:
        with sp.get_device(weights):
            P = sp.linop.Multiply(A.oshape, weights**0.5)
        A = P * A

    return A
