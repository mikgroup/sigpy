# -*- coding: utf-8 -*-
"""Block reshape functions.

"""
import numpy as np
import numba as nb

from sigpy import backend, config, util


__all__ = ['array_to_blocks', 'blocks_to_array']


def array_to_blocks(input, blk_shape, blk_strides):
    """Extract blocks from an array in a sliding window manner.

    Args:
        input (array): input array of shape [..., N_1, ..., N_ndim]
        blk_shape (tuple): block shape of length ndim, with ndim <= 4.
        blk_strides (tuple): block strides of length ndim.

    Returns:
        array: array of shape [...] + num_blks + blk_shape, where
            num_blks = (N - blk_shape + blk_strides) // blk_strides.

    Example:

        >>> input = np.array([0, 1, 2, 3, 4, 5])
        >>> print(array_to_blocks(input, [2], [2]))
        [[0, 1],
         [2, 3],
         [4, 5]]

    """
    if len(blk_shape) != len(blk_strides):
        raise ValueError('blk_shape must have the same length as blk_strides.')

    ndim = len(blk_shape)
    num_blks = [(i - b + s) // s for i, b,
                s in zip(input.shape[-ndim:], blk_shape, blk_strides)]
    batch_shape = list(input.shape[:-ndim])
    batch_size = util.prod(batch_shape)
    xp = backend.get_array_module(input)
    output = xp.zeros([batch_size] + num_blks + blk_shape,
                      dtype=input.dtype)
    input = input.reshape([batch_size] + list(input.shape[-ndim:]))

    if ndim == 1:
        if xp == np:
            _array_to_blocks1(output, input,
                              batch_size,
                              blk_shape[-1],
                              blk_strides[-1],
                              num_blks[-1])
        else:  # pragma: no cover
            _array_to_blocks1_cuda(input,
                                   batch_size,
                                   blk_shape[-1],
                                   blk_strides[-1],
                                   num_blks[-1],
                                   output,
                                   size=batch_size *
                                   num_blks[-1] * blk_shape[-1])
    elif ndim == 2:
        if xp == np:
            _array_to_blocks2(output, input,
                              batch_size, blk_shape[-1], blk_shape[-2],
                              blk_strides[-1], blk_strides[-2],
                              num_blks[-1], num_blks[-2])
        else:  # pragma: no cover
            _array_to_blocks2_cuda(input,
                                   batch_size,
                                   blk_shape[-1], blk_shape[-2],
                                   blk_strides[-1], blk_strides[-2],
                                   num_blks[-1], num_blks[-2],
                                   output,
                                   size=batch_size *
                                   num_blks[-1] * num_blks[-2] *
                                   blk_shape[-1] * blk_shape[-2])
    elif ndim == 3:
        if xp == np:
            _array_to_blocks3(output,
                              input,
                              batch_size,
                              blk_shape[-1],
                              blk_shape[-2],
                              blk_shape[-3],
                              blk_strides[-1],
                              blk_strides[-2],
                              blk_strides[-3],
                              num_blks[-1],
                              num_blks[-2],
                              num_blks[-3])
        else:  # pragma: no cover
            _array_to_blocks3_cuda(input,
                                   batch_size,
                                   blk_shape[-1], blk_shape[-2],
                                   blk_shape[-3],
                                   blk_strides[-1], blk_strides[-2],
                                   blk_strides[-3],
                                   num_blks[-1], num_blks[-2],
                                   num_blks[-3],
                                   output,
                                   size=batch_size *
                                   num_blks[-1] * num_blks[-2] *
                                   num_blks[-3] *
                                   blk_shape[-1] * blk_shape[-2] *
                                   blk_shape[-3])
    elif ndim == 4:
        if xp == np:
            _array_to_blocks4(output,
                              input,
                              batch_size,
                              blk_shape[-1],
                              blk_shape[-2],
                              blk_shape[-3],
                              blk_shape[-4],
                              blk_strides[-1],
                              blk_strides[-2],
                              blk_strides[-3],
                              blk_strides[-4],
                              num_blks[-1],
                              num_blks[-2],
                              num_blks[-3],
                              num_blks[-4])
        else:  # pragma: no cover
            _array_to_blocks4_cuda(input,
                                   batch_size,
                                   blk_shape[-1], blk_shape[-2],
                                   blk_shape[-3], blk_shape[-4],
                                   blk_strides[-1], blk_strides[-2],
                                   blk_strides[-3], blk_strides[-4],
                                   num_blks[-1], num_blks[-2],
                                   num_blks[-3], num_blks[-4],
                                   output,
                                   size=batch_size *
                                   num_blks[-1] * num_blks[-2] *
                                   num_blks[-3] * num_blks[-4] *
                                   blk_shape[-1] * blk_shape[-2] *
                                   blk_shape[-3] * blk_shape[-4])
    else:
        raise ValueError('Only support ndim <= 4, got {}'.format(ndim))

    return output.reshape(batch_shape + num_blks + blk_shape)


def blocks_to_array(input, oshape, blk_shape, blk_strides):
    """Accumulate blocks into an array in a sliding window manner.

    Args:
        input (array): input array of shape [...] + num_blks + blk_shape
        oshape (tuple): output shape.
        blk_shape (tuple): block shape of length ndim.
        blk_strides (tuple): block strides of length ndim.

    Returns:
        array: array of shape oshape.

    """
    if len(blk_shape) != len(blk_strides):
        raise ValueError('blk_shape must have the same length as blk_strides.')

    ndim = len(blk_shape)
    num_blks = input.shape[-(2 * ndim):-ndim]
    batch_shape = list(oshape[:-ndim])
    batch_size = util.prod(batch_shape)
    xp = backend.get_array_module(input)
    output = xp.zeros([batch_size] + list(oshape[-ndim:]),
                      dtype=input.dtype)
    input = input.reshape([batch_size] + list(input.shape[-2 * ndim:]))

    if ndim == 1:
        if xp == np:
            _blocks_to_array1(output, input,
                              batch_size, blk_shape[-1],
                              blk_strides[-1],
                              num_blks[-1])
        else:  # pragma: no cover
            if np.issubdtype(input.dtype, np.floating):
                _blocks_to_array1_cuda(input,
                                       batch_size, blk_shape[-1],
                                       blk_strides[-1],
                                       num_blks[-1],
                                       output,
                                       size=batch_size *
                                       num_blks[-1] * blk_shape[-1])
            else:
                _blocks_to_array1_cuda_complex(input,
                                               batch_size, blk_shape[-1],
                                               blk_strides[-1],
                                               num_blks[-1],
                                               output,
                                               size=batch_size
                                               * num_blks[-1] *
                                               blk_shape[-1])
    elif ndim == 2:
        if xp == np:
            _blocks_to_array2(output, input,
                              batch_size, blk_shape[-1], blk_shape[-2],
                              blk_strides[-1], blk_strides[-2],
                              num_blks[-1], num_blks[-2])
        else:  # pragma: no cover
            if np.issubdtype(input.dtype, np.floating):
                _blocks_to_array2_cuda(input,
                                       batch_size,
                                       blk_shape[-1], blk_shape[-2],
                                       blk_strides[-1], blk_strides[-2],
                                       num_blks[-1], num_blks[-2],
                                       output,
                                       size=batch_size *
                                       num_blks[-1] * num_blks[-2] *
                                       blk_shape[-1] * blk_shape[-2])
            else:  # pragma: no cover
                _blocks_to_array2_cuda_complex(
                    input,
                    batch_size, blk_shape[-1], blk_shape[-2],
                    blk_strides[-1], blk_strides[-2],
                    num_blks[-1], num_blks[-2],
                    output,
                    size=batch_size * num_blks[-1] * num_blks[-2] *
                    blk_shape[-1] * blk_shape[-2])
    elif ndim == 3:
        if xp == np:
            _blocks_to_array3(output,
                              input,
                              batch_size, blk_shape[-1],
                              blk_shape[-2],
                              blk_shape[-3],
                              blk_strides[-1],
                              blk_strides[-2],
                              blk_strides[-3],
                              num_blks[-1],
                              num_blks[-2],
                              num_blks[-3])
        else:  # pragma: no cover
            if np.issubdtype(input.dtype, np.floating):
                _blocks_to_array3_cuda(
                    input,
                    batch_size,
                    blk_shape[-1], blk_shape[-2], blk_shape[-3],
                    blk_strides[-1], blk_strides[-2], blk_strides[-3],
                    num_blks[-1], num_blks[-2], num_blks[-3],
                    output,
                    size=batch_size * num_blks[-1] * num_blks[-2] *
                    num_blks[-3] * blk_shape[-1] * blk_shape[-2] *
                    blk_shape[-3])
            else:
                _blocks_to_array3_cuda_complex(
                    input,
                    batch_size,
                    blk_shape[-1], blk_shape[-2], blk_shape[-3],
                    blk_strides[-1], blk_strides[-2], blk_strides[-3],
                    num_blks[-1], num_blks[-2], num_blks[-3],
                    output,
                    size=batch_size *
                    num_blks[-1] * num_blks[-2] * num_blks[-3] *
                    blk_shape[-1] * blk_shape[-2] * blk_shape[-3])
    elif ndim == 4:
        if xp == np:
            _blocks_to_array4(output,
                              input,
                              batch_size, blk_shape[-1],
                              blk_shape[-2],
                              blk_shape[-3],
                              blk_shape[-4],
                              blk_strides[-1],
                              blk_strides[-2],
                              blk_strides[-3],
                              blk_strides[-4],
                              num_blks[-1],
                              num_blks[-2],
                              num_blks[-3],
                              num_blks[-4])
        else:  # pragma: no cover
            if np.issubdtype(input.dtype, np.floating):
                _blocks_to_array4_cuda(
                    input,
                    batch_size, blk_shape[-1], blk_shape[-2],
                    blk_shape[-3], blk_shape[-4],
                    blk_strides[-1], blk_strides[-2],
                    blk_strides[-3], blk_strides[-4],
                    num_blks[-1], num_blks[-2],
                    num_blks[-3], num_blks[-4],
                    output,
                    size=batch_size *
                    num_blks[-1] * num_blks[-2] *
                    num_blks[-3] * num_blks[-4] *
                    blk_shape[-1] * blk_shape[-2] *
                    blk_shape[-3] * blk_shape[-4])
            else:
                _blocks_to_array4_cuda_complex(
                    input,
                    batch_size, blk_shape[-1], blk_shape[-2],
                    blk_shape[-3], blk_shape[-4],
                    blk_strides[-1], blk_strides[-2],
                    blk_strides[-3], blk_strides[-4],
                    num_blks[-1], num_blks[-2],
                    num_blks[-3], num_blks[-4],
                    output,
                    size=batch_size * num_blks[-1] * num_blks[-2] *
                    num_blks[-3] * num_blks[-4] *
                    blk_shape[-1] * blk_shape[-2] *
                    blk_shape[-3] * blk_shape[-4])
    else:
        raise ValueError('Only support ndim <= 4, got {}'.format(ndim))

    return output.reshape(oshape)


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _array_to_blocks1(output, input, batch_size, Bx, Sx, Nx):
    for b in range(batch_size):
        for nx in range(Nx):
            for bx in range(Bx):
                ix = nx * Sx + bx
                if ix < input.shape[-1]:
                    output[b, nx, bx] = input[b, ix]


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _array_to_blocks2(output, input, batch_size, Bx, By, Sx, Sy, Nx, Ny):
    for b in range(batch_size):
        for ny in range(Ny):
            for nx in range(Nx):
                for by in range(By):
                    for bx in range(Bx):
                        iy = ny * Sy + by
                        ix = nx * Sx + bx
                        if ix < input.shape[-1] and iy < input.shape[-2]:
                            output[b, ny, nx, by, bx] = input[b, iy, ix]


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _array_to_blocks3(output, input, batch_size, Bx, By, Bz,
                      Sx, Sy, Sz, Nx, Ny, Nz):
    for b in range(batch_size):
        for nz in range(Nz):
            for ny in range(Ny):
                for nx in range(Nx):
                    for bz in range(Bz):
                        for by in range(By):
                            for bx in range(Bx):
                                iz = nz * Sz + bz
                                iy = ny * Sy + by
                                ix = nx * Sx + bx
                                if (ix < input.shape[-1] and
                                    iy < input.shape[-2] and
                                    iz < input.shape[-3]):
                                    output[b, nz, ny, nx, bz, by,
                                           bx] = input[b, iz, iy, ix]


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _array_to_blocks4(output, input, batch_size, Bx, By, Bz, Bt,
                      Sx, Sy, Sz, St,
                      Nx, Ny, Nz, Nt):
    for b in range(batch_size):
        for nt in range(Nt):
            for nz in range(Nz):
                for ny in range(Ny):
                    for nx in range(Nx):
                        for bt in range(Bt):
                            for bz in range(Bz):
                                for by in range(By):
                                    for bx in range(Bx):
                                        it = nt * St + bt
                                        iz = nz * Sz + bz
                                        iy = ny * Sy + by
                                        ix = nx * Sx + bx
                                        if (ix < input.shape[-1] and
                                            iy < input.shape[-2] and
                                            iz < input.shape[-3] and
                                            it < input.shape[-4]):
                                            output[b, nt, nz, ny, nx,
                                                   bt, bz, by,
                                                   bx] = input[b, it,
                                                               iz, iy, ix]


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _blocks_to_array1(output, input, batch_size, Bx, Sx, Nx):
    for b in range(batch_size):
        for nx in range(Nx):
            for bx in range(Bx):
                ix = nx * Sx + bx
                if ix < output.shape[-1]:
                    output[b, ix] += input[b, nx, bx]


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _blocks_to_array2(output, input, batch_size, Bx, By, Sx, Sy, Nx, Ny):
    for b in range(batch_size):
        for ny in range(Ny):
            for nx in range(Nx):
                for by in range(By):
                    for bx in range(Bx):
                        iy = ny * Sy + by
                        ix = nx * Sx + bx
                        if ix < output.shape[-1] and iy < output.shape[-2]:
                            output[b, iy, ix] += input[b, ny, nx, by, bx]


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _blocks_to_array3(output, input, batch_size, Bx, By, Bz, Sx, Sy, Sz,
                      Nx, Ny, Nz):
    for b in range(batch_size):
        for nz in range(Nz):
            for ny in range(Ny):
                for nx in range(Nx):
                    for bz in range(Bz):
                        for by in range(By):
                            for bx in range(Bx):
                                iz = nz * Sz + bz
                                iy = ny * Sy + by
                                ix = nx * Sx + bx
                                if (ix < output.shape[-1]
                                    and iy < output.shape[-2]
                                    and iz < output.shape[-3]):
                                    output[b, iz, iy, ix] += input[b, nz,
                                                                   ny, nx,
                                                                   bz, by, bx]


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _blocks_to_array4(output, input, batch_size, Bx, By, Bz, Bt,
                      Sx, Sy, Sz, St, Nx, Ny, Nz, Nt):
    for b in range(batch_size):
        for nt in range(Nt):
            for nz in range(Nz):
                for ny in range(Ny):
                    for nx in range(Nx):
                        for bt in range(Bt):
                            for bz in range(Bz):
                                for by in range(By):
                                    for bx in range(Bx):
                                        it = nt * St + bt
                                        iz = nz * Sz + bz
                                        iy = ny * Sy + by
                                        ix = nx * Sx + bx
                                        if (ix < output.shape[-1]
                                            and iy < output.shape[-2]
                                            and iz < output.shape[-3]
                                            and it < output.shape[-4]):
                                            output[b, it,
                                                   iz,
                                                   iy,
                                                   ix] += input[b, nt,
                                                                nz,
                                                                ny,
                                                                nx,
                                                                bt,
                                                                bz,
                                                                by,
                                                                bx]


if config.cupy_enabled:  # pragma: no cover
    import cupy as cp

    _array_to_blocks1_cuda = cp.ElementwiseKernel(
        'raw T input, int32 batch_size, int32 Bx, int32 Sx, int32 Nx',
        'raw T output',
        """
        const int ndim = input.ndim;

        int b = i / Bx / Nx;
        i -= b * Bx * Nx;
        int nx = i / Bx;
        i -= nx * Bx;
        int bx = i;

        int ix = nx * Sx + bx;
        if (ix < input.shape()[ndim - 1]) {
            int input_idx[] = {b, ix};
            int output_idx[] = {b, nx, bx};
            output[output_idx] = input[input_idx];
        }
        """,
        name='_array_to_blocks1_cuda')

    _array_to_blocks2_cuda = cp.ElementwiseKernel(
        'raw T input, int32 batch_size, int32 Bx, int32 By, '
        'int32 Sx, int32 Sy, int32 Nx, int32 Ny',
        'raw T output',
        """
        const int ndim = input.ndim;

        int b = i / Bx / By / Nx / Ny;
        i -= b * Bx * By * Nx * Ny;
        int ny = i / Bx / By / Nx;
        i -= ny * Bx * By * Nx;
        int nx = i / Bx / By;
        i -= nx * Bx * By;
        int by = i / Bx;
        i -= by * Bx;
        int bx = i;

        int iy = ny * Sy + by;
        int ix = nx * Sx + bx;
        if (ix < input.shape()[ndim - 1] && iy < input.shape()[ndim - 2]) {
            int input_idx[] = {b, iy, ix};
            int output_idx[] = {b, ny, nx, by, bx};
            output[output_idx] = input[input_idx];
        }
        """,
        name='_array_to_blocks2_cuda')

    _array_to_blocks3_cuda = cp.ElementwiseKernel(
        'raw T input, int32 batch_size, int32 Bx, int32 By, int32 Bz, '
        'int32 Sx, int32 Sy, int32 Sz, int32 Nx, int32 Ny, int32 Nz',
        'raw T output',
        """
        const int ndim = input.ndim;

        int b = i / Bx / By / Bz / Nx / Ny / Nz;
        i -= b * Bx * By * Bz * Nx * Ny * Nz;
        int nz = i / Bx / By / Bz / Nx / Ny;
        i -= nz * Bx * By * Bz * Nx * Ny;
        int ny = i / Bx / By / Bz / Nx;
        i -= ny * Bx * By * Bz * Nx;
        int nx = i / Bx / By / Bz;
        i -= nx * Bx * By * Bz;
        int bz = i / Bx / By;
        i -= bz * Bx * By;
        int by = i / Bx;
        i -= by * Bx;
        int bx = i;

        int iz = nz * Sz + bz;
        int iy = ny * Sy + by;
        int ix = nx * Sx + bx;
        if (ix < input.shape()[ndim - 1] && iy < input.shape()[ndim - 2]
            && iz < input.shape()[ndim - 3]) {
            int input_idx[] = {b, iz, iy, ix};
            int output_idx[] = {b, nz, ny, nx, bz, by, bx};
            output[output_idx] = input[input_idx];
        }
        """,
        name='_array_to_blocks3_cuda')

    _array_to_blocks4_cuda = cp.ElementwiseKernel(
        'raw T input, int32 batch_size, '
        'int32 Bx, int32 By, int32 Bz, int32 Bt, '
        'int32 Sx, int32 Sy, int32 Sz, int32 St, '
        'int32 Nx, int32 Ny, int32 Nz, int32 Nt',
        'raw T output',
        """
        const int ndim = input.ndim;

        int b = i / Bx / By / Bz / Bt / Nx / Ny / Nz / Nt;
        i -= b * Bx * By * Bz * Bt * Nx * Ny * Nz * Nt;
        int nt = i / Bx / By / Bz / Bt / Nx / Ny / Nz;
        i -= nt * Bx * By * Bz * Bt * Nx * Ny * Nz;
        int nz = i / Bx / By / Bz / Bt / Nx / Ny;
        i -= nz * Bx * By * Bz * Bt * Nx * Ny;
        int ny = i / Bx / By / Bz / Bt / Nx;
        i -= ny * Bx * By * Bz * Bt * Nx;
        int nx = i / Bx / By / Bz / Bt;
        i -= nx * Bx * By * Bz * Bt;
        int bt = i / Bx / By / Bz;
        i -= bt * Bx * By * Bz;
        int bz = i / Bx / By;
        i -= bz * Bx * By;
        int by = i / Bx;
        i -= by * Bx;
        int bx = i;

        int it = nt * St + bt;
        int iz = nz * Sz + bz;
        int iy = ny * Sy + by;
        int ix = nx * Sx + bx;
        if (ix < input.shape()[ndim - 1] && iy < input.shape()[ndim - 2]
            && iz < input.shape()[ndim - 3] && it < input.shape()[ndim - 4]) {
            int input_idx[] = {b, it, iz, iy, ix};
            int output_idx[] = {b, nt, nz, ny, nx, bt, bz, by, bx};
            output[output_idx] = input[input_idx];
        }
        """,
        name='_array_to_blocks4_cuda')

    _blocks_to_array1_cuda = cp.ElementwiseKernel(
        'raw T input, int32 batch_size, int32 Bx, int32 Sx, int32 Nx',
        'raw T output',
        """
        const int ndim = output.ndim;

        int b = i / Bx / Nx;
        i -= b * Bx * Nx;
        int nx = i / Bx;
        i -= nx * Bx;
        int bx = i;

        int ix = nx * Sx + bx;
        if (ix < output.shape()[ndim - 1]) {
            int input_idx[] = {b, nx, bx};
            int output_idx[] = {b, ix};
            atomicAdd(&output[output_idx], input[input_idx]);
        }
        """,
        name='_blocks_to_array1_cuda')

    _blocks_to_array1_cuda_complex = cp.ElementwiseKernel(
        'raw T input, int32 batch_size, int32 Bx, int32 Sx, int32 Nx',
        'raw T output',
        """
        const int ndim = output.ndim;

        int b = i / Bx / Nx;
        i -= b * Bx * Nx;
        int nx = i / Bx;
        i -= nx * Bx;
        int bx = i;

        int ix = nx * Sx + bx;
        if (ix < output.shape()[ndim - 1]) {
            int input_idx[] = {b, nx, bx};
            int output_idx[] = {b, ix};
            atomicAdd(reinterpret_cast<T::value_type*>(&(output[output_idx])),
                input[input_idx].real());
            atomicAdd(
                reinterpret_cast<T::value_type*>(&(output[output_idx])) + 1,
                input[input_idx].imag());
        }
        """,
        name='_blocks_to_array1_cuda_complex')

    _blocks_to_array2_cuda = cp.ElementwiseKernel(
        'raw T input, int32 batch_size, '
        'int32 Bx, int32 By, int32 Sx, int32 Sy, '
        'int32 Nx, int32 Ny',
        'raw T output',
        """
        const int ndim = output.ndim;

        int b = i / Bx / By / Nx / Ny;
        i -= b * Bx * By * Nx * Ny;
        int ny = i / Bx / By / Nx;
        i -= ny * Bx * By * Nx;
        int nx = i / Bx / By;
        i -= nx * Bx * By;
        int by = i / Bx;
        i -= by * Bx;
        int bx = i;

        int iy = ny * Sy + by;
        int ix = nx * Sx + bx;
        if (ix < output.shape()[ndim - 1] && iy < output.shape()[ndim - 2]) {
            int input_idx[] = {b, ny, nx, by, bx};
            int output_idx[] = {b, iy, ix};
            atomicAdd(&output[output_idx], input[input_idx]);
        }
        """,
        name='_blocks_to_array2_cuda')

    _blocks_to_array2_cuda_complex = cp.ElementwiseKernel(
        'raw T input, int32 batch_size, '
        'int32 Bx, int32 By, int32 Sx, int32 Sy, '
        'int32 Nx, int32 Ny',
        'raw T output',
        """
        const int ndim = output.ndim;

        int b = i / Bx / By / Nx / Ny;
        i -= b * Bx * By * Nx * Ny;
        int ny = i / Bx / By / Nx;
        i -= ny * Bx * By * Nx;
        int nx = i / Bx / By;
        i -= nx * Bx * By;
        int by = i / Bx;
        i -= by * Bx;
        int bx = i;

        int iy = ny * Sy + by;
        int ix = nx * Sx + bx;
        if (ix < output.shape()[ndim - 1] && iy < output.shape()[ndim - 2]) {
            int input_idx[] = {b, ny, nx, by, bx};
            int output_idx[] = {b, iy, ix};
            atomicAdd(reinterpret_cast<T::value_type*>(&(output[output_idx])),
                input[input_idx].real());
            atomicAdd(
                reinterpret_cast<T::value_type*>(&(output[output_idx])) + 1,
                input[input_idx].imag());
        }
        """,
        name='_blocks_to_array2_cuda_complex')

    _blocks_to_array3_cuda = cp.ElementwiseKernel(
        'raw T input, int32 batch_size, int32 Bx, int32 By, int32 Bz, '
        'int32 Sx, int32 Sy, int32 Sz, int32 Nx, int32 Ny, int32 Nz',
        'raw T output',
        """
        const int ndim = output.ndim;

        int b = i / Bx / By / Bz / Nx / Ny / Nz;
        i -= b * Bx * By * Bz * Nx * Ny * Nz;
        int nz = i / Bx / By / Bz / Nx / Ny;
        i -= nz * Bx * By * Bz * Nx * Ny;
        int ny = i / Bx / By / Bz / Nx;
        i -= ny * Bx * By * Bz * Nx;
        int nx = i / Bx / By / Bz;
        i -= nx * Bx * By * Bz;
        int bz = i / Bx / By;
        i -= bz * Bx * By;
        int by = i / Bx;
        i -= by * Bx;
        int bx = i;

        int iz = nz * Sz + bz;
        int iy = ny * Sy + by;
        int ix = nx * Sx + bx;
        if (ix < output.shape()[ndim - 1] &&
            iy < output.shape()[ndim - 2] &&
            iz < output.shape()[ndim - 3]) {
            int input_idx[] = {b, nz, ny, nx, bz, by, bx};
            int output_idx[] = {b, iz, iy, ix};
            atomicAdd(&output[output_idx], input[input_idx]);
        }
        """,
        name='_blocks_to_array3_cuda')

    _blocks_to_array3_cuda_complex = cp.ElementwiseKernel(
        'raw T input, int32 batch_size, int32 Bx, int32 By, int32 Bz, '
        'int32 Sx, int32 Sy, int32 Sz, int32 Nx, int32 Ny, int32 Nz',
        'raw T output',
        """
        const int ndim = output.ndim;

        int b = i / Bx / By / Bz / Nx / Ny / Nz;
        i -= b * Bx * By * Bz * Nx * Ny * Nz;
        int nz = i / Bx / By / Bz / Nx / Ny;
        i -= nz * Bx * By * Bz * Nx * Ny;
        int ny = i / Bx / By / Bz / Nx;
        i -= ny * Bx * By * Bz * Nx;
        int nx = i / Bx / By / Bz;
        i -= nx * Bx * By * Bz;
        int bz = i / Bx / By;
        i -= bz * Bx * By;
        int by = i / Bx;
        i -= by * Bx;
        int bx = i;

        int iz = nz * Sz + bz;
        int iy = ny * Sy + by;
        int ix = nx * Sx + bx;
        if (ix < output.shape()[ndim - 1] &&
            iy < output.shape()[ndim - 2] &&
            iz < output.shape()[ndim - 3]) {
            int input_idx[] = {b, nz, ny, nx, bz, by, bx};
            int output_idx[] = {b, iz, iy, ix};
            atomicAdd(reinterpret_cast<T::value_type*>(&(output[output_idx])),
                input[input_idx].real());
            atomicAdd(reinterpret_cast<T::value_type*>(
                &(output[output_idx])) + 1, input[input_idx].imag());
        }
        """,
        name='_blocks_to_array3_cuda_complex')

    _blocks_to_array4_cuda = cp.ElementwiseKernel(
        'raw T input, int32 batch_size, '
        'int32 Bx, int32 By, int32 Bz, int32 Bt, '
        'int32 Sx, int32 Sy, int32 Sz, int32 St, '
        'int32 Nx, int32 Ny, int32 Nz, int32 Nt',
        'raw T output',
        """
        const int ndim = output.ndim;
        int b = i / Bx / By / Bz / Bt / Nx / Ny / Nz / Nt;
        i -= b * Bx * By * Bz * Bt * Nx * Ny * Nz * Nt;
        int nt = i / Bx / By / Bz / Bt / Nx / Ny / Nz;
        i -= nt * Bx * By * Bz * Bt * Nx * Ny * Nz;
        int nz = i / Bx / By / Bz / Bt / Nx / Ny;
        i -= nz * Bx * By * Bz * Bt * Nx * Ny;
        int ny = i / Bx / By / Bz / Bt / Nx;
        i -= ny * Bx * By * Bz * Bt * Nx;
        int nx = i / Bx / By / Bz / Bt;
        i -= nx * Bx * By * Bz * Bt;
        int bt = i / Bx / By / Bz;
        i -= bt * Bx * By * Bz;
        int bz = i / Bx / By;
        i -= bz * Bx * By;
        int by = i / Bx;
        i -= by * Bx;
        int bx = i;

        int it = nt * St + bt;
        int iz = nz * Sz + bz;
        int iy = ny * Sy + by;
        int ix = nx * Sx + bx;

        if (ix < output.shape()[ndim - 1] &&
            iy < output.shape()[ndim - 2] &&
            iz < output.shape()[ndim - 3] &&
            it < output.shape()[ndim - 4]) {
            int input_idx[] = {b, nt, nz, ny, nx, bt, bz, by, bx};
            int output_idx[] = {b, it, iz, iy, ix};
            atomicAdd(&output[output_idx], input[input_idx]);
        }
        """,
        name='_blocks_to_array4_cuda')

    _blocks_to_array4_cuda_complex = cp.ElementwiseKernel(
        'raw T input, int32 batch_size, '
        'int32 Bx, int32 By, int32 Bz, int32 Bt, '
        'int32 Sx, int32 Sy, int32 Sz, int32 St, '
        'int32 Nx, int32 Ny, int32 Nz, int32 Nt',
        'raw T output',
        """
        const int ndim = output.ndim;
        int b = i / Bx / By / Bz / Bt / Nx / Ny / Nz / Nt;
        i -= b * Bx * By * Bz * Bt * Nx * Ny * Nz * Nt;
        int nt = i / Bx / By / Bz / Bt / Nx / Ny / Nz;
        i -= nt * Bx * By * Bz * Bt * Nx * Ny * Nz;
        int nz = i / Bx / By / Bz / Bt / Nx / Ny;
        i -= nz * Bx * By * Bz * Bt * Nx * Ny;
        int ny = i / Bx / By / Bz / Bt / Nx;
        i -= ny * Bx * By * Bz * Bt * Nx;
        int nx = i / Bx / By / Bz / Bt;
        i -= nx * Bx * By * Bz * Bt;
        int bt = i / Bx / By / Bz;
        i -= bt * Bx * By * Bz;
        int bz = i / Bx / By;
        i -= bz * Bx * By;
        int by = i / Bx;
        i -= by * Bx;
        int bx = i;

        int it = nt * St + bt;
        int iz = nz * Sz + bz;
        int iy = ny * Sy + by;
        int ix = nx * Sx + bx;

        if (ix < output.shape()[ndim - 1] &&
            iy < output.shape()[ndim - 2] &&
            iz < output.shape()[ndim - 3] &&
            it < output.shape()[ndim - 4]) {
            int input_idx[] = {b, nt, nz, ny, nx, bt, bz, by, bx};
            int output_idx[] = {b, it, iz, iy, ix};
            atomicAdd(reinterpret_cast<T::value_type*>(&(output[output_idx])),
                input[input_idx].real());
            atomicAdd(reinterpret_cast<T::value_type*>(
                &(output[output_idx])) + 1, input[input_idx].imag());
        }
        """,
        name='_blocks_to_array4_cuda_complex')
