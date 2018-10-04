import numpy as np
import numba as nb

from sigpy import backend, config, util


__all__ = ['array_to_block', 'blocks_to_array']


def array_to_blocks(input, blk_shape, blk_strides):
    """
    output - num_blks + blk_shape
    """
    ndim = input.ndim

    if ndim != len(blk_shape) or ndim != len(blk_strides):
        raise ValueError('Input must have same dimensions as blocks.')

    num_blks = [(i - b + s) // s for i, b, s in zip(input.shape, blk_shape, blk_strides)]
    device = backend.get_device(input)
    xp = device.xp
    with device:
        output = xp.zeros(num_blks + blk_shape, dtype=input.dtype)
    
    if ndim == 1:
        if device == backend.cpu_device:
            _array_to_blocks1(output, input, blk_shape[-1], blk_strides[-1], num_blks[-1])
        else:
            _array_to_blocks1_cuda(output, input, blk_shape[-1], blk_strides[-1], num_blks[-1],
                                   size=num_blks[-1] * blk_shape[-1])

    return output


def blocks_to_array(input, shape, blk_shape, blk_strides):
    """
    output - num_blks + blk_shape
    """
    ndim = len(blk_shape)

    if 2 * ndim != input.ndim or ndim != len(blk_strides):
        raise ValueError('Input must have same dimensions as blocks.')
    
    num_blks = input.shape[:ndim]
    device = backend.get_device(input)
    xp = device.xp
    with device:
        output = xp.zeros(shape, dtype=input.dtype)

    if ndim == 1:
        if device == backend.cpu_device:
            _blocks_to_array1(output, input, blk_shape[-1], blk_strides[-1], num_blks[-1])
        else:
            _blocks_to_array1_cuda(output, input, blk_shape[-1], blk_strides[-1], num_blks[-1],
                                   size=num_blks[-1] * blk_shape[-1])

    return output


@nb.jit(nopython=True, cache=True)
def _array_to_blocks1(output, input, Bx, Sx, Nx):
    ndim = input.ndim
    
    for nx in range(Nx):
        for bx in range(Bx):
            ix = nx * Sx + bx
            if ix < input.shape[-1]:
                output[nx, bx] = input[ix]


@nb.jit(nopython=True, cache=True)
def _blocks_to_array1(output, input, Bx, Sx, Nx):
    ndim = output.ndim
    
    for nx in range(Nx):
        for bx in range(Bx):
            ix = nx * Sx + bx
            if ix < output.shape[-1]:
                output[ix] += input[nx, bx]


if config.cupy_enabled:
    import cupy as cp

    _array_to_blocks1_cuda = cp.ElementwiseKernel(
        'raw T output, raw T input, int32 Bx, int32 Sx, int32 Nx',
        '',
        """
        const int ndim = input.ndim;

        int nx = i / Bx;
        int bx = i % Bx;

        int ix = nx * Sx + bx;
        if (ix < input.shape()[ndim - 1]) {
            int input_idx[] = {ix};
            int output_idx[] = {nx, bx};
            output[output_idx] = input[input_idx];
        }
        """,
        name='_array_to_blocks1_cuda')

    _blocks_to_array1_cuda = cp.ElementwiseKernel(
        'raw T output, raw T input, int32 Bx, int32 Sx, int32 Nx',
        '',
        """
        const int ndim = output.ndim;

        int nx = i / Bx;
        int bx = i % Bx;

        int ix = nx * Sx + bx;
        if (ix < output.shape()[ndim - 1]) {
            int input_idx[] = {nx, bx};
            int output_idx[] = {ix};
            atomicAdd(&output[output_idx], input[input_idx]);
        }
        """,
        name='_blocks_to_array1_cuda')
