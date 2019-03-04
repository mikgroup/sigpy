# -*- coding: utf-8 -*-
"""Interpolation functions.
"""
import numpy as np
import numba as nb

from sigpy import backend, config, util


__all__ = ['interpolate', 'gridding']


def interpolate(input, width, kernel, coord):
    """Interpolation from array to points specified by coordinates.

    Args:
        input (array): Input array of shape [..., ny, nx]
        width (float): Interpolation kernel width.
        kernel (array): Interpolation kernel.
        coord (array): Coordinate array of shape [..., ndim]

    Returns:
        output (array): Output array of coord.shape[:-1]

    """
    ndim = coord.shape[-1]

    batch_shape = input.shape[:-ndim]
    batch_size = util.prod(batch_shape)

    pts_shape = coord.shape[:-1]
    npts = util.prod(pts_shape)

    device = backend.get_device(input)
    xp = device.xp
    isreal = np.issubdtype(input.dtype, np.floating)
    coord = backend.to_device(coord, device)
    kernel = backend.to_device(kernel, device)

    with device:
        input = input.reshape([batch_size] + list(input.shape[-ndim:]))
        coord = coord.reshape([npts, ndim])
        output = xp.zeros([batch_size, npts], dtype=input.dtype)

        _interpolate = _select_interpolate(ndim, npts, device, isreal)
        if device == backend.cpu_device:
            _interpolate(output, input, width, kernel, coord)
        else:  # pragma: no cover
            _interpolate(input, width, kernel, coord, output, size=npts)

        return output.reshape(batch_shape + pts_shape)


def gridding(input, shape, width, kernel, coord):
    """Gridding of points specified by coordinates to array.

    Args:
        input (array): Input array.
        shape (array of ints): Output shape.
        width (float): Interpolation kernel width.
        kernel (array): Interpolation kernel.
        coord (array): Coordinate array of shape [..., ndim]

    Returns:
        output (array): Output array.

    """
    ndim = coord.shape[-1]

    batch_shape = shape[:-ndim]
    batch_size = util.prod(batch_shape)

    pts_shape = coord.shape[:-1]
    npts = util.prod(pts_shape)

    device = backend.get_device(input)
    xp = device.xp
    isreal = np.issubdtype(input.dtype, np.floating)

    with device:
        input = input.reshape([batch_size, npts])
        coord = coord.reshape([npts, ndim])
        output = xp.zeros(
            [batch_size] + list(shape[-ndim:]), dtype=input.dtype)

        _gridding = _select_gridding(ndim, npts, device, isreal)
        if device == backend.cpu_device:
            _gridding(output, input, width, kernel, coord)
        else:  # pragma: no cover
            _gridding(input, width, kernel, coord, output, size=npts)

        return output.reshape(shape)


def _select_interpolate(ndim, npts, device, isreal):
    if ndim == 1:
        if device == backend.cpu_device:
            _interpolate = _interpolate1
        else:  # pragma: no cover
            _interpolate = _interpolate1_cuda
    elif ndim == 2:
        if device == backend.cpu_device:
            _interpolate = _interpolate2
        else:  # pragma: no cover
            _interpolate = _interpolate2_cuda
    elif ndim == 3:
        if device == backend.cpu_device:
            _interpolate = _interpolate3
        else:  # pragma: no cover
            _interpolate = _interpolate3_cuda
    else:
        raise ValueError(
            'Number of dimensions can only be 1, 2 or 3, got {}'.format(ndim))

    return _interpolate


def _select_gridding(ndim, npts, device, isreal):
    if ndim == 1:
        if device == backend.cpu_device:
            _gridding = _gridding1
        else:  # pragma: no cover
            if isreal:
                _gridding = _gridding1_cuda
            else:
                _gridding = _gridding1_cuda_complex
    elif ndim == 2:
        if device == backend.cpu_device:
            _gridding = _gridding2
        else:  # pragma: no cover
            if isreal:
                _gridding = _gridding2_cuda
            else:
                _gridding = _gridding2_cuda_complex
    elif ndim == 3:
        if device == backend.cpu_device:
            _gridding = _gridding3
        else:  # pragma: no cover
            if isreal:
                _gridding = _gridding3_cuda
            else:
                _gridding = _gridding3_cuda_complex
    else:
        raise ValueError(
            'Number of dimensions can only be 1, 2 or 3, got {}'.format(ndim))

    return _gridding


@nb.jit(nopython=True)  # pragma: no cover
def lin_interpolate(kernel, x):
    if x >= 1:
        return 0.0
    n = len(kernel)
    idx = int(x * n)
    frac = x * n - idx

    left = kernel[idx]
    if idx == n - 1:
        right = 0.0
    else:
        right = kernel[idx + 1]
    return (1.0 - frac) * left + frac * right


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _interpolate1(output, input, width, kernel, coord):
    batch_size, nx = input.shape
    npts = coord.shape[0]

    for i in range(npts):

        kx = coord[i, -1]

        x0 = np.ceil(kx - width / 2)
        x1 = np.floor(kx + width / 2)

        for x in range(x0, x1 + 1):

            w = lin_interpolate(kernel, abs(x - kx) / (width / 2))

            for b in range(batch_size):
                output[b, i] += w * input[b, x % nx]

    return output


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _gridding1(output, input, width, kernel, coord):
    batch_size, nx = output.shape
    npts = coord.shape[0]

    for i in range(npts):

        kx = coord[i, -1]

        x0 = np.ceil(kx - width / 2)
        x1 = np.floor(kx + width / 2)

        for x in range(x0, x1 + 1):

            w = lin_interpolate(kernel, abs(x - kx) / (width / 2))

            for b in range(batch_size):
                output[b, x % nx] += w * input[b, i]

    return output


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _interpolate2(output, input, width, kernel, coord):

    batch_size, ny, nx = input.shape
    npts = coord.shape[0]

    for i in range(npts):

        kx, ky = coord[i, -1], coord[i, -2]

        x0, y0 = (np.ceil(kx - width / 2),
                  np.ceil(ky - width / 2))

        x1, y1 = (np.floor(kx + width / 2),
                  np.floor(ky + width / 2))

        for y in range(y0, y1 + 1):
            wy = lin_interpolate(kernel, abs(y - ky) / (width / 2))

            for x in range(x0, x1 + 1):
                w = wy * lin_interpolate(kernel, abs(x - kx) / (width / 2))

                for b in range(batch_size):
                    output[b, i] += w * input[b, y % ny, x % nx]

    return output


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _gridding2(output, input, width, kernel, coord):
    batch_size, ny, nx = output.shape
    npts = coord.shape[0]

    for i in range(npts):

        kx, ky = coord[i, -1], coord[i, -2]

        x0, y0 = (np.ceil(kx - width / 2),
                  np.ceil(ky - width / 2))

        x1, y1 = (np.floor(kx + width / 2),
                  np.floor(ky + width / 2))

        for y in range(y0, y1 + 1):
            wy = lin_interpolate(kernel, abs(y - ky) / (width / 2))

            for x in range(x0, x1 + 1):
                w = wy * lin_interpolate(kernel, abs(x - kx) / (width / 2))

                for b in range(batch_size):
                    output[b, y % ny, x % nx] += w * input[b, i]

    return output


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _interpolate3(output, input, width, kernel, coord):
    batch_size, nz, ny, nx = input.shape
    npts = coord.shape[0]

    for i in range(npts):

        kx, ky, kz = coord[i, -1], coord[i, -2], coord[i, -3]

        x0, y0, z0 = (np.ceil(kx - width / 2),
                      np.ceil(ky - width / 2),
                      np.ceil(kz - width / 2))

        x1, y1, z1 = (np.floor(kx + width / 2),
                      np.floor(ky + width / 2),
                      np.floor(kz + width / 2))

        for z in range(z0, z1 + 1):
            wz = lin_interpolate(kernel, abs(z - kz) / (width / 2))

            for y in range(y0, y1 + 1):
                wy = wz * lin_interpolate(kernel, abs(y - ky) / (width / 2))

                for x in range(x0, x1 + 1):
                    w = wy * lin_interpolate(kernel, abs(x - kx) / (width / 2))

                    for b in range(batch_size):
                        output[b, i] += w * input[b, z % nz, y % ny, x % nx]

    return output


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _gridding3(output, input, width, kernel, coord):
    batch_size, nz, ny, nx = output.shape
    npts = coord.shape[0]

    for i in range(npts):

        kx, ky, kz = coord[i, -1], coord[i, -2], coord[i, -3]

        x0, y0, z0 = (np.ceil(kx - width / 2),
                      np.ceil(ky - width / 2),
                      np.ceil(kz - width / 2))

        x1, y1, z1 = (np.floor(kx + width / 2),
                      np.floor(ky + width / 2),
                      np.floor(kz + width / 2))

        for z in range(z0, z1 + 1):
            wz = lin_interpolate(kernel, abs(z - kz) / (width / 2))

            for y in range(y0, y1 + 1):
                wy = wz * lin_interpolate(kernel, abs(y - ky) / (width / 2))

                for x in range(x0, x1 + 1):
                    w = wy * lin_interpolate(kernel, abs(x - kx) / (width / 2))

                    for b in range(batch_size):
                        output[b, z % nz, y % ny, x % nx] += w * input[b, i]

    return output


if config.cupy_enabled:  # pragma: no cover
    import cupy as cp

    lin_interpolate_cuda = """
    __device__ inline S lin_interpolate(const S* kernel, int n, S x) {
        if (x >= 1)
           return 0;
        const int idx = x * n;
        const S frac = x * n - idx;

        const S left = kernel[idx];
        S right = 0;
        if (idx != n - 1)
            right = kernel[idx + 1];

        return (1 - frac) * left + frac * right;
    }
    """
    mod_cuda = """
    __device__ inline int mod(int x, int n) {
        return (x % n + n) % n;
    }
    """

    _interpolate1_cuda = cp.ElementwiseKernel(
        'raw T input, raw S width, raw S kernel, raw S coord',
        'raw T output',
        """
        const int batch_size = input.shape()[0];
        const int nx = input.shape()[1];

        const int coord_idx[] = {i, 0};
        const S kx = coord[coord_idx];
        const int x0 = ceil(kx - width / 2.0);
        const int x1 = floor(kx + width / 2.0);

        for (int x = x0; x < x1 + 1; x++) {
            const S w = lin_interpolate(&kernel[0], kernel.size(),
                fabsf((S) x - kx) / (width / 2.0));
            for (int b = 0; b < batch_size; b++) {
                const int input_idx[] = {b, mod(x, nx)};
                const T v = (T) w * input[input_idx];
                const int output_idx[] = {b, i};
                output[output_idx] += v;
            }
        }
        """,
        name='interpolate1',
        preamble=lin_interpolate_cuda + mod_cuda,
        reduce_dims=False)

    _gridding1_cuda = cp.ElementwiseKernel(
        'raw T input, raw S width, raw S kernel, raw S coord',
        'raw T output',
        """
        const int batch_size = output.shape()[0];
        const int nx = output.shape()[1];

        const int coord_idx[] = {i, 0};
        const S kx = coord[coord_idx];
        const int x0 = ceil(kx - width / 2.0);
        const int x1 = floor(kx + width / 2.0);

        for (int x = x0; x < x1 + 1; x++) {
            const S w = lin_interpolate(&kernel[0], kernel.size(),
                fabsf((S) x - kx) / (width / 2.0));
            for (int b = 0; b < batch_size; b++) {
                const int input_idx[] = {b, i};
                const T v = (T) w * input[input_idx];
                const int output_idx[] = {b, mod(x, nx)};
                atomicAdd(&output[output_idx], v);
            }
        }
        """,
        name='gridding1',
        preamble=lin_interpolate_cuda + mod_cuda,
        reduce_dims=False)

    _gridding1_cuda_complex = cp.ElementwiseKernel(
        'raw T input, raw S width, raw S kernel, raw S coord',
        'raw T output',
        """
        const int batch_size = output.shape()[0];
        const int nx = output.shape()[1];

        const int coord_idx[] = {i, 0};
        const S kx = coord[coord_idx];
        const int x0 = ceil(kx - width / 2.0);
        const int x1 = floor(kx + width / 2.0);

        for (int x = x0; x < x1 + 1; x++) {
            const S w = lin_interpolate(&kernel[0], kernel.size(),
                fabsf((S) x - kx) / (width / 2.0));
            for (int b = 0; b < batch_size; b++) {
                const int input_idx[] = {b, i};
                const T v = (T) w * input[input_idx];
                const int output_idx[] = {b, mod(x, nx)};
                atomicAdd(
                    reinterpret_cast<T::value_type*>(
                        &(output[output_idx])), v.real());
                atomicAdd(
                    reinterpret_cast<T::value_type*>(
                        &(output[output_idx])) + 1, v.imag());
            }
        }
        """,
        name='gridding1_complex',
        preamble=lin_interpolate_cuda + mod_cuda,
        reduce_dims=False)

    _interpolate2_cuda = cp.ElementwiseKernel(
        'raw T input, raw S width, raw S kernel, raw S coord',
        'raw T output',
        """
        const int batch_size = input.shape()[0];
        const int ny = input.shape()[1];
        const int nx = input.shape()[2];

        const int coordx_idx[] = {i, 1};
        const S kx = coord[coordx_idx];
        const int coordy_idx[] = {i, 0};
        const S ky = coord[coordy_idx];

        const int x0 = ceil(kx - width / 2.0);
        const int y0 = ceil(ky - width / 2.0);

        const int x1 = floor(kx + width / 2.0);
        const int y1 = floor(ky + width / 2.0);

        for (int y = y0; y < y1 + 1; y++) {
            const S wy = lin_interpolate(&kernel[0], kernel.size(),
                fabsf((S) y - ky) / (width / 2.0));
            for (int x = x0; x < x1 + 1; x++) {
                const S w = wy * lin_interpolate(&kernel[0], kernel.size(),
                    fabsf((S) x - kx) / (width / 2.0));
                for (int b = 0; b < batch_size; b++) {
                    const int input_idx[] = {b, mod(y, ny), mod(x, nx)};
                    const T v = (T) w * input[input_idx];
                    const int output_idx[] = {b, i};
                    output[output_idx] += v;
                }
            }
        }
        """,
        name='interpolate2',
        preamble=lin_interpolate_cuda + mod_cuda,
        reduce_dims=False)

    _gridding2_cuda = cp.ElementwiseKernel(
        'raw T input, raw S width, raw S kernel, raw S coord', 'raw T output', """
        const int batch_size = output.shape()[0];
        const int ny = output.shape()[1];
        const int nx = output.shape()[2];

        const int coordx_idx[] = {i, 1};
        const S kx = coord[coordx_idx];
        const int coordy_idx[] = {i, 0};
        const S ky = coord[coordy_idx];

        const int x0 = ceil(kx - width / 2.0);
        const int y0 = ceil(ky - width / 2.0);

        const int x1 = floor(kx + width / 2.0);
        const int y1 = floor(ky + width / 2.0);

        for (int y = y0; y < y1 + 1; y++) {
            const S wy = lin_interpolate(&kernel[0], kernel.size(),
                fabsf((S) y - ky) / (width / 2.0));
            for (int x = x0; x < x1 + 1; x++) {
                const S w = wy * lin_interpolate(&kernel[0], kernel.size(),
                    fabsf((S) x - kx) / (width / 2.0));
                for (int b = 0; b < batch_size; b++) {
                    const int input_idx[] = {b, i};
                    const T v = (T) w * input[input_idx];
                    const int output_idx[] = {b, mod(y, ny), mod(x, nx)};
                    atomicAdd(&output[output_idx], v);
                }
            }
        }
        """, name='gridding2', preamble=lin_interpolate_cuda + mod_cuda,
        reduce_dims=False)

    _gridding2_cuda_complex = cp.ElementwiseKernel(
        'raw T input, raw S width, raw S kernel, raw S coord',
        'raw T output',
        """
        const int batch_size = output.shape()[0];
        const int ny = output.shape()[1];
        const int nx = output.shape()[2];

        const int coordx_idx[] = {i, 1};
        const S kx = coord[coordx_idx];
        const int coordy_idx[] = {i, 0};
        const S ky = coord[coordy_idx];

        const int x0 = ceil(kx - width / 2.0);
        const int y0 = ceil(ky - width / 2.0);

        const int x1 = floor(kx + width / 2.0);
        const int y1 = floor(ky + width / 2.0);

        for (int y = y0; y < y1 + 1; y++) {
            const S wy = lin_interpolate(&kernel[0], kernel.size(),
                fabsf((S) y - ky) / (width / 2.0));
            for (int x = x0; x < x1 + 1; x++) {
                const S w = wy * lin_interpolate(&kernel[0], kernel.size(),
                    fabsf((S) x - kx) / (width / 2.0));
                for (int b = 0; b < batch_size; b++) {
                    const int input_idx[] = {b, i};
                    const T v = (T) w * input[input_idx];
                    const int output_idx[] = {b, mod(y, ny), mod(x, nx)};
                    atomicAdd(reinterpret_cast<T::value_type*>(
                        &(output[output_idx])), v.real());
                    atomicAdd(reinterpret_cast<T::value_type*>(
                        &(output[output_idx])) + 1, v.imag());
                }
            }
        }
        """,
        name='gridding2_complex',
        preamble=lin_interpolate_cuda + mod_cuda,
        reduce_dims=False)

    _interpolate3_cuda = cp.ElementwiseKernel(
        'raw T input, raw S width, raw S kernel, raw S coord', 'raw T output', """
        const int batch_size = input.shape()[0];
        const int nz = input.shape()[1];
        const int ny = input.shape()[2];
        const int nx = input.shape()[3];

        const int coordz_idx[] = {i, 0};
        const S kz = coord[coordz_idx];
        const int coordy_idx[] = {i, 1};
        const S ky = coord[coordy_idx];
        const int coordx_idx[] = {i, 2};
        const S kx = coord[coordx_idx];

        const int x0 = ceil(kx - width / 2.0);
        const int y0 = ceil(ky - width / 2.0);
        const int z0 = ceil(kz - width / 2.0);

        const int x1 = floor(kx + width / 2.0);
        const int y1 = floor(ky + width / 2.0);
        const int z1 = floor(kz + width / 2.0);

        for (int z = z0; z < z1 + 1; z++) {
            const S wz = lin_interpolate(&kernel[0], kernel.size(),
                fabsf((S) z - kz) / (width / 2.0));
            for (int y = y0; y < y1 + 1; y++) {
                const S wy = wz * lin_interpolate(&kernel[0], kernel.size(),
                    fabsf((S) y - ky) / (width / 2.0));
                for (int x = x0; x < x1 + 1; x++) {
                    const S w = wy * lin_interpolate(&kernel[0], kernel.size(),
                        fabsf((S) x - kx) / (width / 2.0));
                    for (int b = 0; b < batch_size; b++) {
                        const int input_idx[] = {b, mod(z, nz), mod(y, ny),
                            mod(x, nx)};
                        const T v = (T) w * input[input_idx];
                        const int output_idx[] = {b, i};
                        output[output_idx] += v;
                    }
                }
            }
        }
        """, name='interpolate3', preamble=lin_interpolate_cuda + mod_cuda,
        reduce_dims=False)

    _gridding3_cuda = cp.ElementwiseKernel(
        'raw T input, raw S width, raw S kernel, raw S coord', 'raw T output', """
        const int batch_size = output.shape()[0];
        const int nz = output.shape()[1];
        const int ny = output.shape()[2];
        const int nx = output.shape()[3];

        const int coordz_idx[] = {i, 0};
        const S kz = coord[coordz_idx];
        const int coordy_idx[] = {i, 1};
        const S ky = coord[coordy_idx];
        const int coordx_idx[] = {i, 2};
        const S kx = coord[coordx_idx];

        const int x0 = ceil(kx - width / 2.0);
        const int y0 = ceil(ky - width / 2.0);
        const int z0 = ceil(kz - width / 2.0);

        const int x1 = floor(kx + width / 2.0);
        const int y1 = floor(ky + width / 2.0);
        const int z1 = floor(kz + width / 2.0);

        for (int z = z0; z < z1 + 1; z++) {
            const S wz = lin_interpolate(&kernel[0], kernel.size(),
                fabsf((S) z - kz) / (width / 2.0));
            for (int y = y0; y < y1 + 1; y++) {
                const S wy = wz * lin_interpolate(&kernel[0], kernel.size(),
                    fabsf((S) y - ky) / (width / 2.0));
                for (int x = x0; x < x1 + 1; x++) {
                    const S w = wy * lin_interpolate(&kernel[0], kernel.size(),
                        fabsf((S) x - kx) / (width / 2.0));
                    for (int b = 0; b < batch_size; b++) {
                        const int input_idx[] = {b, i};
                        const T v = (T) w * input[input_idx];
                        const int output_idx[] = {b, mod(z, nz), mod(y, ny),
                            mod(x, nx)};
                        atomicAdd(&output[output_idx], v);
                    }
                }
            }
        }
        """, name='gridding3', preamble=lin_interpolate_cuda + mod_cuda,
        reduce_dims=False)

    _gridding3_cuda_complex = cp.ElementwiseKernel(
        'raw T input, raw S width, raw S kernel, raw S coord',
        'raw T output',
        """
        const int batch_size = output.shape()[0];
        const int nz = output.shape()[1];
        const int ny = output.shape()[2];
        const int nx = output.shape()[3];

        const int coordz_idx[] = {i, 0};
        const S kz = coord[coordz_idx];
        const int coordy_idx[] = {i, 1};
        const S ky = coord[coordy_idx];
        const int coordx_idx[] = {i, 2};
        const S kx = coord[coordx_idx];

        const int x0 = ceil(kx - width / 2.0);
        const int y0 = ceil(ky - width / 2.0);
        const int z0 = ceil(kz - width / 2.0);

        const int x1 = floor(kx + width / 2.0);
        const int y1 = floor(ky + width / 2.0);
        const int z1 = floor(kz + width / 2.0);

        for (int z = z0; z < z1 + 1; z++) {
            const S wz = lin_interpolate(&kernel[0], kernel.size(),
                fabsf((S) z - kz) / (width / 2.0));
            for (int y = y0; y < y1 + 1; y++) {
                const S wy = wz * lin_interpolate(&kernel[0], kernel.size(),
                     fabsf((S) y - ky) / (width / 2.0));
                for (int x = x0; x < x1 + 1; x++) {
                    const S w = wy * lin_interpolate(&kernel[0], kernel.size(),
                        fabsf((S) x - kx) / (width / 2.0));
                    for (int b = 0; b < batch_size; b++) {
                        const int input_idx[] = {b, i};
                        const T v = (T) w * input[input_idx];
                        const int output_idx[] = {b, mod(z, nz), mod(y, ny),
                            mod(x, nx)};
                        atomicAdd(reinterpret_cast<T::value_type*>(
                            &(output[output_idx])), v.real());
                        atomicAdd(reinterpret_cast<T::value_type*>(
                            &(output[output_idx])) + 1, v.imag());
                    }
                }
            }
        }
        """,
        name='gridding3_complex',
        preamble=lin_interpolate_cuda + mod_cuda,
        reduce_dims=False)
