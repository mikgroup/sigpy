import math
import numpy as np
import numba as nb

from sigpy import config, util

if config.cupy_enabled:
    import cupy as cp


__all__ = ['interp', 'gridding']


def interp(input, width, table, coord):
    """Interpolation from array to points specified by coordinates.

    Args:
        input (array): Input array of shape [..., ny, nx]
        width (float): Interpolation kernel width.
        table (array): Interpolation kernel.
        coord (array): Coordinate array of shape [..., ndim]

    Returns:
        output (array): Output array of coord.shape[:-1]
    """

    ndim = coord.shape[-1]

    batch_shape = input.shape[:-ndim]
    batch = util.prod(batch_shape)

    pts_shape = coord.shape[:-1]
    npts = util.prod(pts_shape)

    device = util.get_device(input)
    xp = device.xp
    isreal = np.issubdtype(input.dtype, np.floating)
    coord = util.move(coord, device)
    table = util.move(table, device)

    with device:
        input = input.reshape([batch] + list(input.shape[-ndim:]))
        coord = coord.reshape([npts, ndim])
        output = xp.zeros([batch, npts], dtype=input.dtype)

        _interp = _select_interp(ndim, npts, device, isreal)
        if device == util.cpu_device:
            _interp(output, input, width, table, coord)
        else:
            _interp(output, input, width, table, coord, size=npts)

        return output.reshape(batch_shape + pts_shape)


def gridding(input, shape, width, table, coord):
    """Gridding of points specified by coordinates to array.

    Args:
        input (array): Input array.
        shape (array of ints): Output shape.
        width (float): Interpolation kernel width.
        table (array): Interpolation kernel.
        coord (array): Coordinate array of shape [..., ndim]

    Returns:
        output (array): Output array.
    """

    ndim = coord.shape[-1]

    batch_shape = shape[:-ndim]
    batch = util.prod(batch_shape)

    pts_shape = coord.shape[:-1]
    npts = util.prod(pts_shape)

    device = util.get_device(input)
    xp = device.xp
    isreal = np.issubdtype(input.dtype, np.floating)

    with device:
        input = input.reshape([batch, npts])
        coord = coord.reshape([npts, ndim])
        output = xp.zeros([batch] + list(shape[-ndim:]), dtype=input.dtype)

        _gridding = _select_gridding(ndim, npts, device, isreal)
        if device == -1:
            _gridding(output, input, width, table, coord)
        else:
            _gridding(output, input, width, table, coord, size=npts)

        return output.reshape(shape)


def _select_interp(ndim, npts, device, isreal):

    if ndim == 1:
        if device == util.cpu_device:
            _interp = _interp1
        else:
            _interp = _interp1_cuda
    elif ndim == 2:
        if device == util.cpu_device:
            _interp = _interp2
        else:
            _interp = _interp2_cuda
    elif ndim == 3:
        if device == util.cpu_device:
            _interp = _interp3
        else:
            _interp = _interp3_cuda
    else:
        raise ValueError(
            'Number of dimensions can only be 1, 2 or 3, got {}'.format(ndim))

    return _interp


def _select_gridding(ndim, npts, device, isreal):

    if ndim == 1:
        if device == util.cpu_device:
            _gridding = _gridding1
        else:
            if isreal:
                _gridding = _gridding1_cuda
            else:
                _gridding = _gridding1_cuda_complex
    elif ndim == 2:
        if device == util.cpu_device:
            _gridding = _gridding2
        else:
            if isreal:
                _gridding = _gridding2_cuda
            else:
                _gridding = _gridding2_cuda_complex
    elif ndim == 3:
        if device == util.cpu_device:
            _gridding = _gridding3
        else:
            if isreal:
                _gridding = _gridding3_cuda
            else:
                _gridding = _gridding3_cuda_complex
    else:
        raise ValueError(
            'Number of dimensions can only be 1, 2 or 3, got {}'.format(ndim))

    return _gridding


@nb.jit(nopython=True)
def lin_interp(table, x):
    if x >= 1:
        return 0.0
    n = len(table)
    idx = int(x * n)
    frac = x * n - idx

    left = table[idx]
    if idx == n - 1:
        right = 0.0
    else:
        right = table[idx + 1]
    return (1.0 - frac) * left + frac * right


@nb.jit(nopython=True, cache=True)
def _interp1(output, input, width, table, coord):
    batch, nx = input.shape
    npts = coord.shape[0]

    for i in nb.prange(npts):

        posx = coord[i, -1]

        startx = math.ceil(posx - width / 2)
        endx = math.floor(posx + width / 2)

        for x in range(startx, endx + 1):

            w = lin_interp(table, abs(x - posx) / (width / 2))

            for b in range(batch):
                output[b, i] += w * input[b, x % nx]

    return output


@nb.jit(nopython=True, cache=True)
def _gridding1(output, input, width, table, coord):
    batch, nx = output.shape
    npts = coord.shape[0]

    for i in nb.prange(npts):

        posx = coord[i, -1]

        startx = math.ceil(posx - width / 2)
        endx = math.floor(posx + width / 2)

        for x in range(startx, endx + 1):

            w = lin_interp(table, abs(x - posx) / (width / 2))

            for b in range(batch):
                output[b, x % nx] += w * input[b, i]

    return output


@nb.jit(nopython=True, cache=True)
def _interp2(output, input, width, table, coord):

    batch, ny, nx = input.shape
    npts = coord.shape[0]

    for i in nb.prange(npts):

        posx, posy = coord[i, -1], coord[i, -2]

        startx, starty = (math.ceil(posx - width / 2),
                          math.ceil(posy - width / 2))

        endx, endy = (math.floor(posx + width / 2),
                      math.floor(posy + width / 2))

        for y in range(starty, endy + 1):
            wy = lin_interp(table, abs(y - posy) / (width / 2))

            for x in range(startx, endx + 1):
                w = wy * lin_interp(table, abs(x - posx) / (width / 2))

                for b in range(batch):
                    output[b, i] += w * input[b, y % ny, x % nx]

    return output


@nb.jit(nopython=True, cache=True)
def _gridding2(output, input, width, table, coord):
    batch, ny, nx = output.shape
    npts = coord.shape[0]

    for i in nb.prange(npts):

        posx, posy = coord[i, -1], coord[i, -2]

        startx, starty = (math.ceil(posx - width / 2),
                          math.ceil(posy - width / 2))

        endx, endy = (math.floor(posx + width / 2),
                      math.floor(posy + width / 2))

        for y in range(starty, endy + 1):
            wy = lin_interp(table, abs(y - posy) / (width / 2))

            for x in range(startx, endx + 1):
                w = wy * lin_interp(table, abs(x - posx) / (width / 2))

                for b in range(batch):
                    output[b, y % ny, x % nx] += w * input[b, i]

    return output


@nb.jit(nopython=True, cache=True)
def _interp3(output, input, width, table, coord):
    batch, nz, ny, nx = input.shape
    npts = coord.shape[0]

    for i in nb.prange(npts):

        posx, posy, posz = coord[i, -1], coord[i, -2], coord[i, -3]

        startx, starty, startz = (math.ceil(posx - width / 2),
                                  math.ceil(posy - width / 2),
                                  math.ceil(posz - width / 2))

        endx, endy, endz = (math.floor(posx + width / 2),
                            math.floor(posy + width / 2),
                            math.floor(posz + width / 2))

        for z in range(startz, endz + 1):
            wz = lin_interp(table, abs(z - posz) / (width / 2))

            for y in range(starty, endy + 1):
                wy = wz * lin_interp(table, abs(y - posy) / (width / 2))

                for x in range(startx, endx + 1):
                    w = wy * lin_interp(table, abs(x - posx) / (width / 2))

                    for b in range(batch):
                        output[b, i] += w * input[b, z % nz, y % ny, x % nx]

    return output


@nb.jit(nopython=True, cache=True)
def _gridding3(output, input, width, table, coord):
    batch, nz, ny, nx = output.shape
    npts = coord.shape[0]

    for i in nb.prange(npts):

        posx, posy, posz = coord[i, -1], coord[i, -2], coord[i, -3]

        startx, starty, startz = (math.ceil(posx - width / 2),
                                  math.ceil(posy - width / 2),
                                  math.ceil(posz - width / 2))

        endx, endy, endz = (math.floor(posx + width / 2),
                            math.floor(posy + width / 2),
                            math.floor(posz + width / 2))

        for z in range(startz, endz + 1):
            wz = lin_interp(table, abs(z - posz) / (width / 2))

            for y in range(starty, endy + 1):
                wy = wz * lin_interp(table, abs(y - posy) / (width / 2))

                for x in range(startx, endx + 1):
                    w = wy * lin_interp(table, abs(x - posx) / (width / 2))

                    for b in range(batch):
                        output[b, z % nz, y % ny, x % nx] += w * input[b, i]

    return output


if config.cupy_enabled:

    lin_interp_cuda = """
    __device__ inline S lin_interp(S* table, int n, S x) {
        if (x >= 1)
           return 0;
        const int idx = x * n;
        const S frac = x * n - idx;
        
        const S left = table[idx];
        S right = 0;
        if (idx != n - 1)
            right = table[idx + 1];

        return (1 - frac) * left + frac * right;
    }
    """
    pos_mod_cuda = """
    __device__ inline int pos_mod(int x, int n) {
        return (x % n + n) % n;
    }
    """

    _interp1_cuda = cp.ElementwiseKernel(
        'raw T output, raw T input, raw S width, raw S table, raw S coord',
        '',
        """
        const int batch = input.shape()[0];
        const int nx = input.shape()[1];

        const int coord_idx[] = {i, 0};
        const S posx = coord[coord_idx];
        const int startx = ceil(posx - width / 2.0);
        const int endx = floor(posx + width / 2.0);

        for (int x = startx; x < endx + 1; x++) {
            const S w = lin_interp(&table[0], table.size(), fabsf((S) x - posx) / (width / 2.0));

            for (int b = 0; b < batch; b++) {
                const int input_idx[] = {b, pos_mod(x, nx)};
                const T v = (T) w * input[input_idx];
                const int output_idx[] = {b, i};
                output[output_idx] += v;
            }
        }
        """,
        name='interp1', preamble=lin_interp_cuda + pos_mod_cuda, reduce_dims=False)

    _gridding1_cuda = cp.ElementwiseKernel(
        'raw T output, raw T input, raw S width, raw S table, raw S coord',
        '',
        """
        const int batch = output.shape()[0];
        const int nx = output.shape()[1];

        const int coord_idx[] = {i, 0};
        const S posx = coord[coord_idx];
        const int startx = ceil(posx - width / 2.0);
        const int endx = floor(posx + width / 2.0);

        for (int x = startx; x < endx + 1; x++) {
            const S w = lin_interp(&table[0], table.size(), fabsf((S) x - posx) / (width / 2.0));

            for (int b = 0; b < batch; b++) {
                const int input_idx[] = {b, i};
                const T v = (T) w * input[input_idx];
                const int output_idx[] = {b, pos_mod(x, nx)};
                atomicAdd(&output[output_idx], v);
            }
        }
        """,
        name='gridding1', preamble=lin_interp_cuda + pos_mod_cuda, reduce_dims=False)

    _gridding1_cuda_complex = cp.ElementwiseKernel(
        'raw T output, raw T input, raw S width, raw S table, raw S coord',
        '',
        """
        const int batch = output.shape()[0];
        const int nx = output.shape()[1];

        const int coord_idx[] = {i, 0};
        const S posx = coord[coord_idx];
        const int startx = ceil(posx - width / 2.0);
        const int endx = floor(posx + width / 2.0);

        for (int x = startx; x < endx + 1; x++) {
            const S w = lin_interp(&table[0], table.size(), fabsf((S) x - posx) / (width / 2.0));

            for (int b = 0; b < batch; b++) {
                const int input_idx[] = {b, i};
                const T v = (T) w * input[input_idx];
                const int output_idx[] = {b, pos_mod(x, nx)};
                atomicAdd(reinterpret_cast<T::value_type*>(&(output[output_idx])), v.real());
                atomicAdd(reinterpret_cast<T::value_type*>(&(output[output_idx])) + 1, v.imag());
            }
        }
        """,
        name='gridding1_complex',
        preamble=lin_interp_cuda + pos_mod_cuda,
        reduce_dims=False)

    _interp2_cuda = cp.ElementwiseKernel(
        'raw T output, raw T input, raw S width, raw S table, raw S coord',
        '',
        """
        const int batch = input.shape()[0];
        const int ny = input.shape()[1];
        const int nx = input.shape()[2];

        const int coordx_idx[] = {i, 1};
        const S posx = coord[coordx_idx];
        const int coordy_idx[] = {i, 0};
        const S posy = coord[coordy_idx];

        const int startx = ceil(posx - width / 2.0);
        const int starty = ceil(posy - width / 2.0);

        const int endx = floor(posx + width / 2.0);
        const int endy = floor(posy + width / 2.0);

        for (int y = starty; y < endy + 1; y++) {
            const S wy = lin_interp(&table[0], table.size(), fabsf((S) y - posy) / (width / 2.0));
            for (int x = startx; x < endx + 1; x++) {
                const S w = wy * lin_interp(&table[0], table.size(), fabsf((S) x - posx) / (width / 2.0));

                for (int b = 0; b < batch; b++) {
                    const int input_idx[] = {b, pos_mod(y, ny), pos_mod(x, nx)};
                    const T v = (T) w * input[input_idx];
                    const int output_idx[] = {b, i};
                    output[output_idx] += v;
                }
            }
        }
        """,
        name='interp2', preamble=lin_interp_cuda + pos_mod_cuda, reduce_dims=False)

    _gridding2_cuda = cp.ElementwiseKernel(
        'raw T output, raw T input, raw S width, raw S table, raw S coord',
        '',
        """
        const int batch = output.shape()[0];
        const int ny = output.shape()[1];
        const int nx = output.shape()[2];

        const int coordx_idx[] = {i, 1};
        const S posx = coord[coordx_idx];
        const int coordy_idx[] = {i, 0};
        const S posy = coord[coordy_idx];

        const int startx = ceil(posx - width / 2.0);
        const int starty = ceil(posy - width / 2.0);

        const int endx = floor(posx + width / 2.0);
        const int endy = floor(posy + width / 2.0);

        for (int y = starty; y < endy + 1; y++) {
            const S wy = lin_interp(&table[0], table.size(), fabsf((S) y - posy) / (width / 2.0));
            for (int x = startx; x < endx + 1; x++) {
                const S w = wy * lin_interp(&table[0], table.size(), fabsf((S) x - posx) / (width / 2.0));

                for (int b = 0; b < batch; b++) {
                    const int input_idx[] = {b, i};
                    const T v = (T) w * input[input_idx];
                    const int output_idx[] = {b, pos_mod(y, ny), pos_mod(x, nx)};
                    atomicAdd(&output[output_idx], v);
                }
            }
        }
        """,
        name='gridding2', preamble=lin_interp_cuda + pos_mod_cuda, reduce_dims=False)

    _gridding2_cuda_complex = cp.ElementwiseKernel(
        'raw T output, raw T input, raw S width, raw S table, raw S coord',
        '',
        """
        const int batch = output.shape()[0];
        const int ny = output.shape()[1];
        const int nx = output.shape()[2];

        const int coordx_idx[] = {i, 1};
        const S posx = coord[coordx_idx];
        const int coordy_idx[] = {i, 0};
        const S posy = coord[coordy_idx];

        const int startx = ceil(posx - width / 2.0);
        const int starty = ceil(posy - width / 2.0);

        const int endx = floor(posx + width / 2.0);
        const int endy = floor(posy + width / 2.0);

        for (int y = starty; y < endy + 1; y++) {
            const S wy = lin_interp(&table[0], table.size(), fabsf((S) y - posy) / (width / 2.0));
            for (int x = startx; x < endx + 1; x++) {
                const S w = wy * lin_interp(&table[0], table.size(), fabsf((S) x - posx) / (width / 2.0));

                for (int b = 0; b < batch; b++) {
                    const int input_idx[] = {b, i};
                    const T v = (T) w * input[input_idx];
                    const int output_idx[] = {b, pos_mod(y, ny), pos_mod(x, nx)};
                    atomicAdd(reinterpret_cast<T::value_type*>(&(output[output_idx])), v.real());
                    atomicAdd(reinterpret_cast<T::value_type*>(&(output[output_idx])) + 1, v.imag());
                }
            }
        }
        """,
        name='gridding2_complex',
        preamble=lin_interp_cuda + pos_mod_cuda,
        reduce_dims=False)

    _interp3_cuda = cp.ElementwiseKernel(
        'raw T output, raw T input, raw S width, raw S table, raw S coord',
        '',
        """
        const int batch = input.shape()[0];
        const int nz = input.shape()[1];
        const int ny = input.shape()[2];
        const int nx = input.shape()[3];

        const int coordz_idx[] = {i, 0};
        const S posz = coord[coordz_idx];
        const int coordy_idx[] = {i, 1};
        const S posy = coord[coordy_idx];
        const int coordx_idx[] = {i, 2};
        const S posx = coord[coordx_idx];

        const int startx = ceil(posx - width / 2.0);
        const int starty = ceil(posy - width / 2.0);
        const int startz = ceil(posz - width / 2.0);

        const int endx = floor(posx + width / 2.0);
        const int endy = floor(posy + width / 2.0);
        const int endz = floor(posz + width / 2.0);

        for (int z = startz; z < endz + 1; z++) {
            const S wz = lin_interp(&table[0], table.size(), fabsf((S) z - posz) / (width / 2.0));
            for (int y = starty; y < endy + 1; y++) {
                const S wy = wz * lin_interp(&table[0], table.size(), fabsf((S) y - posy) / (width / 2.0));
                for (int x = startx; x < endx + 1; x++) {
                    const S w = wy * lin_interp(&table[0], table.size(), fabsf((S) x - posx) / (width / 2.0));

                    for (int b = 0; b < batch; b++) {
                        const int input_idx[] = {b, pos_mod(z, nz), pos_mod(y, ny), pos_mod(x, nx)};
                        const T v = (T) w * input[input_idx];
                        const int output_idx[] = {b, i};
                        output[output_idx] += v;
                    }
                }
            }
        }
        """,
        name='interp3', preamble=lin_interp_cuda + pos_mod_cuda, reduce_dims=False)

    _gridding3_cuda = cp.ElementwiseKernel(
        'raw T output, raw T input, raw S width, raw S table, raw S coord',
        '',
        """
        const int batch = output.shape()[0];
        const int nz = output.shape()[1];
        const int ny = output.shape()[2];
        const int nx = output.shape()[3];

        const int coordz_idx[] = {i, 0};
        const S posz = coord[coordz_idx];
        const int coordy_idx[] = {i, 1};
        const S posy = coord[coordy_idx];
        const int coordx_idx[] = {i, 2};
        const S posx = coord[coordx_idx];

        const int startx = ceil(posx - width / 2.0);
        const int starty = ceil(posy - width / 2.0);
        const int startz = ceil(posz - width / 2.0);

        const int endx = floor(posx + width / 2.0);
        const int endy = floor(posy + width / 2.0);
        const int endz = floor(posz + width / 2.0);

        for (int z = startz; z < endz + 1; z++) {
            const S wz = lin_interp(&table[0], table.size(), fabsf((S) z - posz) / (width / 2.0));
            for (int y = starty; y < endy + 1; y++) {
                const S wy = wz * lin_interp(&table[0], table.size(), fabsf((S) y - posy) / (width / 2.0));
                for (int x = startx; x < endx + 1; x++) {
                    const S w = wy * lin_interp(&table[0], table.size(), fabsf((S) x - posx) / (width / 2.0));

                    for (int b = 0; b < batch; b++) {
                        const int input_idx[] = {b, i};
                        const T v = (T) w * input[input_idx];
                        const int output_idx[] = {b, pos_mod(z, nz), pos_mod(y, ny), pos_mod(x, nx)};
                        atomicAdd(&output[output_idx], v);
                    }
                }
            }
        }
        """,
        name='gridding3', preamble=lin_interp_cuda + pos_mod_cuda, reduce_dims=False)

    _gridding3_cuda_complex = cp.ElementwiseKernel(
        'raw T output, raw T input, raw S width, raw S table, raw S coord',
        '',
        """
        const int batch = output.shape()[0];
        const int nz = output.shape()[1];
        const int ny = output.shape()[2];
        const int nx = output.shape()[3];

        const int coordz_idx[] = {i, 0};
        const S posz = coord[coordz_idx];
        const int coordy_idx[] = {i, 1};
        const S posy = coord[coordy_idx];
        const int coordx_idx[] = {i, 2};
        const S posx = coord[coordx_idx];

        const int startx = ceil(posx - width / 2.0);
        const int starty = ceil(posy - width / 2.0);
        const int startz = ceil(posz - width / 2.0);

        const int endx = floor(posx + width / 2.0);
        const int endy = floor(posy + width / 2.0);
        const int endz = floor(posz + width / 2.0);

        for (int z = startz; z < endz + 1; z++) {
            const S wz = lin_interp(&table[0], table.size(), fabsf((S) z - posz) / (width / 2.0));
            for (int y = starty; y < endy + 1; y++) {
                const S wy = wz * lin_interp(&table[0], table.size(), fabsf((S) y - posy) / (width / 2.0));
                for (int x = startx; x < endx + 1; x++) {
                    const S w = wy * lin_interp(&table[0], table.size(), fabsf((S) x - posx) / (width / 2.0));

                    for (int b = 0; b < batch; b++) {
                        const int input_idx[] = {b, i};
                        const T v = (T) w * input[input_idx];
                        const int output_idx[] = {b, pos_mod(z, nz), pos_mod(y, ny), pos_mod(x, nx)};
                        atomicAdd(reinterpret_cast<T::value_type*>(&(output[output_idx])), v.real());
                        atomicAdd(reinterpret_cast<T::value_type*>(&(output[output_idx])) + 1, v.imag());
                    }
                }
            }
        }
        """,
        name='gridding3_complex',
        preamble=lin_interp_cuda + pos_mod_cuda,
        reduce_dims=False)
