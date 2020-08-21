# -*- coding: utf-8 -*-
"""Interpolation functions.
"""
import numpy as np
import numba as nb

from sigpy import backend, config, util


__all__ = ['interpolate', 'gridding']


KERNELS = ['spline', 'kaiser_bessel']


def interpolate(input, coord, kernel='spline', width=2, param=1):
    r"""Interpolation from array to points specified by coordinates.

    Let :math:`x` be the input, :math:`y` be the output,
    :math:`c` be the coordinates, :math:`W` be the kernel width,
    and :math:`K` be the interpolation kernel, then the function computes,

    .. math ::
        y[j] = \sum_{i : \| i - c[j] \|_\infty \leq W / 2}
               K\left(\frac{i - c[j]}{W / 2}\right) x[i]

    There are two types of kernels: 'spline' and 'kaiser_bessel'.

    'spline' uses the cardinal B-spline functions as kernels.
    The order of the spline can be specified using param.
    For example, param=1 performs linear interpolation.
    Concretely, for param=0, :math:`K(x) = 1`,
    for param=1, :math:`K(x) = 1 - |x|`, and
    for param=2, :math:`K(x) = \frac{9}{8} (1 - |x|)^2`
    for :math:`|x| > \frac{1}{3}`
    and :math:`K(x) = \frac{3}{4} (1 - 3 x^2)` for :math:`|x| < \frac{1}{3}`.

    These function expressions are derived from the reference wikipedia
    page by shifting and scaling the range to -1 to 1.
    When the coordinates specifies a uniformly spaced grid,
    it is recommended to use the original scaling with width=param + 1
    so that the interpolation weights add up to one.

    'kaiser_bessel' uses the Kaiser-Bessel function as kernel.
    Concretely, :math:`K(x) = I_0(\beta \sqrt{1 - x^2})`,
    where :math:`I_0` is the modified Bessel function of the first kind.
    The beta parameter can be specified with param.
    The modified Bessel function of the first kind is approximated
    using the power series, following the reference.

    Args:
        input (array): Input array of shape.
        coord (array): Coordinate array of shape [..., ndim]
        width (float or tuple of floats): Interpolation kernel full-width.
        kernel (str): Interpolation kernel, {'spline', 'kaiser_bessel'}.
        param (float or tuple of floats): Kernel parameter.

    Returns:
        output (array): Output array.

    References:
        https://en.wikipedia.org/wiki/Spline_wavelet#Cardinal_B-splines_of_small_orders
        http://people.math.sfu.ca/~cbm/aands/page_378.htm
    """
    ndim = coord.shape[-1]

    batch_shape = input.shape[:-ndim]
    batch_size = util.prod(batch_shape)

    pts_shape = coord.shape[:-1]
    npts = util.prod(pts_shape)

    xp = backend.get_array_module(input)

    input = input.reshape([batch_size] + list(input.shape[-ndim:]))
    coord = coord.reshape([npts, ndim])
    output = xp.zeros([batch_size, npts], dtype=input.dtype)

    if np.isscalar(param):
        param = xp.array([param] * ndim, coord.dtype)
    else:
        param = xp.array(param, coord.dtype)

    if np.isscalar(width):
        width = xp.array([width] * ndim, coord.dtype)
    else:
        width = xp.array(width, coord.dtype)

    if xp == np:
        _interpolate[kernel][ndim - 1](output, input, coord, width, param)
    else:  # pragma: no cover
        _interpolate_cuda[kernel][ndim - 1](
            input, coord, width, param, output, size=npts)

    return output.reshape(batch_shape + pts_shape)


def gridding(input, coord, shape, kernel="spline", width=2, param=1):
    r"""Gridding of points specified by coordinates to array.

    Let :math:`y` be the input, :math:`x` be the output,
    :math:`c` be the coordinates, :math:`W` be the kernel width,
    and :math:`K` be the interpolation kernel, then the function computes,

    .. math ::
        x[i] = \sum_{j : \| i - c[j] \|_\infty \leq W / 2}
               K\left(\frac{i - c[j]}{W / 2}\right) y[j]

    There are two types of kernels: 'spline' and 'kaiser_bessel'.

    'spline' uses the cardinal B-spline functions as kernels.
    The order of the spline can be specified using param.
    For example, param=1 performs linear interpolation.
    Concretely, for param=0, :math:`K(x) = 1`,
    for param=1, :math:`K(x) = 1 - |x|`, and
    for param=2, :math:`K(x) = \frac{9}{8} (1 - |x|)^2`
    for :math:`|x| > \frac{1}{3}`
    and :math:`K(x) = \frac{3}{4} (1 - 3 x^2)` for :math:`|x| < \frac{1}{3}`.

    These function expressions are derived from the reference wikipedia
    page by shifting and scaling the range to -1 to 1.
    When the coordinates specifies a uniformly spaced grid,
    it is recommended to use the original scaling with width=param + 1
    so that the interpolation weights add up to one.

    'kaiser_bessel' uses the Kaiser-Bessel function as kernel.
    Concretely, :math:`K(x) = I_0(\beta \sqrt{1 - x^2})`,
    where :math:`I_0` is the modified Bessel function of the first kind.
    The beta parameter can be specified with param.
    The modified Bessel function of the first kind is approximated
    using the power series, following the reference.

    Args:
        input (array): Input array.
        coord (array): Coordinate array of shape [..., ndim]
        width (float or tuple of floats): Interpolation kernel full-width.
        kernel (str): Interpolation kernel, {"spline", "kaiser_bessel"}.
        param (float or tuple of floats): Kernel parameter.

    Returns:
        output (array): Output array.

    References:
        https://en.wikipedia.org/wiki/Spline_wavelet#Cardinal_B-splines_of_small_orders
        http://people.math.sfu.ca/~cbm/aands/page_378.htm
    """
    ndim = coord.shape[-1]

    batch_shape = shape[:-ndim]
    batch_size = util.prod(batch_shape)

    pts_shape = coord.shape[:-1]
    npts = util.prod(pts_shape)

    xp = backend.get_array_module(input)
    isreal = np.issubdtype(input.dtype, np.floating)

    input = input.reshape([batch_size, npts])
    coord = coord.reshape([npts, ndim])
    output = xp.zeros([batch_size] + list(shape[-ndim:]), dtype=input.dtype)

    if np.isscalar(param):
        param = xp.array([param] * ndim, coord.dtype)
    else:
        param = xp.array(param, coord.dtype)

    if np.isscalar(width):
        width = xp.array([width] * ndim, coord.dtype)
    else:
        width = xp.array(width, coord.dtype)

    if xp == np:
        _gridding[kernel][ndim - 1](output, input, coord, width, param)
    else:  # pragma: no cover
        if isreal:
            _gridding_cuda[kernel][ndim - 1](
                input, coord, width, param, output, size=npts)
        else:
            _gridding_cuda_complex[kernel][ndim - 1](
                input, coord, width, param, output, size=npts)

    return output.reshape(shape)


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _spline_kernel(x, order):
    if abs(x) > 1:
        return 0

    if order == 0:
        return 1
    elif order == 1:
        return 1 - abs(x)
    elif order == 2:
        if abs(x) > 1 / 3:
            return 9 / 8 * (1 - abs(x))**2
        else:
            return 3 / 4 * (1 - 3 * x**2)


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _kaiser_bessel_kernel(x, beta):
    if abs(x) > 1:
        return 0

    x = beta * (1 - x**2)**0.5
    t = x / 3.75
    if x < 3.75:
        return 1 + 3.5156229 * t**2 + 3.0899424 * t**4 +\
            1.2067492 * t**6 + 0.2659732 * t**8 +\
            0.0360768 * t**10 + 0.0045813 * t**12
    else:
        return x**-0.5 * np.exp(x) * (
            0.39894228 + 0.01328592 * t**-1 +
            0.00225319 * t**-2 - 0.00157565 * t**-3 +
            0.00916281 * t**-4 - 0.02057706 * t**-5 +
            0.02635537 * t**-6 - 0.01647633 * t**-7 +
            0.00392377 * t**-8)


def _get_interpolate(kernel):
    if kernel == 'spline':
        kernel = _spline_kernel
    elif kernel == 'kaiser_bessel':
        kernel = _kaiser_bessel_kernel

    @nb.jit(nopython=True)  # pragma: no cover
    def _interpolate1(output, input, coord, width, param):
        batch_size, nx = input.shape
        npts = coord.shape[0]

        for i in range(npts):
            kx = coord[i, -1]

            x0 = np.ceil(kx - width[-1] / 2)
            x1 = np.floor(kx + width[-1] / 2)

            for x in range(x0, x1 + 1):

                w = kernel((x - kx) / (width[-1] / 2), param[-1])

                for b in range(batch_size):
                    output[b, i] += w * input[b, x % nx]

        return output

    @nb.jit(nopython=True)  # pragma: no cover
    def _interpolate2(output, input, coord, width, param):

        batch_size, ny, nx = input.shape
        npts = coord.shape[0]

        for i in range(npts):
            kx, ky = coord[i, -1], coord[i, -2]

            x0, y0 = (np.ceil(kx - width[-1] / 2),
                      np.ceil(ky - width[-2] / 2))

            x1, y1 = (np.floor(kx + width[-1] / 2),
                      np.floor(ky + width[-2] / 2))

            for y in range(y0, y1 + 1):
                wy = kernel((y - ky) / (width[-2] / 2), param[-2])

                for x in range(x0, x1 + 1):
                    w = wy * kernel((x - kx) / (width[-1] / 2), param[-1])

                    for b in range(batch_size):
                        output[b, i] += w * input[b, y % ny, x % nx]

        return output

    @nb.jit(nopython=True)  # pragma: no cover
    def _interpolate3(output, input, coord, width, param):
        batch_size, nz, ny, nx = input.shape
        npts = coord.shape[0]

        for i in range(npts):
            kx, ky, kz = coord[i, -1], coord[i, -2], coord[i, -3]

            x0, y0, z0 = (np.ceil(kx - width[-1] / 2),
                          np.ceil(ky - width[-2] / 2),
                          np.ceil(kz - width[-3] / 2))

            x1, y1, z1 = (np.floor(kx + width[-1] / 2),
                          np.floor(ky + width[-2] / 2),
                          np.floor(kz + width[-3] / 2))

            for z in range(z0, z1 + 1):
                wz = kernel((z - kz) / (width[-3] / 2), param[-3])

                for y in range(y0, y1 + 1):
                    wy = wz * kernel((y - ky) / (width[-2] / 2), param[-2])

                    for x in range(x0, x1 + 1):
                        w = wy * kernel((x - kx) / (width[-1] / 2), param[-1])

                        for b in range(batch_size):
                            output[b, i] += w * input[
                                b, z % nz, y % ny, x % nx]

        return output

    return _interpolate1, _interpolate2, _interpolate3


def _get_gridding(kernel):
    if kernel == 'spline':
        kernel = _spline_kernel
    elif kernel == 'kaiser_bessel':
        kernel = _kaiser_bessel_kernel

    @nb.jit(nopython=True)  # pragma: no cover
    def _gridding1(output, input, coord, width, param):
        batch_size, nx = output.shape
        npts = coord.shape[0]

        for i in range(npts):
            kx = coord[i, -1]

            x0 = np.ceil(kx - width[-1] / 2)
            x1 = np.floor(kx + width[-1] / 2)
            for x in range(x0, x1 + 1):
                w = kernel((x - kx) / (width[-1] / 2), param[-1])

                for b in range(batch_size):
                    output[b, x % nx] += w * input[b, i]

        return output

    @nb.jit(nopython=True)  # pragma: no cover
    def _gridding2(output, input, coord, width, param):
        batch_size, ny, nx = output.shape
        npts = coord.shape[0]

        for i in range(npts):
            kx, ky = coord[i, -1], coord[i, -2]

            x0, y0 = (np.ceil(kx - width[-1] / 2),
                      np.ceil(ky - width[-2] / 2))

            x1, y1 = (np.floor(kx + width[-1] / 2),
                      np.floor(ky + width[-2] / 2))
            for y in range(y0, y1 + 1):
                wy = kernel((y - ky) / (width[-2] / 2), param[-2])
                for x in range(x0, x1 + 1):
                    w = wy * kernel((x - kx) / (width[-1] / 2), param[-1])

                    for b in range(batch_size):
                        output[b, y % ny, x % nx] += w * input[b, i]

        return output

    @nb.jit(nopython=True)  # pragma: no cover
    def _gridding3(output, input, coord, width, param):
        batch_size, nz, ny, nx = output.shape
        npts = coord.shape[0]

        for i in range(npts):

            kx, ky, kz = coord[i, -1], coord[i, -2], coord[i, -3]

            x0, y0, z0 = (np.ceil(kx - width[-1] / 2),
                          np.ceil(ky - width[-2] / 2),
                          np.ceil(kz - width[-3] / 2))

            x1, y1, z1 = (np.floor(kx + width[-1] / 2),
                          np.floor(ky + width[-2] / 2),
                          np.floor(kz + width[-3] / 2))

            for z in range(z0, z1 + 1):
                wz = kernel((z - kz) / (width[-3] / 2), param[-3])

                for y in range(y0, y1 + 1):
                    wy = wz * kernel((y - ky) / (width[-2] / 2), param[-2])

                    for x in range(x0, x1 + 1):
                        w = wy * kernel(
                            (x - kx) / (width[-1] / 2), param[-1])

                        for b in range(batch_size):
                            output[b, z % nz, y % ny, x % nx] += w * input[
                                b, i]

        return output

    return _gridding1, _gridding2, _gridding3


_interpolate = {}
_gridding = {}
for kernel in KERNELS:
    _interpolate[kernel] = _get_interpolate(kernel)
    _gridding[kernel] = _get_gridding(kernel)

if config.cupy_enabled:  # pragma: no cover
    import cupy as cp

    _spline_kernel_cuda = """
    __device__ inline S kernel(S x, S order) {
        if (fabsf(x) > 1)
            return 0;

        if (order == 0)
            return 1;
        else if (order == 1)
            return 1 - fabsf(x);
        else if (fabsf(x) > 1 / 3)
            return 9 / 8 * (1 - fabsf(x)) * (1 - fabsf(x));
        else
            return 3 / 4 * (1 - 3 * x * x);
    }
    """

    _kaiser_bessel_kernel_cuda = """
    __device__ inline S kernel(S x, S beta) {
        if (fabsf(x) > 1)
            return 0;

        x = beta * sqrt(1 - x * x);
        S t = x / 3.75;
        S t2 = t * t;
        S t4 = t2 * t2;
        S t6 = t4 * t2;
        S t8 = t6 * t2;
        if (x < 3.75) {
            S t10 = t8 * t2;
            S t12 = t10 * t2;
            return 1 + 3.5156229 * t2 + 3.0899424 * t4 +
                1.2067492 * t6 + 0.2659732 * t8 +
                0.0360768 * t10 + 0.0045813 * t12;
        } else {
            S t3 = t * t2;
            S t5 = t3 * t2;
            S t7 = t5 * t2;

            return exp(x) / sqrt(x) * (
                0.39894228 + 0.01328592 / t +
                0.00225319 / t2 - 0.00157565 / t3 +
                0.00916281 / t4 - 0.02057706 / t5 +
                0.02635537 / t6 - 0.01647633 / t7 +
                0.00392377 / t8);
        }
    }
    """

    mod_cuda = """
    __device__ inline int mod(int x, int n) {
        return (x % n + n) % n;
    }
    """

    def _get_interpolate_cuda(kernel):
        if kernel == 'spline':
            kernel = _spline_kernel_cuda
        elif kernel == 'kaiser_bessel':
            kernel = _kaiser_bessel_kernel_cuda

        _interpolate1_cuda = cp.ElementwiseKernel(
            'raw T input, raw S coord, raw S width, raw S param',
            'raw T output',
            """
            const int ndim = 1;
            const int batch_size = input.shape()[0];
            const int nx = input.shape()[1];

            const int coord_idx[] = {i, 0};
            const S kx = coord[coord_idx];
            const int x0 = ceil(kx - width[ndim - 1] / 2.0);
            const int x1 = floor(kx + width[ndim - 1] / 2.0);

            for (int x = x0; x < x1 + 1; x++) {
                const S w = kernel(
                    ((S) x - kx) / (width[ndim - 1] / 2.0), param[ndim - 1]);
                for (int b = 0; b < batch_size; b++) {
                    const int input_idx[] = {b, mod(x, nx)};
                    const T v = (T) w * input[input_idx];
                    const int output_idx[] = {b, i};
                    output[output_idx] += v;
                }
            }
            """,
            name='interpolate1',
            preamble=kernel + mod_cuda,
            reduce_dims=False)

        _interpolate2_cuda = cp.ElementwiseKernel(
            'raw T input, raw S coord, raw S width, raw S param',
            'raw T output',
            """
            const int ndim = 2;
            const int batch_size = input.shape()[0];
            const int ny = input.shape()[1];
            const int nx = input.shape()[2];

            const int coordx_idx[] = {i, 1};
            const S kx = coord[coordx_idx];
            const int coordy_idx[] = {i, 0};
            const S ky = coord[coordy_idx];

            const int x0 = ceil(kx - width[ndim - 1] / 2.0);
            const int y0 = ceil(ky - width[ndim - 2] / 2.0);

            const int x1 = floor(kx + width[ndim - 1] / 2.0);
            const int y1 = floor(ky + width[ndim - 2] / 2.0);

            for (int y = y0; y < y1 + 1; y++) {
                const S wy = kernel(
                    ((S) y - ky) / (width[ndim - 2] / 2.0),
                    param[ndim - 2]);
                for (int x = x0; x < x1 + 1; x++) {
                    const S w = wy * kernel(
                        ((S) x - kx) / (width[ndim - 1] / 2.0),
                        param[ndim - 1]);
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
            preamble=kernel + mod_cuda,
            reduce_dims=False)

        _interpolate3_cuda = cp.ElementwiseKernel(
            'raw T input, raw S coord, raw S width, raw S param', 'raw T output', """
            const int ndim = 3;
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

            const int x0 = ceil(kx - width[ndim - 1] / 2.0);
            const int y0 = ceil(ky - width[ndim - 2] / 2.0);
            const int z0 = ceil(kz - width[ndim - 3] / 2.0);

            const int x1 = floor(kx + width[ndim - 1] / 2.0);
            const int y1 = floor(ky + width[ndim - 2] / 2.0);
            const int z1 = floor(kz + width[ndim - 3] / 2.0);

            for (int z = z0; z < z1 + 1; z++) {
                const S wz = kernel(
                    ((S) z - kz) / (width[ndim - 3] / 2.0),
                    param[ndim - 3]);
                for (int y = y0; y < y1 + 1; y++) {
                    const S wy = wz * kernel(
                        ((S) y - ky) / (width[ndim - 2] / 2.0),
                        param[ndim - 2]);
                    for (int x = x0; x < x1 + 1; x++) {
                        const S w = wy * kernel(
                            ((S) x - kx) / (width[ndim - 1] / 2.0),
                            param[ndim - 1]);
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
            """, name='interpolate3', preamble=kernel + mod_cuda,
            reduce_dims=False)

        return _interpolate1_cuda, _interpolate2_cuda, _interpolate3_cuda

    def _get_gridding_cuda(kernel):
        if kernel == 'spline':
            kernel = _spline_kernel_cuda
        elif kernel == 'kaiser_bessel':
            kernel = _kaiser_bessel_kernel_cuda

        _gridding1_cuda = cp.ElementwiseKernel(
            'raw T input, raw S coord, raw S width, raw S param',
            'raw T output',
            """
            const int ndim = 1;
            const int batch_size = output.shape()[0];
            const int nx = output.shape()[1];

            const int coord_idx[] = {i, 0};
            const S kx = coord[coord_idx];
            const int x0 = ceil(kx - width[ndim - 1] / 2.0);
            const int x1 = floor(kx + width[ndim - 1] / 2.0);

            for (int x = x0; x < x1 + 1; x++) {
                const S w = kernel(
                    ((S) x - kx) / (width[ndim - 1] / 2.0), param[ndim - 1]);
                for (int b = 0; b < batch_size; b++) {
                    const int input_idx[] = {b, i};
                    const T v = (T) w * input[input_idx];
                    const int output_idx[] = {b, mod(x, nx)};
                    atomicAdd(&output[output_idx], v);
                }
            }
            """,
            name='gridding1',
            preamble=kernel + mod_cuda,
            reduce_dims=False)

        _gridding2_cuda = cp.ElementwiseKernel(
            'raw T input, raw S coord, raw S width, raw S param', 'raw T output', """
            const int ndim = 2;
            const int batch_size = output.shape()[0];
            const int ny = output.shape()[1];
            const int nx = output.shape()[2];

            const int coordx_idx[] = {i, 1};
            const S kx = coord[coordx_idx];
            const int coordy_idx[] = {i, 0};
            const S ky = coord[coordy_idx];

            const int x0 = ceil(kx - width[ndim - 1] / 2.0);
            const int y0 = ceil(ky - width[ndim - 2] / 2.0);

            const int x1 = floor(kx + width[ndim - 1] / 2.0);
            const int y1 = floor(ky + width[ndim - 2] / 2.0);

            for (int y = y0; y < y1 + 1; y++) {
                const S wy = kernel(
                    ((S) y - ky) / (width[ndim - 2] / 2.0),
                    param[ndim - 2]);
                for (int x = x0; x < x1 + 1; x++) {
                    const S w = wy * kernel(
                        ((S) x - kx) / (width[ndim - 1] / 2.0),
                        param[ndim - 1]);
                    for (int b = 0; b < batch_size; b++) {
                        const int input_idx[] = {b, i};
                        const T v = (T) w * input[input_idx];
                        const int output_idx[] = {b, mod(y, ny), mod(x, nx)};
                        atomicAdd(&output[output_idx], v);
                    }
                }
            }
            """, name='gridding2', preamble=kernel + mod_cuda,
            reduce_dims=False)

        _gridding3_cuda = cp.ElementwiseKernel(
            'raw T input, raw S coord, raw S width, raw S param', 'raw T output', """
            const int ndim = 3;
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

            const int x0 = ceil(kx - width[ndim - 1] / 2.0);
            const int y0 = ceil(ky - width[ndim - 2] / 2.0);
            const int z0 = ceil(kz - width[ndim - 3] / 2.0);

            const int x1 = floor(kx + width[ndim - 1] / 2.0);
            const int y1 = floor(ky + width[ndim - 2] / 2.0);
            const int z1 = floor(kz + width[ndim - 3] / 2.0);

            for (int z = z0; z < z1 + 1; z++) {
                const S wz = kernel(
                    ((S) z - kz) / (width[ndim - 3] / 2.0),
                    param[ndim - 3]);
                for (int y = y0; y < y1 + 1; y++) {
                    const S wy = wz * kernel(
                        ((S) y - ky) / (width[ndim - 2] / 2.0),
                        param[ndim - 2]);
                    for (int x = x0; x < x1 + 1; x++) {
                        const S w = wy * kernel(
                            ((S) x - kx) / (width[ndim - 1] / 2.0),
                            param[ndim - 1]);
                        for (int b = 0; b < batch_size; b++) {
                            const int input_idx[] = {b, i};
                            const T v = (T) w * input[input_idx];
                            const int output_idx[] = {
                                b, mod(z, nz), mod(y, ny), mod(x, nx)};
                            atomicAdd(&output[output_idx], v);
                        }
                    }
                }
            }
            """, name='gridding3', preamble=kernel + mod_cuda,
            reduce_dims=False)

        return _gridding1_cuda, _gridding2_cuda, _gridding3_cuda

    def _get_gridding_cuda_complex(kernel):
        if kernel == 'spline':
            kernel = _spline_kernel_cuda
        elif kernel == 'kaiser_bessel':
            kernel = _kaiser_bessel_kernel_cuda

        _gridding1_cuda_complex = cp.ElementwiseKernel(
            'raw T input, raw S coord, raw S width, raw S param',
            'raw T output',
            """
            const int ndim = 1;
            const int batch_size = output.shape()[0];
            const int nx = output.shape()[1];

            const int coord_idx[] = {i, 0};
            const S kx = coord[coord_idx];
            const int x0 = ceil(kx - width[ndim - 1] / 2.0);
            const int x1 = floor(kx + width[ndim - 1] / 2.0);

            for (int x = x0; x < x1 + 1; x++) {
                const S w = kernel(
                    ((S) x - kx) / (width[ndim - 1] / 2.0), param[ndim - 1]);
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
            preamble=kernel + mod_cuda,
            reduce_dims=False)
        _gridding2_cuda_complex = cp.ElementwiseKernel(
            'raw T input, raw S coord, raw S width, raw S param',
            'raw T output',
            """
            const int ndim = 2;
            const int batch_size = output.shape()[0];
            const int ny = output.shape()[1];
            const int nx = output.shape()[2];

            const int coordx_idx[] = {i, 1};
            const S kx = coord[coordx_idx];
            const int coordy_idx[] = {i, 0};
            const S ky = coord[coordy_idx];

            const int x0 = ceil(kx - width[ndim - 1] / 2.0);
            const int y0 = ceil(ky - width[ndim - 2] / 2.0);

            const int x1 = floor(kx + width[ndim - 1] / 2.0);
            const int y1 = floor(ky + width[ndim - 2] / 2.0);

            for (int y = y0; y < y1 + 1; y++) {
                const S wy = kernel(
                    ((S) y - ky) / (width[ndim - 2] / 2.0),
                    param[ndim - 2]);
                for (int x = x0; x < x1 + 1; x++) {
                    const S w = wy * kernel(
                        ((S) x - kx) / (width[ndim - 1] / 2.0),
                        param[ndim - 1]);
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
            preamble=kernel + mod_cuda,
            reduce_dims=False)

        _gridding3_cuda_complex = cp.ElementwiseKernel(
            'raw T input, raw S coord, raw S width, raw S param',
            'raw T output',
            """
            const int ndim = 3;
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

            const int x0 = ceil(kx - width[ndim - 1] / 2.0);
            const int y0 = ceil(ky - width[ndim - 2] / 2.0);
            const int z0 = ceil(kz - width[ndim - 3] / 2.0);

            const int x1 = floor(kx + width[ndim - 1] / 2.0);
            const int y1 = floor(ky + width[ndim - 2] / 2.0);
            const int z1 = floor(kz + width[ndim - 3] / 2.0);

            for (int z = z0; z < z1 + 1; z++) {
                const S wz = kernel(
                    ((S) z - kz) / (width[ndim - 3] / 2.0),
                    param[ndim - 3]);
                for (int y = y0; y < y1 + 1; y++) {
                    const S wy = wz * kernel(
                         ((S) y - ky) / (width[ndim - 2] / 2.0),
                         param[ndim - 2]);
                    for (int x = x0; x < x1 + 1; x++) {
                        const S w = wy * kernel(
                            ((S) x - kx) / (width[ndim - 1] / 2.0),
                            param[ndim - 1]);
                        for (int b = 0; b < batch_size; b++) {
                            const int input_idx[] = {b, i};
                            const T v = (T) w * input[input_idx];
                            const int output_idx[] = {
                                b, mod(z, nz), mod(y, ny), mod(x, nx)};
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
            preamble=kernel + mod_cuda,
            reduce_dims=False)

        return _gridding1_cuda_complex, _gridding2_cuda_complex, \
            _gridding3_cuda_complex

    _interpolate_cuda = {}
    _gridding_cuda = {}
    _gridding_cuda_complex = {}
    for kernel in KERNELS:
        _interpolate_cuda[kernel] = _get_interpolate_cuda(kernel)
        _gridding_cuda[kernel] = _get_gridding_cuda(kernel)
        _gridding_cuda_complex[kernel] = _get_gridding_cuda_complex(kernel)
