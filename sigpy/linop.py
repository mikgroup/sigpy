import numpy as np

from itertools import product
from sigpy import config, fft, nufft, util, interp, conv, wavelet

if config.cupy_enabled:
    import cupy as cp


def _check_shape_positive(shape):

    if not all(s > 0 for s in shape):
        raise ValueError(
            'Shapes must be positive, got {shape}'.format(shape=shape))


class Linop(object):
    """Abstraction for linear operator.

    Linop can be called on or multiply to an array to perform a linear operation.
    Its adjoint linear operator can be obtained using the .H attribute.
    Linops can be scaled, added, subtracted, stacked and composed.

    Args:
        oshape: Output shape.
        ishape: Input shape.
        repr_str (string or None): default: class name.

    Attributes:
        oshape: output shape.
        ishape: input shape.
        H: adjoint linear operator.
    """

    def __init__(self, oshape, ishape, repr_str=None):
        self.oshape = list(oshape)
        self.ishape = list(ishape)

        _check_shape_positive(oshape)
        _check_shape_positive(ishape)

        if repr_str is None:
            self.repr_str = self.__class__.__name__
        else:
            self.repr_str = repr_str

    def _check_domain(self, input):

        for i1, i2 in zip(input.shape, self.ishape):
            if i2 != -1 and i1 != i2:
                raise ValueError('input shape mismatch for {s}, got {input_shape}'.format(
                    s=self, input_shape=input.shape))

    def _check_codomain(self, output):

        for o1, o2 in zip(output.shape, self.oshape):
            if o2 != -1 and o1 != o2:
                raise ValueError('output shape mismatch for {s}, got {output_shape}'.format(
                    s=self, output_shape=output.shape))

    def _apply(self, input):
        raise NotImplementedError

    def apply(self, input):
        self._check_domain(input)
        with util.get_device(input):
            output = self._apply(input)
        self._check_codomain(output)

        return output

    def _adjoint_linop(self):
        raise NotImplementedError

    @property
    def H(self):
        return self._adjoint_linop()

    def __call__(self, input):
        return self.__mul__(input)

    def __mul__(self, input):
        if isinstance(input, Linop):
            return Compose([self, input])
        elif np.isscalar(input):
            M = Multiply(self.ishape, input)
            return Compose([self, M])
        elif isinstance(input, util.get_xp(input).ndarray):
            return self.apply(input)

        return NotImplemented

    def __rmul__(self, input):

        if np.isscalar(input):
            M = Multiply(self.oshape, input)
            return Compose([M, self])

        return NotImplemented

    def __add__(self, input):

        if isinstance(input, Linop):
            return Add([self, input])
        else:
            raise NotImplementedError

    def __neg__(self):

        return -1 * self

    def __sub__(self, input):

        return self.__add__(-input)

    def __repr__(self):

        return '<{oshape}x{ishape}> {repr_str} Linop>'.format(
            oshape=self.oshape, ishape=self.ishape, repr_str=self.repr_str)


class Identity(Linop):
    """Identity linear operator.

    Args:
        shape (tuple of ints): Input shape
    """

    def __init__(self, shape):

        super().__init__(shape, shape)

    def _apply(self, input):
        return input

    def _adjoint_linop(self):
        return self


class Move(Linop):
    """Move input between devices.

    Args:
        shape (tuple of ints): Input/output shape.
        odevice (tuple of ints): Output device
        idevice (tuple of ints): Input device
    """

    def __init__(self, shape, odevice, idevice):

        self.odevice = odevice
        self.idevice = idevice

        super().__init__(shape, shape)

    def _apply(self, input):
        return util.move(input, self.odevice)

    def _adjoint_linop(self):
        return Move(self.ishape, self.idevice, self.odevice)


class Conj(Linop):
    """Returns complex conjugate of linear operator.

    Args:
        A (Linop): Input linear operator.
    """

    def __init__(self, A):

        self.A = A

        super().__init__(A.oshape, A.ishape, repr_str=A.repr_str)

    def _apply(self, input):

        device = util.get_device(input)
        with device:
            input = device.xp.conj(input)

        output = self.A._apply(input)

        device = util.get_device(output)
        with device:
            return device.xp.conj(output)

    def _adjoint_linop(self):
        return Conj(self.A.H)


class Add(Linop):
    """Addition of linear operators.

    ishape, and oshape must match.

    Args:
        linops (list of Linops): Input linear operators.

    Returns: 
        Linop: linops[0] + linops[1] + ... + linops[n - 1]
    """

    def __init__(self, linops):

        _check_linops_same_ishape(linops)
        _check_linops_same_oshape(linops)

        self.linops = linops
        oshape = linops[0].oshape
        ishape = linops[0].ishape

        super().__init__(oshape, ishape,
                         repr_str=' + '.join([linop.repr_str for linop in linops]))

    def _apply(self, input):

        output = 0
        for linop in self.linops:
            outputi = linop._apply(input)
            with util.get_device(outputi):
                output += outputi

        return output

    def _adjoint_linop(self):
        return Add([linop.H for linop in self.linops])


def _check_compose_linops(linops):

    for linop1, linop2 in zip(linops[:-1], linops[1:]):
        if (linop1.ishape != linop2.oshape):
            raise ValueError('cannot compose {linop1} and {linop2}.'.format(
                linop1=linop1, linop2=linop2))


def _combine_compose_linops(linops):

    combined_linops = []
    for linop in linops:
        if isinstance(linop, Compose):
            combined_linops += linop.linops
        else:
            combined_linops.append(linop)

    return combined_linops


class Compose(Linop):
    """Composition of linear operators.

    Args:
        linops (list of Linops): Linear operators to be composed.

    Returns: 
        Linop: linops[0] * linops[1] * ... * linops[n - 1]
    """

    def __init__(self, linops):

        _check_compose_linops(linops)
        self.linops = _combine_compose_linops(linops)

        super().__init__(self.linops[0].oshape, self.linops[-1].ishape,
                         repr_str=' * '.join([linop.repr_str for linop in linops]))

    def _apply(self, input):

        output = input
        for linop in self.linops[::-1]:
            output = linop._apply(output)
            linop._check_codomain(output)

        return output

    def _adjoint_linop(self):
        return Compose([linop.H for linop in self.linops[::-1]])


def _check_linops_same_ishape(linops):

    for linop in linops:
        if (linop.ishape != linops[0].ishape):
            raise ValueError('Linops must have the same ishape, got {linops}.'.format(
                linops=linops))


def _check_linops_same_oshape(linops):

    for linop in linops:
        if (linop.oshape != linops[0].oshape):
            raise ValueError('Linops must have the same oshape, got {linops}.'.format(
                linops=linops))


def _hstack_params(shapes, axis):

    if axis is None:
        return _hstack_params([[util.prod(shape)] for shape in shapes], 0)

    ishape = list(shapes[0])
    ndim = len(ishape)
    idx = shapes[0][axis]
    indices = []

    for shape in shapes[1:]:
        if len(shape) != ndim:
            raise Exception(
                'Shapes must have the same lengths to concatenate.')

        for i in range(ndim):
            if i == axis:
                ishape[i] += shape[i]
                indices.append(idx)
                idx += shape[i]
            elif shape[i] != ishape[i]:
                raise Exception(
                    'Shapes not along axis must be the same to concatenate.')

    return ishape, indices


class Hstack(Linop):
    """Horizontally stack linear operators.

    Creates a Linop that splits the input, applies Linops independently, and sums outputs.
    In matrix form, this is equivalant to given matrices {A1, ..., An}, returns [A1, ..., An].

    Args:
        linops (list of Linops): list of linops with the same output shape.
        axis (int or None): If None, inputs are vectorized and concatenated.
            Otherwise, inputs are stacked along axis.
    """

    def __init__(self, linops, axis=None):
        self.nops = len(linops)
        _check_linops_same_oshape(linops)

        self.linops = linops
        self.axis = axis

        ishape, self.indices = _hstack_params(
            [linop.ishape for linop in self.linops], axis)
        oshape = self.linops[0].oshape

        super().__init__(oshape, ishape)

    def _apply(self, input):

        device = util.get_device(input)
        xp = device.xp
        output = 0
        with device:
            for n, linop in enumerate(self.linops):
                if n == 0:
                    start = 0
                else:
                    start = self.indices[n - 1]

                if n == self.nops - 1:
                    end = None
                else:
                    end = self.indices[n]

                if self.axis is None:
                    output += linop(input[start:end].reshape(linop.ishape))
                else:
                    ndim = len(linop.oshape)
                    axis = self.axis % ndim

                    slc = ([slice(None)] * axis + [slice(start, end)] +
                           [slice(None)] * (ndim - axis - 1))

                    output += linop(input[slc])

        return output

    def _adjoint_linop(self):

        return Vstack([op.H for op in self.linops], axis=self.axis)


def _vstack_params(shapes, axis):

    if axis is None:
        return _vstack_params([[util.prod(shape)] for shape in shapes], 0)

    oshape = list(shapes[0])
    ndim = len(oshape)
    idx = shapes[0][axis]
    indices = []

    for shape in shapes[1:]:
        if len(shape) != ndim:
            raise Exception(
                'Shapes must have the same lengths to concatenate.')

        for i in range(ndim):
            if i == axis:
                oshape[i] += shape[i]
                indices.append(idx)
                idx += shape[i]
            elif shape[i] != oshape[i]:
                raise Exception(
                    'Shapes not along axis must be the same to concatenate.')

    return oshape, indices


class Vstack(Linop):
    """Vertically stack linear operators.

    Creates a Linop that applies linops independently, and concatenates outputs.
    In matrix form, this is equivalant to given matrices {A1, ..., An}, 
    returns [A1.T, ..., An.T].T.

    Args:
        linops (list of Linops): list of linops with the same input shape.
        axis (int or None): If None, outputs are vectorized and concatenated.
    """

    def __init__(self, linops, axis=None):
        self.nops = len(linops)
        _check_linops_same_ishape(linops)

        self.axis = axis
        self.linops = linops

        oshape, self.indices = _vstack_params(
            [linop.oshape for linop in self.linops], axis)
        ishape = self.linops[0].ishape

        super().__init__(oshape, ishape)

    def _apply(self, input):

        device = util.get_device(input)
        xp = device.xp
        output = util.empty(self.oshape, dtype=input.dtype, device=device)
        with device:
            for n, linop in enumerate(self.linops):
                if n == 0:
                    start = 0
                else:
                    start = self.indices[n - 1]

                if n == self.nops - 1:
                    end = None
                else:
                    end = self.indices[n]

                if self.axis is None:
                    output[start:end] = linop(input).ravel()
                else:
                    ndim = len(linop.oshape)
                    axis = self.axis % ndim
                    slc = ([slice(None)] * axis + [slice(start, end)] +
                           [slice(None)] * (ndim - axis - 1))
                    output[slc] = linop(input)

        return output

    def _adjoint_linop(self):

        return Hstack([op.H for op in self.linops], axis=self.axis)


class Diag(Linop):
    """Diagonally stack linear operators.

    Create a Linop that splits input, applies linops independently, and concatenates outputs.
    In matrix form, given matrices {A1, ..., An}, returns diag([A1, ..., An]).

    Args:
        linops (list of Linops): list of linops with the same input and output shape.
        axis (int or None): If None, inputs/outputs are vectorized and concatenated.
    """

    def __init__(self, linops, axis=None):
        self.nops = len(linops)

        self.linops = linops
        self.axis = axis
        ishape, self.iindices = _hstack_params(
            [linop.ishape for linop in self.linops], axis)
        oshape, self.oindices = _vstack_params(
            [linop.oshape for linop in self.linops], axis)

        super().__init__(oshape, ishape)

    def _apply(self, input):

        device = util.get_device(input)
        xp = device.xp
        output = util.empty(self.oshape, dtype=input.dtype, device=device)
        with device:
            for n, linop in enumerate(self.linops):
                if n == 0:
                    istart = 0
                    ostart = 0
                else:
                    istart = self.iindices[n - 1]
                    ostart = self.oindices[n - 1]

                if n == self.nops - 1:
                    iend = None
                    oend = None
                else:
                    iend = self.iindices[n]
                    oend = self.oindices[n]

                if self.axis is None:
                    output[ostart:oend] = linop(
                        input[istart:iend].reshape(linop.ishape)).ravel()
                else:
                    ndim = len(linop.oshape)
                    axis = self.axis % ndim
                    oslc = ([slice(None)] * axis + [slice(ostart, oend)] +
                            [slice(None)] * (ndim - axis - 1))

                    ndim = len(linop.ishape)
                    axis = self.axis % ndim
                    islc = ([slice(None)] * axis + [slice(istart, iend)] +
                            [slice(None)] * (ndim - axis - 1))

                    output[oslc] = linop(input[islc])

        return output

    def _adjoint_linop(self):

        return Diag([op.H for op in self.linops], axis=self.axis)


class Reshape(Linop):
    """Linear operator that reshapes input to given output shape.

    Args:
        oshape (tuple of ints): Output shape.
        ishape (tuple of ints): Input shape.
    """

    def __init__(self, oshape, ishape):
        super().__init__(oshape, ishape)

    def _apply(self, input):
        return input.reshape(self.oshape)

    def _adjoint_linop(self):
        return Reshape(self.ishape, self.oshape)


class Transpose(Linop):
    """Linear operator that transposes input with the given axes.

    Args:
        ishape (tuple of ints): Input shape.
        axes (None or tuple of ints): Axes to transpose input.
    """

    def __init__(self, ishape, axes=None):
        self.axes = axes
        if axes is None:
            self.iaxes = None
            oshape = ishape[::-1]
        else:
            self.iaxes = np.argsort(axes)
            oshape = [ishape[a] for a in axes]

        super().__init__(oshape, ishape)

    def _apply(self, input):
        return input.transpose(self.axes)

    def _adjoint_linop(self):

        if self.axes is None:
            iaxes = None
            oshape = self.ishape[::-1]
        else:
            iaxes = np.argsort(self.axes)
            oshape = [self.ishape[a] for a in self.axes]

        return Transpose(oshape, axes=iaxes)


class FFT(Linop):
    """FFT linear operator.

    Args:
        ishape (tuple of int): Input shape
        axes (None or tuple of int): Axes to perform FFT. If None, applies on all axes.
        center (bool): Toggle center FFT.
    """

    def __init__(self, shape, axes=None, center=True):

        self.axes = axes
        self.center = center

        super().__init__(shape, shape)

    def _apply(self, input):

        return fft.fft(input, axes=self.axes, center=self.center)

    def _adjoint_linop(self):
        return IFFT(self.ishape, axes=self.axes, center=self.center)


class IFFT(Linop):
    """IFFT linear operator.

    Args:
        ishape (tuple of int): Input shape
        axes (None or tuple of int): Axes to perform FFT. If None, applies on all axes.
        center (bool): Toggle center FFT.
    """

    def __init__(self, shape, axes=None, center=True):

        self.axes = axes
        self.center = center

        super().__init__(shape, shape)

    def _apply(self, input):

        return fft.ifft(input, axes=self.axes, center=self.center)

    def _adjoint_linop(self):
        return FFT(self.ishape, axes=self.axes, center=self.center)


def _get_matmul_oshape(ishape, mshape, adjoint):

    ishape_exp, mshape_exp = util._expand_shapes(ishape, mshape)
    if adjoint:
        mshape_exp[-1], mshape_exp[-2] = mshape_exp[-2], mshape_exp[-1]

    max_ndim = max(len(ishape), len(mshape))
    oshape = []
    for i, m, d in zip(ishape_exp[:-2], mshape_exp[:-2], range(max_ndim - 2)):
        if not (i == m or i == 1 or m == 1):
            raise ValueError('Invalid shapes: {ishape}, {mshape}.'.format(
                ishape=ishape, mshape=mshape))

        oshape.append(max(i, m))

    if mshape_exp[-1] != ishape_exp[-2]:
        raise ValueError('Invalid shapes: {ishape}, {mshape}.'.format(
            ishape=ishape, mshape=mshape))

    oshape += [mshape_exp[-2], ishape_exp[-1]]

    return oshape


def _get_matmul_adjoint_sum_axes(oshape, ishape, mshape):

    ishape_exp, mshape_exp = util._expand_shapes(ishape, mshape)
    max_ndim = max(len(ishape), len(mshape))
    sum_axes = []
    for i, m, o, d in zip(ishape_exp[:-2], mshape_exp[:-2], oshape[:-2], range(max_ndim - 2)):
        if (i == 1 and (m != 1 or o != 1)):
            sum_axes.append(d)

    return sum_axes


class MatMul(Linop):
    """Linear operator that performs matrix multiplication.

    Args:
        ishape (tuple of ints): Input shape. It must be able to broadcast with mat.shape.
        mat (array): Matrix of shape [..., m, n]
        adjoint (bool): Toggle adjoint. If True, performs conj(mat).swapaxes(-1, -2)
            before performing matrix multiplication.
    """

    def __init__(self, ishape, mat, adjoint=False):
        self.mat = mat
        self.adjoint = adjoint

        oshape = _get_matmul_oshape(ishape, mat.shape, adjoint)

        super().__init__(oshape, ishape)

    def _apply(self, input):

        device = util.get_device(input)
        xp = device.xp
        mat = util.move(self.mat, device)
        with device:
            if self.adjoint:
                mat = xp.conj(mat).swapaxes(-1, -2)

            return xp.matmul(mat, input)

    def _adjoint_linop(self):

        sum_axes = _get_matmul_adjoint_sum_axes(
            self.oshape, self.ishape, self.mat.shape)

        M = MatMul(self.oshape, self.mat, adjoint=not self.adjoint)
        S = Sum(M.oshape, sum_axes)
        R = Reshape(self.ishape, S.oshape)
        return R * S * M


def _get_right_matmul_oshape(ishape, mshape, adjoint):

    ishape_exp, mshape_exp = util._expand_shapes(ishape, mshape)
    if adjoint:
        mshape_exp[-1], mshape_exp[-2] = mshape_exp[-2], mshape_exp[-1]

    max_ndim = max(len(ishape), len(mshape))
    oshape = []
    for i, m, d in zip(ishape_exp[:-2], mshape_exp[:-2], range(max_ndim - 2)):
        if not (i == m or i == 1 or m == 1):
            raise ValueError('Invalid shapes: {ishape}, {mshape}.'.format(
                ishape=ishape, mshape=mshape))

        oshape.append(max(i, m))

    if ishape_exp[-1] != mshape_exp[-2]:
        raise ValueError('Invalid shapes: {ishape}, {mshape}.'.format(
            ishape=ishape, mshape=mshape))

    oshape += [ishape_exp[-2], mshape_exp[-1]]

    return oshape


class RightMatMul(Linop):
    """Linear operator that performs matrix multiplication on the right.

    Args:
        ishape (tuple of ints): Input shape. It must be able to broadcast with mat.shape.
        mat (array): Matrix of shape [..., m, n]
        adjoint (bool): Toggle adjoint. If True, performs conj(mat).swapaxes(-1, -2)
            before performing matrix multiplication.
    """

    def __init__(self, ishape, mat, adjoint=False):
        self.mat = mat
        self.adjoint = adjoint

        oshape = _get_right_matmul_oshape(ishape, mat.shape, adjoint)

        super().__init__(oshape, ishape)

    def _apply(self, input):

        device = util.get_device(input)
        xp = device.xp
        mat = util.move(self.mat, device)
        with device:
            if self.adjoint:
                mat = xp.conj(mat).swapaxes(-1, -2)

            return xp.matmul(input, mat)

    def _adjoint_linop(self):

        sum_axes = _get_matmul_adjoint_sum_axes(
            self.oshape, self.ishape, self.mat.shape)

        M = RightMatMul(self.oshape, self.mat, adjoint=not self.adjoint)
        S = Sum(M.oshape, sum_axes)
        R = Reshape(self.ishape, S.oshape)
        return R * S * M


def _get_multiply_oshape(ishape, mshape):

    ishape_exp, mshape_exp = util._expand_shapes(ishape, mshape)
    max_ndim = max(len(ishape), len(mshape))
    oshape = []
    for i, m, d in zip(ishape_exp, mshape_exp, range(max_ndim)):
        if not (i == m or i == 1 or m == 1):
            raise ValueError('Invalid shapes: {ishape}, {mshape}.'.format(
                ishape=ishape, mshape=mshape))

        oshape.append(max(i, m))

    return oshape


def _get_multiply_adjoint_sum_axes(oshape, ishape, mshape):

    ishape_exp, mshape_exp = util._expand_shapes(ishape, mshape)
    max_ndim = max(len(ishape), len(mshape))
    sum_axes = []
    for i, m, o, d in zip(ishape_exp, mshape_exp, oshape, range(max_ndim)):
        if (i == 1 and (m != 1 or o != 1)):
            sum_axes.append(d)

    return sum_axes


class Multiply(Linop):
    """Multiplication linear operator.

    Args:
        ishape (tuple of ints): Input shape.
        mult (array): Array to multiply.
    """

    def __init__(self, ishape, mult, conj=False):
        self.mult = mult
        self.conj = conj
        if np.isscalar(mult):
            self.mshape = [1]
        else:
            self.mshape = mult.shape

        oshape = _get_multiply_oshape(ishape, self.mshape)

        super().__init__(oshape, ishape)

    def _apply(self, input):
        device = util.get_device(input)
        xp = device.xp

        if np.isscalar(self.mult):
            if self.mult == 1:
                return input

            mult = util.array(self.mult, dtype=input.dtype, device=device)
        else:
            mult = util.move(self.mult, device).astype(input.dtype)

        with device:
            if self.conj:
                mult = xp.conj(mult)

            return input * mult

    def _adjoint_linop(self):

        sum_axes = _get_multiply_adjoint_sum_axes(
            self.oshape, self.ishape, self.mshape)

        M = Multiply(self.oshape, self.mult, conj=not self.conj)
        S = Sum(M.oshape, sum_axes)
        R = Reshape(self.ishape, S.oshape)
        return R * S * M


class Interp(Linop):
    """Interpolation linear operator.

    Args:
        ishape (tuple of ints): Input shape = batch_shape + grd_shape
        coord (array): Coordinates, values from - nx / 2 to nx / 2 - 1.
                ndim can only be 1, 2 or 3, of shape pts_shape + [ndim]
        width (float): Width of interp. kernel in grid size.
        table (array): Look-up table of kernel K, from K[0] to K[width].
        scale (float): Scaling of coordinates.
        shift (float): Shifting of coordinates.
    """

    def __init__(self, ishape, coord, width, table, scale=1, shift=0):

        ndim = coord.shape[-1]

        oshape = list(ishape[:-ndim]) + list(coord.shape[:-1])

        self.coord = coord
        self.width = width
        self.table = table
        self.shift = shift
        self.scale = scale

        super().__init__(oshape, ishape)

    def _apply(self, input):

        device = util.get_device(input)
        coord = util.move(self.coord, device)
        table = util.move(self.table, device)
        shift = util.move(self.shift, device)

        with device:
            return interp.interp(input, self.width, table,
                                 coord * self.scale + shift)

    def _adjoint_linop(self):

        return Gridding(self.ishape, self.coord, self.width, self.table,
                        scale=self.scale, shift=self.shift)


class Gridding(Linop):
    """Gridding linear operator.

    Args:
        oshape (tuple of ints): Output shape = batch_shape + pts_shape
        ishape (tuple of ints): Input shape = batch_shape + grd_shape
        coord (array): Coordinates, values from - nx / 2 to nx / 2 - 1.
                ndim can only be 1, 2 or 3. of shape pts_shape + [ndim]
        width (float): Width of interp. kernel in grid size
        table (array): Llook-up table of kernel K, from K[0] to K[width]
            scale (float): Scaling of coordinates.
            shift (float): Shifting of coordinates.
    """

    def __init__(self, oshape, coord, width, table, scale=1, shift=0):

        ndim = coord.shape[-1]

        ishape = list(oshape[:-ndim]) + list(coord.shape[:-1])

        self.coord = coord
        self.width = width
        self.table = table
        self.shift = shift
        self.scale = scale

        super().__init__(oshape, ishape)

    def _apply(self, input):
        device = util.get_device(input)
        coord = util.move(self.coord, device)
        table = util.move(self.table, device)
        shift = util.move(self.shift, device)

        with device:
            return interp.gridding(input, self.oshape, self.width, table,
                                   coord * self.scale + shift)

    def _adjoint_linop(self):

        return Interp(self.oshape, self.coord, self.width, self.table,
                      scale=self.scale, shift=self.shift)


class Resize(Linop):
    """Resize linear operator.

    Args:
        oshape (tuple of int): Output shape.
        ishape (tuple of int): Input shape
    """

    def __init__(self, oshape, ishape, ishift=None, oshift=None):

        self.ishift = ishift
        self.oshift = oshift

        super().__init__(oshape, ishape)

    def _apply(self, input):

        return util.resize(input, self.oshape, ishift=self.ishift, oshift=self.oshift)

    def _adjoint_linop(self):

        return Resize(self.ishape, self.oshape, ishift=self.oshift, oshift=self.ishift)


class Flip(Linop):
    """Flip linear operator.

    Args:
        shape (tuple of int): Input shape
    """

    def __init__(self, shape, axes=None):
        self.axes = axes

        super().__init__(shape, shape)

    def _apply(self, input):
        return util.flip(input, self.axes)

    def _adjoint_linop(self):
        return self


class Downsample(Linop):
    """Downsampling linear operator.

    Args:
        ishape (tuple of ints): Input shape.
        factor (tuple of ints): Downsampling factor.
        shift (None of tuple of ints): Shifts before down-sampling.
    """

    def __init__(self, ishape, factors, shift=None):
        self.factors = factors

        if shift is None:
            shift = [0] * len(ishape)
        self.shift = shift

        oshape = [((i - s + f - 1) // f)
                  for i, f, s in zip(ishape, factors, shift)]

        super().__init__(oshape, ishape)

    def _apply(self, input):
        return util.downsample(input, self.factors, shift=self.shift)

    def _adjoint_linop(self):
        return Upsample(self.ishape, self.factors, shift=self.shift)


class Upsample(Linop):
    """Upsampling linear operator.

    Args:
        ishape (tuple of ints): Input shape.
        factor (tuple of ints): Upsampling factor.
        shift (None of tuple of ints): Shifts before up-sampling.
    """

    def __init__(self, oshape, factors, shift=None):
        self.factors = factors

        if shift is None:
            shift = [0] * len(oshape)
        self.shift = shift

        ishape = [((i - s + f - 1) // f)
                  for i, f, s in zip(oshape, factors, shift)]

        super().__init__(oshape, ishape)

    def _apply(self, input):
        return util.upsample(input, self.oshape, self.factors, shift=self.shift)

    def _adjoint_linop(self):
        return Downsample(self.oshape, self.factors, shift=self.shift)


class Circshift(Linop):
    """Circular shift linear operator.
    
    Args:
        shape (tuple of ints): Input/output shape.
        shift (tuple of ints): Shifts.
        axes (None or tuple of ints): Axes to perform circular shift.
    """

    def __init__(self, shape, shift, axes=None):

        self.axes = axes
        self.shift = shift
        self.ishift = [-s for s in self.shift]

        super().__init__(shape, shape)

    def _apply(self, input):

        return util.circshift(input, self.shift, self.axes)

    def _adjoint_linop(self):

        return Circshift(self.ishape, [-s for s in self.shift], axes=self.axes)


class Wavelet(Linop):
    """Wavelet transform linear operator.

    Currently only has CPU implementation. GPU inputs will be copied to CPU,
    and back to compute on CPU.

    Args:
        ishape (tuple of int): Input shape.
        axes (None or tuple of int): Axes to perform wavelet transform.
        wave_name (str): Wavelet name.
        level (None or int): Number of wavelet levels.
    """

    def __init__(self, ishape, axes=None, wave_name='db4', level=None):
        self.wave_name = wave_name
        self.axes = axes
        self.level = level
        oshape, _ = wavelet.get_wavelet_shape(ishape, wave_name, axes, level)

        super().__init__(oshape, ishape)

    def _apply(self, input):
        return wavelet.fwt(
            input, wave_name=self.wave_name, axes=self.axes, level=self.level)

    def _adjoint_linop(self):
        return InverseWavelet(
            self.ishape, axes=self.axes, wave_name=self.wave_name, level=self.level)


class InverseWavelet(Linop):
    """Wavelet transform linear operator.

    Currently only has CPU implementation. GPU inputs will be copied to CPU,
    and back to compute on CPU.

    Args:
        oshape (tuple of int): Output shape.
        axes (None or tuple of int): Axes to perform wavelet transform.
        wave_name (str): Wavelet name.
        level (None or int): Number of wavelet levels.
    """

    def __init__(self, oshape, axes=None, wave_name='db4', level=None):
        self.wave_name = wave_name
        self.axes = axes
        self.level = level
        ishape, self.coeff_slices = wavelet.get_wavelet_shape(
            oshape, wave_name, axes, level)
        super().__init__(oshape, ishape)

    def _apply(self, input):
        return wavelet.iwt(
            input, self.oshape, self.coeff_slices,
            wave_name=self.wave_name, axes=self.axes, level=self.level)

    def _adjoint_linop(self):
        return Wavelet(self.oshape, axes=self.axes, wave_name=self.wave_name, level=self.level)


class Sum(Linop):
    """Sum linear operator. Accumulate axes by summing.
    
    Args:
        ishape (tuple of ints): Input shape.
        axes (tuple of ints): Axes to sum over.
    """

    def __init__(self, ishape, axes):

        self.axes = tuple(i % len(ishape) for i in axes)
        oshape = [ishape[i] for i in range(len(ishape)) if i not in self.axes]

        super().__init__(oshape, ishape)

    def _apply(self, input):

        device = util.get_device(input)
        xp = device.xp
        with device:
            return xp.sum(input, axis=self.axes)

    def _adjoint_linop(self):

        return Tile(self.ishape, self.axes)


class Tile(Linop):
    """Tile linear operator.
    
    Args:
        oshape (tuple of ints): Output shape.
        axes (tuple of ints): Axes to tile.
    """

    def __init__(self, oshape, axes):

        self.axes = tuple(a % len(oshape) for a in axes)
        ishape = [oshape[d] for d in range(len(oshape)) if d not in self.axes]
        self.expanded_ishape = []
        self.reps = []
        for d in range(len(oshape)):
            if d in self.axes:
                self.expanded_ishape.append(1)
                self.reps.append(oshape[d])
            else:
                self.expanded_ishape.append(oshape[d])
                self.reps.append(1)

        super().__init__(oshape, ishape)

    def _apply(self, input):

        device = util.get_device(input)
        xp = device.xp
        with device:
            return xp.tile(input.reshape(self.expanded_ishape), self.reps)

    def _adjoint_linop(self):

        return Sum(self.oshape, self.axes)


class TensorToBlocks(Linop):
    """Block partition input array. Block shape must divide input shape.
    
    Args:
        ishape (tuple of ints): Input shape.
        blk_shape (tuple of ints): Block shape.
    """

    def __init__(self, ishape, blk_shape):

        if not all([i % b == 0 for i, b in zip(ishape, blk_shape)]):
            raise ValueError('blk_shape should divide ishape, got {ishape} and {blk_shape}.'.format(
                ishape=ishape, blk_shape=blk_shape))

        ndim = len(blk_shape)
        self.blk_shape = list(blk_shape)

        batch_shape = ishape[:-ndim]
        batch_ndim = len(batch_shape)
        num_blocks = [i // b for i, b in zip(ishape[-ndim:], self.blk_shape)]
        oshape = batch_shape + num_blocks + self.blk_shape

        self.ireshape = batch_shape.copy()
        for n, b in zip(num_blocks, blk_shape):
            self.ireshape += [n, b]

        self.perm = (list(range(batch_ndim)) +
                     [batch_ndim + 2 * d for d in range(ndim)] +
                     [batch_ndim + 2 * d + 1 for d in range(ndim)])

        super().__init__(oshape, ishape)

    def _apply(self, input):

        with util.get_device(input):
            return input.reshape(self.ireshape).transpose(self.perm).reshape(self.oshape)

    def _adjoint_linop(self):

        return BlocksToTensor(self.ishape, self.blk_shape)


class BlocksToTensor(Linop):
    """Sum blocks to array. Block shape must divide output shape.
    
    Args:
        oshape (tuple of ints): Output shape.
        blk_shape (tuple of ints): Block shape.
    """

    def __init__(self, oshape, blk_shape):

        if not all([o % b == 0 for o, b in zip(oshape, blk_shape)]):
            raise ValueError(
                'blk_shape must divide oshape, got {oshape}, and {blk_shape}.')

        ndim = len(blk_shape)
        self.blk_shape = list(blk_shape)

        batch_shape = oshape[:-ndim]
        batch_ndim = len(batch_shape)
        num_blocks = [i // b for i, b in zip(oshape[-ndim:], self.blk_shape)]
        ishape = batch_shape + num_blocks + self.blk_shape

        self.perm = list(range(batch_ndim))
        for i in range(ndim):
            self.perm += [batch_ndim + i, batch_ndim + ndim + i]

        super().__init__(oshape, ishape)

    def _apply(self, input):

        with util.get_device(input):
            return input.transpose(self.perm).reshape(self.oshape)

    def _adjoint_linop(self):

        return TensorToBlocks(self.oshape, self.blk_shape)


def Gradient(ishape):
    """Linear operator that computes numerical gradient.

    Args:
       ishape (tuple of ints): Input shape.
    """

    I = Identity(ishape)
    shifts = list(product(*([[0, 1]] * len(ishape))))
    scale = 1 / (2 * len(shifts))**0.5

    G = scale * Vstack([I - Circshift(ishape, shift) for shift in shifts])
    G.repr_str = 'Gradient'

    return G


class NUFFT(Linop):
    """NUFFT linear operator.

    Args:
        ishape (tuple of int): Input shape.
        coord (array): Coordinates, with values [-ishape / 2, ishape / 2]
        oversamp (float): Oversampling factor.
        width (float): Kernel width.
        n (int): Table sampling number.
    """

    def __init__(self, ishape, coord, oversamp=1.25, width=4.0, n=128):
        self.coord = coord
        self.oversamp = oversamp
        self.width = width
        self.n = n

        ndim = coord.shape[-1]

        oshape = list(ishape[:-ndim]) + list(coord.shape[:-1])

        super().__init__(oshape, ishape)

    def _apply(self, input):

        return nufft.nufft(input, self.coord, oversamp=self.oversamp, width=self.width, n=self.n)

    def _adjoint_linop(self):

        return NUFFTAdjoint(self.ishape, self.coord,
                            oversamp=self.oversamp, width=self.width, n=self.n)


class NUFFTAdjoint(Linop):
    """NUFFT adjoint linear operator.

    Args:
        oshape (tuple of int): Output shape
        coord (array): Coordinates, with values [-ishape / 2, ishape / 2]
        oversamp (float): Oversampling factor.
        width (float): Kernel width.
        n (int): Table sampling number.
    """

    def __init__(self, oshape, coord, oversamp=1.25, width=4.0, n=128):
        self.coord = coord
        self.oversamp = oversamp
        self.width = width
        self.n = n

        ndim = coord.shape[-1]

        ishape = list(oshape[:-ndim]) + list(coord.shape[:-1])

        super().__init__(oshape, ishape)

    def _apply(self, input):

        return nufft.nufft_adjoint(input, self.coord, self.oshape,
                                   oversamp=self.oversamp, width=self.width, n=self.n)

    def _adjoint_linop(self):

        return NUFFTAdjoint(self.ishape, self.coord,
                            oversamp=self.oversamp, width=self.width, n=self.n)


def _get_convolve_adjoint_mode(ishape, fshape, axes, mode):

    if mode == 'full':
        return 'valid'
    else:
        ishape_exp, fshape_exp = util._expand_shapes(ishape, fshape)

        i_greater_f = [ishape_exp[a] >= fshape_exp[a] for a in axes]
        i_smaller_f = [ishape_exp[a] <= fshape_exp[a] for a in axes]
        if all(i_greater_f):
            return 'full'
        elif all(i_smaller_f):
            return 'valid'
        else:
            raise ValueError('ishape should be either all >=, or <= fshape,'
                             'got {ishape}, and {fshape}.'.format(ishape=ishape, fshape=fshape))


def _get_convolve_oshape(ishape, fshape, axes, mode):

    ishape_exp, fshape_exp = util._expand_shapes(ishape, fshape)
    max_ndim = max(len(ishape), len(fshape))
    oshape = []
    for i, f, d in zip(ishape_exp, fshape_exp, range(max_ndim)):

        if d in axes:
            if mode == 'full':
                oshape.append(i + f - 1)
            elif mode == 'valid':
                oshape.append(max(i, f) - min(i, f) + 1)
        else:
            if not (i == f or i == 1 or f == 1):
                raise ValueError('Invalid shapes: {ishape}, {fshape}.'.format(
                    ishape=ishape, fshape=fshape))

            oshape.append(max(i, f))

    return oshape


def _get_convolve_adjoint_sum_axes(oshape, ishape, fshape, axes):

    ishape_exp, fshape_exp = util._expand_shapes(ishape, fshape)
    max_ndim = max(len(ishape), len(fshape))
    sum_axes = []
    for i, f, o, d in zip(ishape_exp, fshape_exp, oshape, range(max_ndim)):
        if d not in axes:
            if (i == 1 and (f != 1 or o != 1)):
                sum_axes.append(d)

    return sum_axes


class Convolve(Linop):
    """Convolve linear operator.
    
    Args:
        ishape (tuple of ints): Input shape.
        filt (array): Filter.
        axes (None or tuple of ints): Axes to perform convolution.
        mode (str): {'full', 'valid'}
    """

    def __init__(self, ishape, filt, axes=None, mode='full'):
        self.filt = filt

        max_ndim = max(len(ishape), filt.ndim)
        self.axes = util._normalize_axes(axes, max_ndim)
        self.mode = mode

        self.fshape = list(filt.shape)
        oshape = _get_convolve_oshape(ishape, self.fshape, self.axes, mode)

        super().__init__(oshape, ishape)

    def _apply(self, input):

        return conv.convolve(input, self.filt, axes=self.axes, mode=self.mode)

    def _adjoint_linop(self):

        mode = _get_convolve_adjoint_mode(
            self.ishape, self.fshape, self.axes, self.mode)
        sum_axes = _get_convolve_adjoint_sum_axes(
            self.oshape, self.ishape, self.fshape, self.axes)

        C = Correlate(self.oshape, self.filt, axes=self.axes, mode=mode)
        S = Sum(C.oshape, axes=sum_axes)
        R = Reshape(self.ishape, S.oshape)

        return R * S * C


class Correlate(Linop):
    """Correlate linear operator.
    
    Args:
        ishape (tuple of ints): Input shape.
        filt (array): Filter.
        axes (None or tuple of ints): Axes to perform convolution.
        mode (str): {'full', 'valid'}
    """

    def __init__(self, ishape, filt, axes=None, mode='full'):
        self.filt = filt

        max_ndim = max(len(ishape), filt.ndim)
        self.axes = util._normalize_axes(axes, max_ndim)
        self.mode = mode

        self.fshape = list(filt.shape)
        oshape = _get_convolve_oshape(ishape, self.fshape, self.axes, mode)

        super().__init__(oshape, ishape)

    def _apply(self, input):
        return conv.correlate(input, self.filt, axes=self.axes, mode=self.mode)

    def _adjoint_linop(self):

        mode = _get_convolve_adjoint_mode(
            self.ishape, self.fshape, self.axes, self.mode)
        sum_axes = _get_convolve_adjoint_sum_axes(
            self.oshape, self.ishape, self.fshape, self.axes)

        C = Convolve(self.oshape, self.filt, axes=self.axes, mode=mode)
        S = Sum(C.oshape, axes=sum_axes)
        R = Reshape(self.ishape, S.oshape)

        return R * S * C


if config.cudnn_enabled:

    class CudnnConvolveData(Linop):
        """
        ishape - (b, c_I, m_1, m_2, ..., m_N)
        filt - (c_O, c_I, n_1, n_2, ..., n_N)
        """

        def __init__(self, x_shape, W, mode='full'):
            self.W = W
            self.mode = mode
            y_shape = conv._get_cudnn_convolve_y_shape(x_shape, W.shape, mode)

            super().__init__(y_shape, x_shape)

        def _apply(self, input):
            return conv.cudnn_convolve(input, self.W, mode=self.mode)

        def _adjoint_linop(self):

            return CudnnConvolveBackwardData(self.ishape, self.W, mode=self.mode)

    class CudnnConvolveBackwardData(Linop):

        def __init__(self, x_shape, W, mode='full'):
            self.W = W
            self.mode = mode
            y_shape = conv._get_cudnn_convolve_y_shape(x_shape, W.shape, mode)

            super().__init__(x_shape, y_shape)

        def _apply(self, input):
            return conv.cudnn_convolve_backward_data(self.W, input, mode=self.mode)

        def _adjoint_linop(self):
            return CudnnConvolveData(self.oshape, self.W, mode=self.mode)

    class CudnnConvolveFilter(Linop):

        def __init__(self, W_shape, x, mode='full'):
            self.x = x
            self.mode = mode
            y_shape = conv._get_cudnn_convolve_y_shape(x.shape, W_shape, mode)

            super().__init__(y_shape, W_shape)

        def _apply(self, input):
            return conv.cudnn_convolve(self.x, input, mode=self.mode)

        def _adjoint_linop(self):
            return CudnnConvolveBackwardFilter(self.ishape, self.x, mode=self.mode)

    class CudnnConvolveBackwardFilter(Linop):

        def __init__(self, W_shape, x, mode='full'):
            self.x = x
            self.mode = mode
            y_shape = conv._get_cudnn_convolve_y_shape(x.shape, W_shape, mode)

            super().__init__(W_shape, y_shape)

        def _apply(self, input):
            return conv.cudnn_convolve_backward_filter(self.x, input, mode=self.mode)

        def _adjoint_linop(self):
            return CudnnConvolveFilter(self.oshape, self.x, mode=self.mode)
