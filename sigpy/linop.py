# -*- coding: utf-8 -*-
"""This module contains an abstraction class Linop for linear operators,
and provides commonly used linear operators, including signal transforms
such as FFT, NUFFT, and wavelet, and array manipulation operators,
such as reshape, transpose, and resize.
"""
import numpy as np

from sigpy import backend, block, fourier, util, interp, conv, wavelet


def _check_shape_positive(shape):

    if not all(s > 0 for s in shape):
        raise ValueError(
            'Shapes must be positive, got {shape}'.format(shape=shape))


class Linop(object):
    """Abstraction for linear operator.

    Linop can be called or multiplied to an array to
    perform a linear operation.
    Given a Linop A, and an appropriately shaped input x, the following are
    both valid operations to compute x -> A(x):

       >>> y = A * x
       >>> y = A(x)

    Its adjoint linear operator can be obtained using the .H attribute.
    Linops can be scaled, added, subtracted, stacked and composed.
    Here are some example of valid operations on Linop A, Linop B,
    and a scalar a:

       >>> A.H
       >>> a * A + B
       >>> a * A * B

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
                raise ValueError(
                    'input shape mismatch for {s}, got {input_shape}'.format(
                        s=self, input_shape=input.shape))

    def _check_codomain(self, output):
        for o1, o2 in zip(output.shape, self.oshape):
            if o2 != -1 and o1 != o2:
                raise ValueError(
                    'output shape mismatch for {s}, got {output_shape}'.format(
                        s=self, output_shape=output.shape))

    def _apply(self, input):
        raise NotImplementedError

    def apply(self, input):
        """Apply linear operation on input.

        This function checks for the input/output shapes,
        and calls the internal user-defined _apply() method.

        Args:
            input (array): input array of shape `ishape`.

        Returns:
            array: output array of shape `oshape`.

        """
        self._check_domain(input)
        with backend.get_device(input):
            output = self._apply(input)

        self._check_codomain(output)
        return output

    def _adjoint_linop(self):
        raise NotImplementedError

    @property
    def H(self):
        r"""Return adjoint linear operator.

        An adjoint linear operator :math:`A^H` for
        a linear operator :math:`A` is defined as:

        .. math:
            \left< A x, y \right> = \left< x, A^H, y \right>

        Returns:
            Linop: adjoint linear operator.

        """
        return self._adjoint_linop()

    def __call__(self, input):
        return self.__mul__(input)

    def __mul__(self, input):
        if isinstance(input, Linop):
            return Compose([self, input])
        elif np.isscalar(input):
            M = Multiply(self.ishape, input)
            return Compose([self, M])
        elif isinstance(input, backend.get_array_module(input).ndarray):
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


class ToDevice(Linop):
    """Move input between devices.

    Args:
        shape (tuple of ints): Input/output shape.
        odevice (Device): Output device
        idevice (Device): Input device
    """

    def __init__(self, shape, odevice, idevice):
        self.odevice = odevice
        self.idevice = idevice

        super().__init__(shape, shape)

    def _apply(self, input):
        return backend.to_device(input, self.odevice)

    def _adjoint_linop(self):
        return ToDevice(self.ishape, self.idevice, self.odevice)


class AllReduce(Linop):
    """Performs all reduce between MPI ranks.

    Args:
        shape (tuple of ints): Input/output shape.
        comm (Communicator): Communicator.

    """

    def __init__(self, shape, comm, in_place=False):
        self.comm = comm
        self.in_place = in_place

        super().__init__(shape, shape)

    def _apply(self, input):
        with backend.get_device(input):
            if self.in_place:
                output = input
            else:
                output = input.copy()

            self.comm.allreduce(output)
            return output

    def _adjoint_linop(self):
        return AllReduceAdjoint(self.ishape, self.comm, in_place=self.in_place)


class AllReduceAdjoint(Linop):
    """All reduce adjoint operator. Equivalant to identity.

    Args:
        shape (tuple of ints): Input/output shape.
        comm (Communicator): Communicator.

    """

    def __init__(self, shape, comm, in_place=False):
        self.comm = comm
        self.in_place = in_place

        super().__init__(shape, shape)

    def _apply(self, input):
        return input

    def _adjoint_linop(self):
        return AllReduce(self.ishape, self.comm, in_place=self.in_place)


class Conj(Linop):
    """Complex conjugate of linear operator.

    Args:
        A (Linop): Input linear operator.

    """

    def __init__(self, A):
        self.A = A

        super().__init__(A.oshape, A.ishape, repr_str=A.repr_str)

    def _apply(self, input):
        device = backend.get_device(input)
        with device:
            input = device.xp.conj(input)

        output = self.A(input)

        device = backend.get_device(output)
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

        super().__init__(
            oshape, ishape,
            repr_str=' + '.join([linop.repr_str for linop in linops]))

    def _apply(self, input):
        output = 0
        with backend.get_device(output):
            for linop in self.linops:
                output += linop(input)

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

        super().__init__(
            self.linops[0].oshape, self.linops[-1].ishape,
            repr_str=' * '.join([linop.repr_str for linop in linops]))

    def _apply(self, input):
        output = input
        for linop in self.linops[::-1]:
            output = linop(output)

        return output

    def _adjoint_linop(self):
        return Compose([linop.H for linop in self.linops[::-1]])


def _check_linops_same_ishape(linops):
    for linop in linops:
        if (linop.ishape != linops[0].ishape):
            raise ValueError(
                'Linops must have the same ishape, got {linops}.'.format(
                    linops=linops))


def _check_linops_same_oshape(linops):
    for linop in linops:
        if (linop.oshape != linops[0].oshape):
            raise ValueError(
                'Linops must have the same oshape, got {linops}.'.format(
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
                raise RuntimeError(
                    'Shapes not along axis must be the same to concatenate.')

    return ishape, indices


class Hstack(Linop):
    """Horizontally stack linear operators.

    Creates a Linop that splits the input, applies Linops independently,
    and sums outputs.
    In matrix form, this is equivalant to given matrices {A1, ..., An},
    returns [A1, ..., An].

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
        device = backend.get_device(input)
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
                    ndim = len(linop.ishape)
                    axis = self.axis % ndim

                    slc = tuple([slice(None)] * axis + [slice(start, end)] +
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

    Creates a Linop that applies linops independently,
    and concatenates outputs.
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
        device = backend.get_device(input)
        xp = device.xp
        with device:
            output = xp.empty(self.oshape, dtype=input.dtype)
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
                    slc = tuple([slice(None)] * axis + [slice(start, end)] +
                                [slice(None)] * (ndim - axis - 1))
                    output[slc] = linop(input)

        return output

    def _adjoint_linop(self):

        return Hstack([op.H for op in self.linops], axis=self.axis)


class Diag(Linop):
    """Diagonally stack linear operators.

    Create a Linop that splits input, applies linops independently,
    and concatenates outputs.
    In matrix form, given matrices {A1, ..., An}, returns diag([A1, ..., An]).

    Args:
        linops (list of Linops): list of linops with the same input and
            output shape.
        axis (int or None): If None, inputs/outputs are vectorized
            and concatenated.

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
        device = backend.get_device(input)
        xp = device.xp
        with device:
            output = xp.empty(self.oshape, dtype=input.dtype)
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
                    oslc = tuple([slice(None)] * axis + [slice(ostart, oend)] +
                                 [slice(None)] * (ndim - axis - 1))

                    ndim = len(linop.ishape)
                    axis = self.axis % ndim
                    islc = tuple([slice(None)] * axis + [slice(istart, iend)] +
                                 [slice(None)] * (ndim - axis - 1))

                    output[oslc] = linop(input[islc])

        return output

    def _adjoint_linop(self):
        return Diag([op.H for op in self.linops], axis=self.axis)


class Reshape(Linop):
    """Reshape input to given output shape.

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
    """Tranpose input with the given axes.

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
        axes (None or tuple of int): Axes to perform FFT.
            If None, applies on all axes.
        center (bool): Toggle center FFT.

    """

    def __init__(self, shape, axes=None, center=True):

        self.axes = axes
        self.center = center

        super().__init__(shape, shape)

    def _apply(self, input):
        return fourier.fft(input, axes=self.axes, center=self.center)

    def _adjoint_linop(self):
        return IFFT(self.ishape, axes=self.axes, center=self.center)


class IFFT(Linop):
    """IFFT linear operator.

    Args:
        ishape (tuple of int): Input shape
        axes (None or tuple of int): Axes to perform FFT.
            If None, applies on all axes.
        center (bool): Toggle center FFT.

    """

    def __init__(self, shape, axes=None, center=True):

        self.axes = axes
        self.center = center

        super().__init__(shape, shape)

    def _apply(self, input):
        return fourier.ifft(input, axes=self.axes, center=self.center)

    def _adjoint_linop(self):
        return FFT(self.ishape, axes=self.axes, center=self.center)


def _get_matmul_oshape(ishape, mshape, adjoint):
    ishape_exp, mshape_exp = util._expand_shapes(ishape, mshape)
    if adjoint:
        mshape_exp[-1], mshape_exp[-2] = mshape_exp[-2], mshape_exp[-1]

    oshape = []
    for i, m in zip(ishape_exp[:-2], mshape_exp[:-2]):
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
    for i, m, o, d in zip(
            ishape_exp[:-2], mshape_exp[:-2], oshape[:-2],
            range(max_ndim - 2)):
        if (i == 1 and (m != 1 or o != 1)):
            sum_axes.append(d)

    return sum_axes


class MatMul(Linop):
    """Matrix multiplication.

    Args:
        ishape (tuple of ints): Input shape.
            It must be able to broadcast with mat.shape.
        mat (array): Matrix of shape [..., m, n]
        adjoint (bool): Toggle adjoint.
            If True, performs conj(mat).swapaxes(-1, -2)
            before performing matrix multiplication.

    """

    def __init__(self, ishape, mat, adjoint=False):
        self.mat = mat
        self.adjoint = adjoint

        oshape = _get_matmul_oshape(ishape, mat.shape, adjoint)

        super().__init__(oshape, ishape)

    def _apply(self, input):
        device = backend.get_device(input)
        xp = device.xp
        mat = backend.to_device(self.mat, device)
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
    """Matrix multiplication on the right.

    Args:
        ishape (tuple of ints): Input shape.
            It must be able to broadcast with mat.shape.
        mat (array): Matrix of shape [..., m, n]
        adjoint (bool): Toggle adjoint.
            If True, performs conj(mat).swapaxes(-1, -2)
            before performing matrix multiplication.

    """

    def __init__(self, ishape, mat, adjoint=False):
        self.mat = mat
        self.adjoint = adjoint

        oshape = _get_right_matmul_oshape(ishape, mat.shape, adjoint)

        super().__init__(oshape, ishape)

    def _apply(self, input):
        device = backend.get_device(input)
        xp = device.xp
        mat = backend.to_device(self.mat, device)
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
        device = backend.get_device(input)
        xp = device.xp

        with device:
            if np.isscalar(self.mult):
                if self.mult == 1:
                    return input

                mult = self.mult
                if self.conj:
                    mult = mult.conjugate()

            else:
                mult = backend.to_device(self.mult, device)
                if mult.dtype != input.dtype:
                    mult = mult.astype(input.dtype)

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


class Interpolate(Linop):
    """Interpolation linear operator.

    Args:
        ishape (tuple of ints): Input shape = batch_shape + grd_shape
        coord (array): Coordinates, values from - nx / 2 to nx / 2 - 1.
                ndim can only be 1, 2 or 3, of shape pts_shape + [ndim]
        width (float): Width of interp. kernel in grid size.
        kernel (array): Look-up kernel of kernel K, from K[0] to K[width].
        scale (float): Scaling of coordinates.
        shift (float): Shifting of coordinates.

    """

    def __init__(self, ishape, coord, width, kernel, scale=1, shift=0):
        ndim = coord.shape[-1]
        oshape = list(ishape[:-ndim]) + list(coord.shape[:-1])

        self.coord = coord
        self.width = width
        self.kernel = kernel
        self.shift = shift
        self.scale = scale

        super().__init__(oshape, ishape)

    def _apply(self, input):

        device = backend.get_device(input)
        coord = backend.to_device(self.coord, device)
        kernel = backend.to_device(self.kernel, device)
        shift = backend.to_device(self.shift, device)

        with device:
            return interp.interpolate(input, self.width, kernel,
                                      coord * self.scale + shift)

    def _adjoint_linop(self):
        return Gridding(self.ishape, self.coord, self.width, self.kernel,
                        scale=self.scale, shift=self.shift)


class Gridding(Linop):
    """Gridding linear operator.

    Args:
        oshape (tuple of ints): Output shape = batch_shape + pts_shape
        ishape (tuple of ints): Input shape = batch_shape + grd_shape
        coord (array): Coordinates, values from - nx / 2 to nx / 2 - 1.
                ndim can only be 1, 2 or 3. of shape pts_shape + [ndim]
        width (float): Width of interp. kernel in grid size
        kernel (array): Llook-up kernel of kernel K, from K[0] to K[width]
            scale (float): Scaling of coordinates.
            shift (float): Shifting of coordinates.

    """

    def __init__(self, oshape, coord, width, kernel, scale=1, shift=0):
        ndim = coord.shape[-1]
        ishape = list(oshape[:-ndim]) + list(coord.shape[:-1])

        self.coord = coord
        self.width = width
        self.kernel = kernel
        self.shift = shift
        self.scale = scale

        super().__init__(oshape, ishape)

    def _apply(self, input):
        device = backend.get_device(input)
        coord = backend.to_device(self.coord, device)
        kernel = backend.to_device(self.kernel, device)
        shift = backend.to_device(self.shift, device)

        with device:
            return interp.gridding(input, self.oshape, self.width, kernel,
                                   coord * self.scale + shift)

    def _adjoint_linop(self):
        return Interpolate(self.oshape, self.coord, self.width, self.kernel,
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

        return util.resize(input, self.oshape,
                           ishift=self.ishift, oshift=self.oshift)

    def _adjoint_linop(self):

        return Resize(self.ishape, self.oshape,
                      ishift=self.oshift, oshift=self.ishift)


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
        return util.upsample(input, self.oshape,
                             self.factors, shift=self.shift)

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
            self.ishape,
            axes=self.axes,
            wave_name=self.wave_name,
            level=self.level)


class InverseWavelet(Linop):
    """Inverse wavelet transform linear operator.

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
        return Wavelet(self.oshape, axes=self.axes,
                       wave_name=self.wave_name, level=self.level)


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
        device = backend.get_device(input)
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

        device = backend.get_device(input)
        xp = device.xp
        with device:
            return xp.tile(input.reshape(self.expanded_ishape), self.reps)

    def _adjoint_linop(self):

        return Sum(self.oshape, self.axes)


class ArrayToBlocks(Linop):
    """Extract blocks from array.

    Args:
        ishape (tuple of ints): Input shape.
        blk_shape (tuple of ints): Block shape.
        blk_shape (tuple of ints): Block strides.

    """

    def __init__(self, ishape, blk_shape, blk_strides):
        self.blk_shape = blk_shape
        self.blk_strides = blk_strides
        num_blks = [(i - b + s) // s for i, b,
                    s in zip(ishape, blk_shape, blk_strides)]
        oshape = num_blks + list(blk_shape)

        super().__init__(oshape, ishape)

    def _apply(self, input):
        return block.array_to_blocks(input, self.blk_shape, self.blk_strides)

    def _adjoint_linop(self):
        return BlocksToArray(self.ishape, self.blk_shape, self.blk_strides)


class BlocksToArray(Linop):
    """Average blocks to array.

    Args:
        oshape (tuple of ints): Output shape.
        blk_shape (tuple of ints): Block shape.
        blk_shape (tuple of ints): Block strides.

    """

    def __init__(self, oshape, blk_shape, blk_strides):
        self.blk_shape = blk_shape
        self.blk_strides = blk_strides
        num_blks = [(i - b + s) // s for i, b,
                    s in zip(oshape, blk_shape, blk_strides)]
        ishape = num_blks + list(blk_shape)

        super().__init__(oshape, ishape)

    def _apply(self, input):
        return block.blocks_to_array(
            input, self.oshape, self.blk_shape, self.blk_strides)

    def _adjoint_linop(self):
        return ArrayToBlocks(self.oshape, self.blk_shape, self.blk_strides)


def Gradient(ishape, axes=None):
    import warnings

    warnings.warn('Gradient Linop is renamed to FiniteDifference, '
                  'Please switch to use FiniteDifference.',
                  category=DeprecationWarning)

    return FiniteDifference(ishape, axes=axes)


def FiniteDifference(ishape, axes=None):
    """Linear operator that computes finite difference gradient.

    Args:
       ishape (tuple of ints): Input shape.

    """
    I = Identity(ishape)
    axes = util._normalize_axes(axes, len(ishape))
    ndim = len(ishape)
    linops = []
    for i in range(ndim):
        D = I - Circshift(ishape, [0] * i + [1] + [0] * (ndim - i - 1))
        R = Reshape([1] + list(ishape), ishape)
        linops.append(R * D)

    G = Vstack(linops, axis=0)

    return G


class NUFFT(Linop):
    """NUFFT linear operator.

    Args:
        ishape (tuple of int): Input shape.
        coord (array): Coordinates, with values [-ishape / 2, ishape / 2]
        oversamp (float): Oversampling factor.
        width (float): Kernel width.
        n (int): Kernel sampling number.

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

        return fourier.nufft(
            input,
            self.coord,
            oversamp=self.oversamp,
            width=self.width,
            n=self.n)

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
        n (int): Kernel sampling number.

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
        return fourier.nufft_adjoint(
            input,
            self.coord,
            self.oshape,
            oversamp=self.oversamp,
            width=self.width,
            n=self.n)

    def _adjoint_linop(self):

        return NUFFT(self.oshape, self.coord,
                     oversamp=self.oversamp, width=self.width, n=self.n)


class ConvolveData(Linop):
    r"""Convolution operator for data arrays.

    Args:
        data_shape (tuple of ints): data array shape:
            :math:`[\ldots, m_1, \ldots, m_D]` if multi_channel is False,
            :math:`[\ldots, c_i, m_1, \ldots, m_D]` otherwise.
        filt (array): filter array of shape:
            :math:`[n_1, \ldots, n_D]` if multi_channel is False
            :math:`[c_o, c_i, n_1, \ldots, n_D]` otherwise.
        mode (str): {'full', 'valid'}.
        strides (None or tuple of ints): convolution strides of length D.
        multi_channel (bool): specify if input/output has multiple channels.

    """
    def __init__(self, data_shape, filt, mode='full', strides=None,
                 multi_channel=False):
        self.filt = filt
        self.mode = mode
        self.strides = strides
        self.multi_channel = multi_channel

        D, b, B, m, n, s, c_i, c_o, p = conv._get_convolve_params(
            data_shape, filt.shape,
            mode, strides, multi_channel)

        if multi_channel:
            output_shape = b + (c_o, ) + p
        else:
            output_shape = b + p

        super().__init__(output_shape, data_shape)

    def _apply(self, input):
        return conv.convolve(input, self.filt, mode=self.mode,
                             strides=self.strides,
                             multi_channel=self.multi_channel)

    def _adjoint_linop(self):
        return ConvolveDataAdjoint(
            self.ishape, self.filt,
            mode=self.mode, strides=self.strides,
            multi_channel=self.multi_channel)


class ConvolveDataAdjoint(Linop):
    r"""Adjoint convolution operator for data arrays.

    Args:
        data_shape (tuple of ints): data array shape:
            :math:`[\ldots, m_1, \ldots, m_D]` if multi_channel is False,
            :math:`[\ldots, c_i, m_1, \ldots, m_D]` otherwise.
        filt (array): filter array of shape:
            :math:`[n_1, \ldots, n_D]` if multi_channel is False
            :math:`[c_o, c_i, n_1, \ldots, n_D]` otherwise.
        mode (str): {'full', 'valid'}.
        strides (None or tuple of ints): convolution strides of length D.
        multi_channel (bool): specify if input/output has multiple channels.

    """
    def __init__(self, data_shape, filt,
                 mode='full', strides=None, multi_channel=False):
        self.filt = filt
        self.mode = mode
        self.strides = strides
        self.multi_channel = multi_channel

        D, b, B, m, n, s, c_i, c_o, p = conv._get_convolve_params(
            data_shape, filt.shape,
            mode, strides, multi_channel)

        if multi_channel:
            output_shape = b + (c_o, ) + p
        else:
            output_shape = b + p

        super().__init__(data_shape, output_shape)

    def _apply(self, input):
        return conv.convolve_data_adjoint(
            input, self.filt, self.oshape,
            mode=self.mode,
            strides=self.strides,
            multi_channel=self.multi_channel)

    def _adjoint_linop(self):
        return ConvolveData(self.oshape, self.filt,
                            mode=self.mode, strides=self.strides,
                            multi_channel=self.multi_channel)


class ConvolveFilter(Linop):
    r"""Convolution operator for filter arrays.

    Args:
        filt_shape (tuple of ints): filter array shape:
            :math:`[n_1, \ldots, n_D]` if multi_channel is False
            :math:`[c_o, c_i, n_1, \ldots, n_D]` otherwise.
        data (array): data array of shape:
            :math:`[\ldots, m_1, \ldots, m_D]` if multi_channel is False,
            :math:`[\ldots, c_i, m_1, \ldots, m_D]` otherwise.
        mode (str): {'full', 'valid'}.
        strides (None or tuple of ints): convolution strides of length D.
        multi_channel (bool): specify if input/output has multiple channels.

    """
    def __init__(self, filt_shape, data,
                 mode='full', strides=None,
                 multi_channel=False):
        self.data = data
        self.mode = mode
        self.strides = strides
        self.multi_channel = multi_channel

        D, b, B, m, n, s, c_i, c_o, p = conv._get_convolve_params(
            data.shape, filt_shape,
            mode, strides, multi_channel)

        if multi_channel:
            output_shape = b + (c_o, ) + p
        else:
            output_shape = b + p

        super().__init__(output_shape, filt_shape)

    def _apply(self, input):
        data = backend.to_device(self.data, backend.get_device(input))
        return conv.convolve(data, input,
                             mode=self.mode, strides=self.strides,
                             multi_channel=self.multi_channel)

    def _adjoint_linop(self):
        return ConvolveFilterAdjoint(
            self.ishape, self.data,
            mode=self.mode, strides=self.strides,
            multi_channel=self.multi_channel)


class ConvolveFilterAdjoint(Linop):
    r"""Adjoint convolution operator for filter arrays.

    Args:
        filt_shape (tuple of ints): filter array shape:
            :math:`[n_1, \ldots, n_D]` if multi_channel is False
            :math:`[c_o, c_i, n_1, \ldots, n_D]` otherwise.
        data (array): data array of shape:
            :math:`[\ldots, m_1, \ldots, m_D]` if multi_channel is False,
            :math:`[\ldots, c_i, m_1, \ldots, m_D]` otherwise.
        mode (str): {'full', 'valid'}.
        strides (None or tuple of ints): convolution strides of length D.
        multi_channel (bool): specify if input/output has multiple channels.

    """
    def __init__(self, filt_shape, data,
                 mode='full', strides=None,
                 multi_channel=False):
        self.data = data
        self.mode = mode
        self.strides = strides
        self.multi_channel = multi_channel

        D, b, B, m, n, s, c_i, c_o, p = conv._get_convolve_params(
            data.shape, filt_shape,
            mode, strides, multi_channel)

        if multi_channel:
            output_shape = b + (c_o, ) + p
        else:
            output_shape = b + p

        super().__init__(filt_shape, output_shape)

    def _apply(self, input):
        return conv.convolve_filter_adjoint(
            input, self.data, self.oshape,
            mode=self.mode, strides=self.strides,
            multi_channel=self.multi_channel)

    def _adjoint_linop(self):
        return ConvolveFilter(self.oshape, self.data,
                              mode=self.mode, strides=self.strides,
                              multi_channel=self.multi_channel)
