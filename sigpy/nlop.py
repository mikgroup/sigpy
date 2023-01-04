# -*- coding: utf-8 -*-
"""This module contains an abstraction class Nlop for non-linear operators,
and provides commonly used non-linear operator, including exponential.

Author: Zhengguo Tan <zhengguo.tan@gmail.com>
"""
import numpy as np

from sigpy import backend, linop


class Nlop():
    """Abstraction for non-linear operator.

    Given a nlop A, and an appropriately shaped input x,
    the following are valid operations:

       >>> y = A.forward(x)         # apply forward operation A on x
       >>> y = A(x)
       >>> y = A * x
       >>> y = A.derivative(x, dx)  # apply Jacobian (derivative) of A on x
       >>> x = A.adjoint(x, dy)     # apply adjoint of derivative on y

    Args:
        oshape (tuple): operator output shape.
        ishape (tuple): operator input shape.
        scale (array or None): Scaling for the input

    Attributes:
        oshape: output shape.
        ishape: input shape.
        forward: apply forward operator.
        get_Jacobian: get the operator's Jacobian matrix.
        derivative: apply derivative operator.
        adjoint: apply adjoint of derivative operator.

    """
    def __init__(self, oshape, ishape,
                 scale=None, repr_str=None):
        self.oshape = oshape
        self.ishape = ishape
        self.scale = scale

        linop._check_shape_positive(oshape)
        linop._check_shape_positive(ishape)

        if repr_str is None:
            self.repr_str = self.__class__.__name__
        else:
            self.repr_str = repr_str

    def _check_ishape(self, input):
        for i1, i2 in zip(input.shape, self.ishape):
            if i2 != -1 and i1 != i2:
                raise ValueError(
                    'input shape mismatch for {s}, got {input_shape}'.format(
                        s=self, input_shape=input.shape))

    def _check_oshape(self, output):
        for o1, o2 in zip(output.shape, self.oshape):
            if o2 != -1 and o1 != o2:
                raise ValueError(
                    'output shape mismatch for {s}, got {output_shape}'.format(
                        s=self, output_shape=output.shape))

    def _get_Jacobian(self, x):
        raise NotImplementedError

    def get_Jacobian(self, x):
        """Compute the Jacobian matrix of non-linear operator.
        """
        try:
            self._check_ishape(x)
            output = self._get_Jacobian(x)
        except Exception as e:
            raise RuntimeError('Exceptions from {}.'.format(self)) from e

        return output

    def _forward(self, input):
        raise NotImplementedError

    def forward(self, input):
        """Apply non-linear forward operation on input.
        """
        try:
            self._check_ishape(input)
            output = self._forward(input)
            self._check_oshape(output)
        except Exception as e:
            raise RuntimeError('Exceptions from {}.'.format(self)) from e

        return output

    def _derivative(self, x, dx):
        raise NotImplementedError

    def derivative(self, x, dx):
        """Apply derivative operation on input.
        """
        try:
            self._check_ishape(x)
            self._check_ishape(dx)
            output = self._derivative(x, dx)
            self._check_oshape(output)
        except Exception as e:
            raise RuntimeError('Exceptions from {}.'.format(self)) from e

        return output

    def _adjoint(self, x, dy):
        raise NotImplementedError

    def adjoint(self, x, dy):
        """Apply adjoint of derivative operation on input.
        """
        try:
            self._check_oshape(dy)
            self._check_ishape(x)
            output = self._adjoint(x, dy)
            self._check_ishape(output)
        except Exception as e:
            raise RuntimeError('Exceptions from {}.'.format(self)) from e

        return output

    def __call__(self, input):
        return self.__mul__(input)

    def __mul__(self, input):
        if isinstance(input, linop.Linop):
            return Compose([self, input])
        elif isinstance(input, backend.get_array_module(input).ndarray):
            return self.forward(input)

        return NotImplemented


class Exponential(Nlop):
    """
    Construction of the non-linear exponential operator.

    Given the unknown x = (b, a, R)^T, where
        b: bias array, scalar, or None,
        a: encoding array or scalar, and
        R: relaxation rate,
    and the encoding array encode,
    the forware operation is
        F(x) = b + a * exp(encode * R).

    Args:
        ishape (tuple): input shape
        encode (array): echo times or B matrix
        bias (boolean): have bias (b) in the model or not
    """
    def __init__(self, ishape, encode,
                 bias=False, const_a=False,
                 scale=None, rvc=False, repr_str=None):
        image_shape = ishape[1:]
        num_param = ishape[0]

        num_encode, num_relax = encode.shape

        assert num_relax == num_param - 2 if bias else num_param - 1

        self.encode = encode
        self.bias = bias
        self.const_a = const_a
        self.rvc = rvc

        if scale is None:
            scale = np.ones([num_param] + [1] * len(image_shape))
        else:
            assert scale.shape[0] == num_param

        oshape = [num_encode] + list(image_shape)

        super().__init__(oshape, ishape, scale=scale, repr_str=repr_str)

    def _check_coil_image_shape(self, coil_shape, image_shape):
        for i1, i2 in zip(coil_shape, image_shape):
            if i1 != i2:
                raise ValueError('coils and image have different shape.')

    def get_params(self, x):
        """Split the unknown x into (b, a, R)
        """
        xp = backend.get_device(x).xp

        with backend.get_device(x):
            xscale = self.scale * x
            ind = 0

            if self.bias is True:
                b = xscale[ind, ...]
                ind += 1
            else:
                b = None

            if self.const_a is True:
                a = xp.ones_like(xscale, shape=xscale.shape[1:])
            else:
                a = xscale[ind, ...]
                ind += 1

            R = xscale[ind:, ...]

            return b, a, R

    def _forward(self, input):
        device = backend.get_device(input)
        xp = device.xp

        with device:
            self.encode = backend.to_device(self.encode, device=device)
            self.scale = backend.to_device(self.scale, device=device)

            self.x = input

            b, a, R = self.get_params(self.x)

            Rr = xp.reshape(R, (R.shape[0], -1))
            output = xp.exp(xp.matmul(self.encode, Rr))
            output = xp.reshape(output, self.oshape)

            output *= a

            if self.bias is True:
                output += b

            return output

    def _get_Jacobian(self, x):
        # For the computation of Jacobian, self.x must exist.
        self.x = x
        device = backend.get_device(self.x)
        xp = device.xp

        image_shape = self.x.shape[1:]

        with device:
            self.encode = backend.to_device(self.encode, device=device)

            b, a, R = self.get_params(self.x)

            Jshape = []   # Nr. of encoding + xshape
            Jshape.append(self.encode.shape[0])
            Jshape += list(self.x.shape)

            output = xp.zeros_like(self.x, shape=Jshape)

            ind = 0

            # Jacobian for b
            if self.bias is True:
                output[:, ind, ...] = xp.ones_like(
                    self.x, shape=image_shape)
                ind += 1

            Rr = xp.reshape(R, (R.shape[0], -1))
            z = xp.exp(xp.matmul(self.encode, Rr))
            z = xp.reshape(z, self.oshape)

            # Jacobian for a
            if self.const_a is False:
                output[:, ind, ...] = z
                ind += 1

            # Jacobian for R
            encode = xp.reshape(self.encode, list(self.encode.shape)
                                + [1] * len(image_shape))
            Z = xp.reshape(a * z, [z.shape[0]] + [1] + list(image_shape))
            output[:, ind:, ...] = encode * Z

            return output

    def _derivative(self, x, dx):
        device = backend.get_device(dx)
        xp = device.xp

        with device:
            self.Jacobian = self._get_Jacobian(x)
            return xp.sum(self.Jacobian * dx, axis=1)

    def _adjoint(self, x, dy):
        device = backend.get_device(dy)
        xp = device.xp

        with device:
            self.Jacobian = self._get_Jacobian(x)
            JH = xp.conjugate(xp.moveaxis(self.Jacobian, 0, 1))
            dx = xp.sum(JH * dy, axis=1)

            if self.rvc:
                dx = dx.real + 0. * 1j

            return dx


def _check_compose_nlops(ops):
    for op1, op2 in zip(ops[:-1], ops[1:]):
        if (op1.ishape != op2.oshape):
            raise ValueError('cannot compose {op1} and {op2}.'.format(
                op1=op1, op2=op2))


def _combine_compose_nlops(ops):
    combined_nlops = []
    for op in ops:
        if isinstance(op, Compose):
            combined_nlops += op.ops
        else:
            combined_nlops.append(op)

    return combined_nlops


class Compose(Nlop):
    """Composition of non-linear operators.

    Args:
        ops (list of operators): (linops and/or nlops) to be composed.

    Returns:
        Nlop: op[0] * op[1] * ... * op[n - 1]
    """
    def __init__(self, ops):
        _check_compose_nlops(ops)
        self.ops = _combine_compose_nlops(ops)

        super().__init__(
            self.ops[0].oshape, self.ops[-1].ishape,
            repr_str=' * '.join([op.repr_str for op in ops]))

    def _forward(self, input):
        output = input
        for op in self.ops[::-1]:
            output = op(output)
        return output

    def _get_Jacobian(self, input):
        output = input
        for op in self.ops[::-1]:
            if isinstance(op, linop.Linop):
                output = op(output)
            elif isinstance(op, Nlop):
                output = op.get_Jacobian(output)

        return output

    def _derivative(self, x, dx):
        output = dx
        for op in self.ops[::-1]:
            if isinstance(op, linop.Linop):
                output = op(output)
            elif isinstance(op, Nlop):
                output = op.derivative(x, output)

        return output

    def _adjoint(self, x, dy):
        output = dy
        for op in self.ops[::1]:
            if isinstance(op, linop.Linop):
                output = op.H(output)
            elif isinstance(op, Nlop):
                output = op.adjoint(x, output)

        return output
