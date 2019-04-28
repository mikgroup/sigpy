# -*- coding: utf-8 -*-
"""This module contains an abstraction class Prox for proximal operators,
and provides commonly used proximal operators, including soft-thresholding,
l1 ball projection, and box constraints.
"""
import numpy as np
from sigpy import backend, util, thresh


class Prox(object):
    r"""Abstraction for proximal operator.

    Prox can be called on a float (:math:`\alpha`) and
    an array (:math:`x`) to perform a proximal operation.

    .. math::
        \text{prox}_{\alpha g} (y) =
        \text{argmin}_x \frac{1}{2} || x - y ||_2^2 + \alpha g(x)

    Prox can be stacked, and conjugated.

    Args:
        shape: Input/output shape.
        repr_str (string or None): default: class name.

    Attributes:
        shape: Input/output shape.

    """

    def __init__(self, shape, repr_str=None):
        self.shape = list(shape)

        if repr_str is None:
            self.repr_str = self.__class__.__name__
        else:
            self.repr_str = repr_str

    def _check_input(self, input):

        if list(input.shape) != self.shape:
            raise ValueError(
                'input shape mismatch for {s}, got {input_shape}.'.format(
                    s=self, input_shape=input.shape))

    def _check_output(self, output):

        if list(output.shape) != self.shape:
            raise ValueError(
                'output shape mismatch, for {s}, got {output_shape}.'.format(
                    s=self, output_shape=output.shape))

    def __call__(self, alpha, input):
        self._check_input(input)
        output = self._prox(alpha, input)
        self._check_output(output)
        return output

    def __repr__(self):
        return '<{shape} {repr_str} Prox>.'.format(
            shape=self.shape, repr_str=self.repr_str)


class Conj(Prox):
    r"""Returns the proximal operator for the convex conjugate function.

    The proximal operator of the convex conjugate function
    :math:`g^*` is defined as:

    .. math::
        \text{prox}_{\alpha g^*} (x) =
        x - \alpha \text{prox}_{\frac{1}{\alpha} g} (\frac{1}{\alpha} x)

    """

    def __init__(self, prox):

        self.prox = prox
        super().__init__(prox.shape)

    def _prox(self, alpha, input):

        with backend.get_device(input):
            return input - alpha * self.prox(1 / alpha, input / alpha)


class NoOp(Prox):
    r"""Proximal operator for empty function. Equivalant to an identity function.

    Args:
       shape (tuple of ints): Input shape

    """

    def __init__(self, shape):
        super().__init__(shape)

    def _prox(self, alpha, input):
        return input


class Stack(Prox):
    r"""Stack outputs of proximal operators.

    Args:
       proxs (list of proxs): Prox of the same shape.

    """

    def __init__(self, proxs):
        self.nops = len(proxs)
        assert(self.nops > 0)

        self.proxs = proxs
        self.shapes = [prox.shape for prox in proxs]
        shape = [sum(util.prod(prox.shape) for prox in proxs)]

        super().__init__(shape)

    def _prox(self, alpha, input):
        if np.isscalar(alpha):
            alphas = [alpha] * self.nops
        else:
            alphas = util.split(alpha, self.shapes)

        inputs = util.split(input, self.shapes)
        outputs = [prox(alpha, input)
                   for prox, input, alpha in zip(self.proxs, inputs, alphas)]
        output = util.vec(outputs)

        return output


class UnitaryTransform(Prox):
    r"""Unitary transform input space.

    Returns a proximal operator that does

    .. math::
        A^H \text{prox}_{\alpha g}(A x)

    Args:
        prox (Prox): Proximal operator.
        A (Linop): Unitary linear operator.

    """

    def __init__(self, prox, A):
        self.prox = prox
        self.A = A

        super().__init__(A.ishape)

    def _prox(self, alpha, input):

        return self.A.H(self.prox(alpha, self.A(input)))


class L2Reg(Prox):
    r"""Proximal operator for l2 regularization.

    .. math::
        \min_x \frac{1}{2} \| x - y \|_2^2 + \frac{\lambda}{2} \| x \|_2^2

    Args:
        shape (tuple of ints): Input shape.
        lamda (float): Regularization parameter.
        y (scalar or array): Bias term.

    """

    def __init__(self, shape, lamda, y=0):
        self.lamda = lamda
        self.y = y

        super().__init__(shape)

    def _prox(self, alpha, input):
        with backend.get_device(input):
            return (input + self.lamda * alpha * self.y) / \
                (1 + self.lamda * alpha)


class L2Proj(Prox):
    r"""Proximal operator for l2 norm projection.

    .. math::
        \min_x \frac{1}{2} \| x - y \|_2^2 + 1\{\| x \|_2 < \epsilon\}

    Args:
        shape (tuple of ints): Input shape.
        epsilon (float): Regularization parameter.
        y (scalar or array): Bias term.

    """

    def __init__(self, shape, epsilon, y=0, axes=None):
        self.epsilon = epsilon
        self.y = y
        self.axes = axes

        super().__init__(shape)

    def _prox(self, alpha, input):
        with backend.get_device(input):
            return thresh.l2_proj(
                self.epsilon, input - self.y, self.axes) + self.y


class L1Reg(Prox):
    r"""Proximal operator for l1 regularization.

    .. math::
        \min_x \frac{1}{2} \| x - y \|_2^2 + \lambda \| x \|_1

    Args:
        shape (tuple of ints): input shape
        lamda (float): regularization parameter

    """

    def __init__(self, shape, lamda):
        self.lamda = lamda

        super().__init__(shape)

    def _prox(self, alpha, input):
        return thresh.soft_thresh(self.lamda * alpha, input)


class L1Proj(Prox):
    r"""Proximal operator for l1 norm projection.

    .. math::
        \min_x \frac{1}{2} \| x - y \|_2^2 + 1\{\| x \|_1 < \epsilon\}

    Args:
        shape (tuple of ints): input shape.
        epsilon (float): regularization parameter.

    """

    def __init__(self, shape, epsilon):
        self.epsilon = epsilon

        super().__init__(shape)

    def _prox(self, alpha, input):
        return thresh.l1_proj(self.epsilon, input)


class BoxConstraint(Prox):
    r"""Box constraint proximal operator.

    .. math::
        \min_{x : l \leq x \leq u} \frac{1}{2} \| x - y \|_2^2

    Args:
        shape (tuple of ints): input shape.
        lower (scalar or array): lower limit.
        upper (scalar or array): upper limit.

    """

    def __init__(self, shape, lower, upper):
        self.lower = lower
        self.upper = upper
        super().__init__(shape)

    def _prox(self, alpha, input):
        device = backend.get_device(input)
        xp = device.xp

        with device:
            return xp.clip(input, self.lower, self.upper)
