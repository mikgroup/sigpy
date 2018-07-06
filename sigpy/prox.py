import numpy as np
from sigpy import config, util, thresh

if config.cupy_enabled:
    import cupy as cp


class Prox(object):
    """Abstraction for proximal operator.

    Prox can be called on to a float and an array to perform a proximal operation,
    which is defined as a, x -> argmin_z 1 / 2 || x - z ||_2^2 + a g(z), for some function g.
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
            raise ValueError('input shape mismatch for {s}, got {input_shape}.'.format(
                s=self, input_shape=input.shape))

    def _check_output(self, output):

        if list(output.shape) != self.shape:
            raise ValueError('output shape mismatch, for {s}, got {output_shape}.'.format(
                s=self, output_shape=output.shape))

    def __call__(self, alpha, input):
        self._check_input(input)
        output = self._prox(alpha, input)
        self._check_output(output)
        return output

    def __repr__(self):

        return '<{shape} {repr_str} Prox>.'.format(shape=self.shape, repr_str=self.repr_str)


class Conj(Prox):
    """Returns the convex conjugate of proximal operator.

    The convex conjugate of a proximal operator P is defined as:
    a, x -> x - a * P(1 / a, x / a)
    """

    def __init__(self, prox):

        self.prox = prox
        super().__init__(prox.shape)

    def _prox(self, alpha, input):

        with util.get_device(input):
            return input - alpha * self.prox(1 / alpha, input / alpha)


class NoOp(Prox):
    """Proximal operator for empty function. Equivalant to an identity function.

    Args:
       shape (tuple of ints): Input shape
    """

    def __init__(self, shape):
        super().__init__(shape)

    def _prox(self, alpha, input):
        return input


class Stack(Prox):
    """Stack outputs of proximal operators.

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

        inputs = util.split(input, self.shapes)
        outputs = [prox(alpha, input)
                   for prox, input in zip(self.proxs, inputs)]
        output = util.vec(outputs)

        return output


class L2Reg(Prox):
    """Proximal operator for lamda / 2 || x - y ||_2^2.

    Performs:
    a, x -> (x + lamda * alpha * y) / (1 + lamda * alpha)

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

        with util.get_device(input):
            return (input + self.lamda * alpha * self.y) / (1 + self.lamda * alpha)


class L2Proj(Prox):
    """Proximal operator for I{ ||x - y||_2 < epsilon}.

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

        with util.get_device(input):
            return thresh.l2_proj(self.epsilon, input - self.y, self.axes) + self.y


class L1Reg(Prox):
    """Proximal operator for lamda * || x ||_1. Soft threshold input.

    Args:
        shape (tuple of ints): input shape
        lamda (float): regularization parameter
        transform (Linop): Unitary linear operator.
    """

    def __init__(self, shape, lamda, transform=None):
        self.lamda = lamda
        self.transform = transform

        super().__init__(shape)

    def _prox(self, alpha, input):

        if self.transform is None:
            return thresh.soft_thresh(self.lamda * alpha, input)
        else:
            return self.transform.H(thresh.soft_thresh(self.lamda * alpha,
                                                       self.transform(input)))


class L0Proj(Prox):
    """Proximal operator for 1{ ||x||_0 < k}.

    Args:
        shape (tuple of ints): input shape.
        k (int): Sparsity.
    """

    def __init__(self, shape, k, axes=None):

        self.k = k
        self.axes = axes

        super().__init__(shape)

    def _prox(self, alpha, input):

        return thresh.l0_proj(self.k, input, axes=self.axes)


class L1Proj(Prox):
    """Proximal operator for 1{ ||x||_1 < epsilon}.

    Args:
        shape (tuple of ints): input shape.
        epsilon (float): regularization parameter.
    """

    def __init__(self, shape, epsilon):

        self.epsilon = epsilon

        super().__init__(shape)

    def _prox(self, alpha, input):

        return thresh.l1_proj(self.epsilon, input)


class L1L2Reg(Prox):
    """Proximal operator for lamda * sum_j ||x_j||_1^2

    Args:
        shape (tuple of ints): input shape.
        lamda (float): regularization parameter.
        axes (None or tuple of ints): Axes to perform operation.
    """

    def __init__(self, shape, lamda, axes=None):

        self.lamda = lamda
        self.axes = axes

        super().__init__(shape)

    def _prox(self, alpha, input):

        return thresh.elitist_thresh(self.lamda * alpha, input, axes=self.axes)
