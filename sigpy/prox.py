import numpy as np
from sigpy import config, util, thresh

if config.cupy_enabled:
    import cupy as cp


class Prox(object):
    '''
    Abstraction for proximal operator
    '''

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
    '''
    Convex conjugate of proximal operator
    '''

    def __init__(self, prox):

        self.prox = prox
        super().__init__(prox.shape)

    def _prox(self, alpha, input):

        with util.get_device(input):
            return input - alpha * self.prox(1 / alpha, input / alpha)


class NoOp(Prox):
    '''
    Proximal operator for empty function.

    Parameters
    ----------
    shape : tuple of int, input shape
    '''

    def __init__(self, shape):
        super().__init__(shape)

    def _prox(self, alpha, input):
        return input


class Stack(Prox):
    '''Stack outputs of proximal operators.
    Parameters
    ----------
    proxs : arrays of proxs with same shape
    '''

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
    '''
    Proximal operator for lamda / 2 || x - y ||_2^2.

    Parameters
    ----------
    shape : tuple of int, input shape
    lamda : float, regularization parameter

    Returns:
    --------
    prox(alpha, x) = (x + lamda * alpha * y) / (1 + lamda * alpha)
    '''

    def __init__(self, shape, lamda, y=0):
        self.lamda = lamda
        self.y = y

        super().__init__(shape)

    def _prox(self, alpha, input):

        with util.get_device(input):
            return ((input + self.lamda * alpha * self.y) /
                    (1 + self.lamda * alpha))


class L2Proj(Prox):
    '''
    Proximal operator for I{ ||x - y||_2 < e}.

    Parameters
    ----------
    shape : tuple of int, input shape
    epsilon : float, regularization parameter
    y : optional bias term
    '''

    def __init__(self, shape, epsilon, y=0, axes=None):
        self.epsilon = epsilon
        self.y = y
        self.axes = axes

        super().__init__(shape)

    def _prox(self, alpha, input):

        with util.get_device(input):
            return thresh.l2_proj(self.epsilon, input - self.y, self.axes) + self.y


class L1Reg(Prox):
    '''
    Proximal operator for lamda * || W x ||_1. Soft threshold input.

    Parameters
    ----------
    shape : tuple of int, input shape
    lamda : float, regularization parameter
    transform : Linop x -> y, unitary transform
    '''

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
    '''
    Proximal operator for 1{ ||x||_0 < k}.

    Parameters
    ----------
    shape : tuple of int, input shape
    k : int, sparsity
    '''

    def __init__(self, shape, k, axes=None):

        self.k = k
        self.axes = axes

        super().__init__(shape)

    def _prox(self, alpha, input):

        return thresh.l0_proj(self.k, input, axes=self.axes)


class L1Proj(Prox):
    '''
    Proximal operator for 1{ ||x||_1 < epsilon}.

    Parameters
    ----------
    shape : tuple of int, input shape
    epsilon : float, regularization parameter
    '''

    def __init__(self, shape, epsilon):

        self.epsilon = epsilon

        super().__init__(shape)

    def _prox(self, alpha, input):

        return thresh.l1_proj(self.epsilon, input)


class L1L2Reg(Prox):
    '''
    Proximal operator for lamda * sum_j ||x_j||_1^2
    '''

    def __init__(self, shape, lamda, axes=None):

        self.lamda = lamda
        self.axes = axes

        super().__init__(shape)

    def _prox(self, alpha, input):

        return thresh.elitist_thresh(self.lamda * alpha, input, axes=self.axes)
