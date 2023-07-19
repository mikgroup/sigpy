# -*- coding: utf-8 -*-
"""This module contains an abstraction class Prox for proximal operators,
and provides commonly used proximal operators, including soft-thresholding,
l1 ball projection, and box constraints.
"""
import numpy as np
import random

from sigpy import backend, util, thresh, linop


class Prox(object):
    r"""Abstraction for proximal operator.

    Prox can be called on a float (:math:`\alpha`) and
    a NumPy or CuPy array (:math:`x`) to perform a proximal operation.

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

    def _check_shape(self, input):
        for i1, i2 in zip(input.shape, self.shape):
            if i2 != -1 and i1 != i2:
                raise ValueError(
                    "shape mismatch for {s}, got {input_shape}.".format(
                        s=self, input_shape=input.shape
                    )
                )

    def __call__(self, alpha, input):
        try:
            self._check_shape(input)
            output = self._prox(alpha, input)
            self._check_shape(output)
        except Exception as e:
            raise RuntimeError("Exceptions from {}.".format(self)) from e

        return output

    def __repr__(self):
        return "<{shape} {repr_str} Prox>.".format(
            shape=self.shape, repr_str=self.repr_str
        )


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
    r"""Proximal operator for empty function.
    Equivalant to an identity function.

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

        assert (self.nops > 0)

        self.proxs = proxs
        self.shapes = [prox.shape for prox in proxs]
        shape = [sum(util.prod(prox.shape) for prox in proxs)]

        super().__init__(shape)

    def _prox(self, alpha, input):
        with backend.get_device(input):
            if np.isscalar(alpha):
                alphas = [alpha] * self.nops
            else:
                alphas = util.split(alpha, self.shapes)

            inputs = util.split(input, self.shapes)
            outputs = [
                prox(alpha, input)
                for prox, input, alpha in zip(self.proxs, inputs, alphas)
            ]
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
        \min_x \frac{1}{2} \|x - y\|_2^2 + \frac{\lambda}{2}\|x-z\|_2^2 + h(x)

    Args:
        shape (tuple of ints): Input shape.
        lamda (float): Regularization parameter.
        y (scalar or array): Bias term.
        proxh (Prox): optional additional proximal operator.

    """

    def __init__(self, shape, lamda, y=None, proxh=None):
        self.lamda = lamda
        self.y = y
        self.proxh = proxh

        super().__init__(shape)

    def _prox(self, alpha, input):
        with backend.get_device(input):
            output = input.copy()
            if self.y is not None:
                output += (self.lamda * alpha) * self.y

            output /= 1 + self.lamda * alpha

            if self.proxh is not None:
                return self.proxh(alpha / (1 + self.lamda * alpha), output)

        return output


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
            return (
                thresh.l2_proj(self.epsilon, input - self.y, self.axes)
                + self.y
            )


class LInfProj(Prox):
    r"""Proximal operator for l-infinity ball projection.

    .. math::
        \min_x \frac{1}{2} \| x - y \|_2^2 + 1\{\| x \|_\infty < \epsilon\}

    Args:
        shape (tuple of ints): Input shape.
        epsilon (float): Regularization parameter.
        y (scalar or array): Bias term.

    """

    def __init__(self, shape, epsilon, bias=None, axes=None):
        self.epsilon = epsilon
        self.bias = bias
        self.axes = axes

        super().__init__(shape)

    def _prox(self, alpha, input):
        with backend.get_device(input):
            return thresh.linf_proj(self.epsilon, input, bias=self.bias)


class PsdProj(Prox):
    r"""Proximal operator for positive semi-definite matrices.

    .. math::
        \min_x \frac{1}{2} \| X - Y \|_2^2 + 1\{\| X \succeq 0\}

    Args:
        shape (tuple of ints): Input shape.

    """

    def _prox(self, alpha, input):
        with backend.get_device(input):
            return thresh.psd_proj(input)


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
        with backend.get_device(input):
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
        with backend.get_device(input):
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


class LLRL1Reg(Prox):
    r"""Local Low Rank L1 Regularization

    Args:
        shape (tuple of int): input shapes.
        lamda (float): regularization parameter.
        randshift (boolean): switch on random shift or not.
        blk_shape (tuple of int): block shape [default: (8, 8)].
        blk_strides (tuple of int): block strides [default: (8, 8)].

    References:
        * Cai JF, Candes EJ, Shen Z.
          A singular value thresholding algorithm
          for matrix completion.
          SIAM J Optim 20:1956-1982 (2010).

        * Trzasko J, Manduca A.
          Local versus global low-rank promotion
          in dynamic MRI series reconstruction.
          Proc. ISMRM 19:4371 (2011).

        * Zhang T, Pauly J, Levesque I.
          Accelerating parameter mapping with a locally low rank constraint.
          Magn Reson Med 73:655-661 (2015).

        * Saucedo A, Lefkimmiatis S, Rangwala N, Sung K.
          Improved computational efficiency of locally low rank
          MRI reconstruction using iterative random patch adjustments.
          IEEE Trans Med Imaging 36:1209-1220 (2017).

        * Hu Y, Wang X, Tian Q, Yang G, Daniel B, McNab J, Hargreaves B.
          Multi-shot diffusion-weighted MRI reconstruction
          with magnitude-based
          spatial-angular locally low-rank regularization (SPA-LLR).
          Magn Reson Med 83:1596-1607 (2020).

    Author:
        Zhengguo Tan <zhengguo.tan@gmail.com>
    """

    def __init__(self, shape, lamda, randshift=True,
                 blk_shape=(8, 8), blk_strides=(8, 8),
                 reg_magnitude=False,
                 verbose=False):
        self.lamda = lamda
        self.randshift = randshift
        self.reg_magnitude = reg_magnitude

        assert len(blk_shape) == len(blk_strides)
        self.blk_shape = blk_shape
        self.blk_strides = blk_strides
        self.verbose = verbose

        # construct forward linops
        self.RandShift = self._linop_randshift(shape, blk_shape, randshift)
        self.A = linop.ArrayToBlocks(shape, blk_shape, blk_strides)
        self.Reshape = self._linop_reshape()

        self.Fwd = self.Reshape * self.A * self.RandShift

        super().__init__(shape)

    def _check_blk(self):
        assert len(self.blk_shape) == len(self.blk_strides)

    def _prox(self, alpha, input):
        device = backend.get_device(input)
        xp = device.xp

        with device:

            if self.reg_magnitude:
                mag = xp.abs(input)
                phs = xp.exp(1j * xp.angle(input))

            else:
                mag = input.copy()
                phs = xp.ones_like(mag)

            output = self.Fwd(mag)

            u, s, vh = xp.linalg.svd(output, full_matrices=False)
            s_thresh = thresh.soft_thresh(self.lamda * alpha, s)

            output = (u * s_thresh[..., None, :]) @ vh

            output = self.Fwd.H(output)

            return output * phs

    def _linop_randshift(self, shape, blk_shape, randshift):

        D = len(blk_shape)

        if randshift is True:
            axes = range(-D, 0)
            shift = [random.randint(0, blk_shape[s]) for s in axes]

            return linop.Circshift(shape, shift, axes)
        else:
            return linop.Identity(shape)

    def _linop_reshape(self):
        D = len(self.blk_shape)

        oshape = [util.prod(self.A.ishape[:-D]),
                  util.prod(self.A.num_blks),
                  util.prod(self.blk_shape)]

        R1 = linop.Reshape(oshape, self.A.oshape)
        R2 = linop.Transpose(R1.oshape, axes=(1, 0, 2))
        return R2 * R1


class SLRMCReg(Prox):
    r"""Structure Low Rank Matrix Completion as Regularization

    Args:
        shape (tuple of int): input shapes.
        lamda (float): regularization parameter.
        blk_shape (tuple of int): block shape [default: (7, 7)].
        blk_strides (tuple of int): block strides [default: (1, 1)].
        thresh (string): thresholding type ['soft' or 'hard'].

    References:
        * Mani M, Jacob M, Kelley D, Magnotta V.
          Multi-shot sensitivity-encoded diffusion data recovery using
          structured low-rank matrix completion (MUSSELS).
          Magn Reson Med 78:494-507 (2017).

        * Bilgic B, Chatnuntawech I, Manhard MK, Tian Q,
          Liao C, Iyer SS, Cauley SF, Huang SY,
          Polimeni JR, Wald LL, Setsompop K.
          Highly accelerated multishot echo planar imaging through
          synergistic machine learning and joint reconstruction.
          Magn Reson Med 82:1343-1358 (2019).

        * Dai E, Mani M, McNab JA.
          Multi-band multi-shot diffusion MRI reconstruction with
          joint usage of structured low-rank constraints
          and explicit phase mapping.
          Magn Reson Med 89:95-111 (2023).

    Author:
        Zhengguo Tan <zhengguo.tan@gmail.com>
    """
    def __init__(self, shape, lamda,
                 blk_shape=(7, 7), blk_strides=(1, 1),
                 thresh='hard', verbose=False):
        self.lamda = lamda

        assert len(blk_shape) == len(blk_strides)
        self.blk_shape = blk_shape
        self.blk_strides = blk_strides
        self.thresh = thresh
        self.verbose = verbose

        # construct forward linops
        self.A = linop.ArrayToBlocks(shape, blk_shape, blk_strides)
        self.Reshape = self._linop_reshape()

        self.Fwd = self.Reshape * self.A

        super().__init__(shape)

    def _prox(self, alpha, input):
        device = backend.get_device(input)
        xp = device.xp

        with device:

            output = self.Fwd(input)

            # SVD
            u, s, vh = xp.linalg.svd(output, full_matrices=False)

            if self.thresh == 'soft':  # soft thresholding

                s_thresh = thresh.soft_thresh(self.lamda * alpha, s)

                output = (u * s_thresh[..., None, :]) @ vh

            else:  # hard thresholding

                keep = int(self.lamda * alpha * len(s))

                if keep >= len(s):
                    keep = len(s)

                if self.verbose:
                    print('>>> shape of the array for SVD: ', output.shape)
                    print('>>> # of singular values kept ' + str(keep)
                          + ' of ' + str(len(s)))

                u_t, s_t, vh_t = u[..., :keep], s[:keep], vh[..., :keep, :]

                output = (u_t * s_t[..., None, :]) @ vh_t

            output = self.Fwd.H(output)

            return output

    def _linop_reshape(self):
        D = len(self.blk_shape)

        oshape1 = [util.prod(self.A.ishape[:-D]),
                   util.prod(self.A.num_blks),
                   util.prod(self.blk_shape)]

        R1 = linop.Reshape(oshape1, self.A.oshape)
        R2 = linop.Transpose(R1.oshape, axes=(0, 2, 1))

        oshape2 = [util.prod(R2.oshape[:-1]),
                   R2.oshape[-1]]

        R3 = linop.Reshape(oshape2, R2.oshape)
        R4 = linop.Transpose(R3.oshape, axes=(1, 0))

        return R4 * R3 * R2 * R1
