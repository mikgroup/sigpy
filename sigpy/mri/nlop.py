# -*- coding: utf-8 -*-
"""MRI non-linear operators.

This module contains these non-linear operators:

    * Nlinv,
        joint coil sensitivity maps and image content.

    * Diffusion,
        exponential diffusion modelling and parallel imaging sampling.

Author: Zhengguo Tan <zhengguo.tan@gmail.com>
"""
import sigpy as sp


class Nlinv(sp.nlop.Nlop):
    """
    Construction of the non-linear parallel imaging (nlinv) operator.

    Given the unknown x = (rho, c_1, ..., c_N)^T
    , where
        rho: image content, and
        c_1, ..., c_N: N coil sensitivity maps,

    the forward operation is,

        F(x) = ( ..., FT{rho * c_n}, ... )^T

    , where
        n in [1, N], and
        FT is either masked FFT or NUFFT.

    Args:
        image_shape (tuple): shape of image.
        coil_shape (tuple): shape of coils.
        coord (None or array): coordinates, i.e. trajectories
        coil (None or array): coil sensitivity maps.
        W_coil (boolean): apply Sobolev weight on coil or not.
        upd_coil (boolean): update coil sensitivity maps or not.

    Reference:
        Bauer F., Kannengiesser S. (2007).
        An alternative approach to the image reconstruction
        for parallel data acquisition in MRI.
        Math. Methods Appl. Sci., 30, 1437-1451.

        Uecker M., Hohage T., Block K. T., Frahm J. (2008).
        Image reconstruction by regularized nonlinear inversion -
        joint estimation of coil sensitivities and image content.
        Magn. Reson. Med., 60, 674-682.
    """

    def __init__(self, image_shape, coil_shape,
                 coord=None, coil=None,
                 W_coil=True, upd_coil=True,
                 repr_str=None):
        self.image_shape = image_shape
        self.coil_shape = coil_shape

        ishape = self._get_xshape()

        self.coord = coord
        self.coil = coil
        self.upd_coil = upd_coil

        # Sobolev linear operator on coils
        if W_coil:
            self.W = sp.linop.Sobolev(self.coil_shape)
        else:
            self.W = sp.linop.Identity(self.coil_shape)

        # FFT or NUFFT operator
        x_ndim = len(ishape)
        if coord is None:
            self.F = sp.linop.FFT(self.coil_shape, axes=range(-x_ndim+1, 0))
        else:
            self.F = sp.linop.NUFFT(self.coil_shape, coord)

        oshape = self.F.oshape

        super().__init__(oshape, ishape, repr_str)

    def _get_xshape(self):

        image_ndim = len(self.image_shape)

        if image_ndim == 2:
            num_coilimg = 1 + self.coil_shape[0]
        else:
            num_coilimg = self.image_shape[0] + self.coil_shape[0]

        xshape = []   # empty list
        xshape.append(num_coilimg)

        return xshape + list(self.image_shape[-2:])

    def _forward(self, input):
        with sp.backend.get_device(input):

            # store the current estimate into class
            self.x = input

            image = self.x[0, :, :]   # extract image
            coil_ksp = self.x[1:, :, :]   # extract coils
            coil_img = self.W * coil_ksp

            return self.F(image * coil_img)

    def _get_Jacobian(self, x):
        return None

    def _derivative(self, x, dx):
        device = sp.backend.get_device(dx)

        self.x = x

        with device:
            image = self.x[0, :, :]
            coil_ksp = self.x[1:, :, :]
            coil_img = self.W * coil_ksp

            dimage = dx[0, :, :]
            dcoil_ksp = dx[1:, :, :]
            dcoil_img = self.W * dcoil_ksp

            return self.F * (dimage * coil_img + image * dcoil_img)

    def _adjoint(self, x, dy):
        device = sp.backend.get_device(dy)
        xp = device.xp

        self.x = x

        output = xp.zeros_like(self.x)

        with device:
            image = self.x[0, :, :]
            coil_ksp = self.x[1:, :, :]
            coil_img = self.W * coil_ksp

            dcoilimg = self.F.H * dy

            output[0, :, :] = xp.sum(xp.conj(coil_img) * dcoilimg, axis=0)

            if self.upd_coil:
                output[1:, :, :] = self.W.H(xp.conj(image) * dcoilimg)

            return output
