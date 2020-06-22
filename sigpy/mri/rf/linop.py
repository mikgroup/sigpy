# -*- coding: utf-8 -*-
"""MRI pulse-design-specific linear operators.
"""
import sigpy as sp
from sigpy import backend


def PtxSpatialExplicit(sens, coord, dt, img_shape, b0=None, ret_array=False):
    """Explicit spatial-domain pulse design linear operator.
    Linear operator relates rf pulses to desired magnetization.
    Equivalent matrix has dimensions [Ns Nt].

    Args:
        sens (array): sensitivity maps. [nc dim dim]
        coord (None or array): coordinates. [nt 2]
        dt (float): hardware sampling dt.
        img_shape (None or tuple): image shape.
        b0 (array): 2D array, B0 inhomogeneity map.
        ret_array (bool): if true, return explicit numpy array.
            Else return linop.

    Returns:
        SigPy linop with A.repr_string 'pTx spatial explicit', or numpy array
        if selected with 'ret_array'


    References:
        Grissom, W., Yip, C., Zhang, Z., Stenger, V. A., Fessler, J. A.
        & Noll, D. C.(2006).
        Spatial Domain Method for the Design of RF Pulses in Multicoil
        Parallel Excitation. Magnetic resonance in medicine, 56, 620-629.
    """
    three_d = False
    if len(img_shape) >= 3:
        three_d = True

    device = backend.get_device(sens)
    xp = device.xp
    with device:
        nc = sens.shape[0]
        dur = dt * coord.shape[0]  # duration of pulse, in s

        # create time vector
        t = xp.expand_dims(xp.linspace(0, dur, coord.shape[0]), axis=1)

        # row-major order
        # x L to R, y T to B
        x_ = xp.linspace(-img_shape[0] / 2,
                         img_shape[0] - img_shape[0] / 2, img_shape[0])
        y_ = xp.linspace(img_shape[1] / 2,
                         -(img_shape[1] - img_shape[1] / 2), img_shape[1])
        if three_d:

            z_ = xp.linspace(-img_shape[2] / 2,
                             img_shape[2] - img_shape[2] / 2, img_shape[2])
            x, y, z = xp.meshgrid(x_, y_, z_, indexing='ij')
        else:
            x, y = xp.meshgrid(x_, y_, indexing='ij')

        # create explicit Ns * Nt system matrix, for 3d or 2d problem
        if three_d:
            if b0 is None:
                AExplicit = xp.exp(1j * (xp.outer(x.flatten(), coord[:, 0]) +
                                         xp.outer(y.flatten(), coord[:, 1]) +
                                         xp.outer(z.flatten(), coord[:, 2])))
            else:
                AExplicit = xp.exp(1j * 2 * xp.pi * xp.transpose(b0.flatten()
                                                                 * (t - dur)) +
                                   1j * (xp.outer(x.flatten(), coord[:, 0])
                                         + xp.outer(y.flatten(), coord[:, 1])
                                         + xp.outer(z.flatten(), coord[:, 2])))
        else:
            if b0 is None:
                AExplicit = xp.exp(1j * (xp.outer(x.flatten(), coord[:, 0]) +
                                         xp.outer(y.flatten(), coord[:, 1])))
            else:
                AExplicit = xp.exp(1j * 2 * xp.pi * xp.transpose(b0.flatten()
                                                                 * (t - dur)) +
                                   1j * (xp.outer(x.flatten(), coord[:, 0])
                                         + xp.outer(y.flatten(),
                                                    coord[:, 1])))

        # add sensitivities to system matrix
        AFullExplicit = xp.empty(AExplicit.shape)
        for ii in range(nc):
            if three_d:
                tmp = xp.squeeze(sens[ii, :, :, :]).flatten()
            else:
                tmp = sens[ii, :, :].flatten()
            D = xp.transpose(xp.tile(tmp, [coord.shape[0], 1]))
            AFullExplicit = xp.concatenate((AFullExplicit, D * AExplicit),
                                           axis=1)

        # remove 1st empty AExplicit entries
        AFullExplicit = AFullExplicit[:, coord.shape[0]:]
        A = sp.linop.MatMul((coord.shape[0] * nc, 1), AFullExplicit)

        # Finally, adjustment of input/output dimensions to be consistent with
        # the existing Sense linop operator. [nc x nt] in, [dim x dim] out
        Ro = sp.linop.Reshape(ishape=A.oshape, oshape=sens.shape[1:])
        Ri = sp.linop.Reshape(ishape=(nc, coord.shape[0]),
                              oshape=(coord.shape[0] * nc, 1))
        A = Ro * A * Ri

        A.repr_str = 'pTx spatial explicit'

        # output a sigpy linop or a numpy array
        if ret_array:
            return A.linops[1].mat
        else:
            return A
