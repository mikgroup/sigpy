# -*- coding: utf-8 -*-
"""MRI pulse-design-specific linear operators.
"""
import sigpy as sp
from sigpy import backend


def PtxSpatialExplicit(sens, coord, dt, img_shape, B0=None, ret_array=False):
    """Explicit spatial-domain pulse design linear operator.
    Linear operator relates rf pulses to desired magnetization.
    Linear operator dimensions will be Ns * Nt

    Args:
        sens (array): sensitivity maps [Nc dim dim]
        B0 (array): 2D array, B0 inhomogeneity map
        coord (None or array): coordinates. [Nt 2]
        dt (float): hardware sampling dt
        img_shape (None or tuple): image shape.
        B0 (array): 2D array, B0 inhomogeneity map
        ret_array (bool): if true, return explicit numpy array.
            Else return linop.


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
        Nc = sens.shape[0]
        T = dt * coord.shape[0]  # duration of pulse, in seconds
        # create time vector
        t = xp.expand_dims(xp.linspace(0, T, coord.shape[0]), axis=1)

        # row-major order
        # x L to R, y T to B
        x_ = xp.linspace(-img_shape[0] / 2,
                         img_shape[0] - img_shape[0] / 2, img_shape[0])
        y_ = xp.linspace(img_shape[1] / 2,
                         -(img_shape[1] - img_shape[1] / 2), img_shape[1])
        if three_d:
            # z B to T
            z_ = xp.linspace(-img_shape[2] / 2,
                             img_shape[2] - img_shape[2] / 2, img_shape[2])
            x, y, z = xp.meshgrid(x_, y_, z_, indexing='ij')
        else:
            x, y = xp.meshgrid(x_, y_, indexing='ij')

        # create explicit Ns * Nt system matrix
        if not three_d:
            if B0 is None:
                AExplicit = xp.exp(1j * (xp.outer(x.flatten(), coord[:, 0]) +
                                         xp.outer(y.flatten(), coord[:, 1])))
            else:
                AExplicit = xp.exp(1j * 2 * xp.pi * xp.transpose(B0.flatten()
                                                                 * (t - T)) +
                                   1j * (xp.outer(x.flatten(), coord[:, 0])
                                         + xp.outer(y.flatten(),
                                                    coord[:, 1])))

        else:
            if B0 is None:
                AExplicit = xp.exp(1j * (xp.outer(x.flatten(), coord[:, 0]) +
                                         xp.outer(y.flatten(), coord[:, 1]) +
                                         xp.outer(z.flatten(), coord[:, 2])))
            else:
                AExplicit = xp.exp(1j * 2 * xp.pi * xp.transpose(B0.flatten()
                                                                 * (t - T)) +
                                   1j * (xp.outer(x.flatten(), coord[:, 0])
                                         + xp.outer(y.flatten(), coord[:, 1])
                                         + xp.outer(z.flatten(), coord[:, 2])))
        AFullExplicit = xp.empty(AExplicit.shape)

        # add sensitivities
        for ii in range(Nc):
            if three_d:
                tmp = xp.squeeze(sens[ii, :, :, :]).flatten()
            else:
                tmp = sens[ii,:, :].flatten()
            D = xp.transpose(xp.tile(tmp, [coord.shape[0], 1]))
            AFullExplicit = xp.concatenate((AFullExplicit, D * AExplicit),
                                           axis=1)

        # remove 1st empty AExplicit entries
        AFullExplicit = AFullExplicit[:, coord.shape[0]:]
        A = sp.linop.MatMul((coord.shape[0] * Nc, 1), AFullExplicit)

        # Finally, adjustment of input/output dimensions to be consistent with
        # the existing Sense linop operator. [Nc x Nt] in, [dim x dim] out
        Ro = sp.linop.Reshape(ishape=A.oshape, oshape=sens.shape[1:])
        Ri = sp.linop.Reshape(ishape=(Nc, coord.shape[0]),
                              oshape=(coord.shape[0] * Nc, 1))
        A = Ro * A * Ri

        A.repr_str = 'pTx spatial explicit'

        if ret_array:
            return A.linops[1].mat
        else:
            return A
