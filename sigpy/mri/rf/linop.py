# -*- coding: utf-8 -*-
"""MRI RF-specific linear operators.

"""
import sigpy as sp
import numpy as np


def pTxSpatialA(sens, coord, dt, img_shape, B0=None, comm=None):
    """Explicit spatial-domain pulse design linear operator.
    Linear operator relates rf pulses to desired magnetization.
    Linear operator dimensions will be Ns * Nt

    Args:
        sens (array): sensitivity maps of length = number of channels. [Nc dim dim].
        B0 (array): 2D array, B0 inhomogeneity map
        coord (None or array): coordinates. [Nt 2]
        dt (float): hardware sampling dt
        img_shape (None or tuple): image shape.
        B0 (array): 2D array, B0 inhomogeneity map
        comm (None or `sigpy.Communicator`): communicator
            for distributed computing.

    """
    Nc = sens[0]
    T = dt * coord.shape[0]  # duration of pulse, in seconds
    t = np.expand_dims(np.linspace(0, T, coord.shape[0]), axis=1)  # create time vector

    x, y = np.ogrid[-img_shape[0] / 2: img_shape[0] - img_shape[0] / 2,
           -img_shape[1] / 2: img_shape[1] - img_shape[1] / 2]

    #make x and y into proper grid layout
    x = x*np.ones(img_shape)
    y = y*np.ones(img_shape)

    # create explicit Ns * Nt system matrix with and without B0 inhomogeneity correction
    if B0 is None:
        AExplicit = np.exp(1j * (np.outer(np.concatenate(x), coord[:, 0]) + np.outer(np.concatenate(y), coord[:, 1])))
    else:
        AExplicit = np.exp(1j * 2 * np.pi * np.transpose(np.concatenate(B0) * (t-T))) * \
            np.exp(1j * (np.outer(np.concatenate(x), coord[:, 0]) + np.outer(np.concatenate(y), coord[:, 1])))

    AFullExplicit = np.empty(AExplicit.shape)

    # add sensitivities
    for ii in range(Nc):
        tmp = np.concatenate(sens[ii, :, :])
        D = np.transpose(np.tile(tmp, [coord.shape[0], 1]))
        AFullExplicit = np.concatenate((AFullExplicit, D*AExplicit), axis=1)

    AFullExplicit = AFullExplicit[:,coord.shape[0]:] # remove 1st empty AExplicit entries
    A = sp.linop.MatMul((coord.shape[0]*Nc, 1), AFullExplicit)

    if comm is not None:
        C = sp.linop.AllReduceAdjoint(img_shape, comm, in_place=True)
        A = A * C

    A.repr_str = 'Sense'
    return A
