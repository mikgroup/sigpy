# -*- coding: utf-8 -*-
"""Functions for simulations.

"""
import numpy as np

__all__ = ['shepp_logan']


def shepp_logan(shape, dtype=np.complex):
    """Generates a Shepp Logan phantom with a given shape and dtype.

    Args:
        shape (tuple of ints): shape, can be of length 2 or 3.
        dtype (Dtype): data type.

    Returns:
        array.

    """
    return phantom(shape, sl_amps, sl_scales, sl_offsets, sl_angles, dtype)


sl_amps = [1, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

sl_scales = [[.6900, .920, .810],  # white big
             [.6624, .874, .780],  # gray big
             [.1100, .310, .220],  # right black
             [.1600, .410, .280],  # left black
             [.2100, .250, .410],  # gray center blob
             [.0460, .046, .050],
             [.0460, .046, .050],
             [.0460, .046, .050],  # left small dot
             [.0230, .023, .020],  # mid small dot
             [.0230, .023, .020]]

sl_offsets = [[0., 0., 0],
              [0., -.0184, 0],
              [.22, 0., 0],
              [-.22, 0., 0],
              [0., .35, -.15],
              [0., .1, .25],
              [0., -.1, .25],
              [-.08, -.605, 0],
              [0., -.606, 0],
              [.06, -.605, 0]]

sl_angles = [[0, 0, 0],
             [0, 0, 0],
             [-18, 0, 10],
             [18, 0, 10],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]


def phantom(shape, amps, scales, offsets, angles, dtype):
    """
    Generate a cube of given shape using a list of ellipsoid
    parameters.
    """

    if len(shape) == 2:

        ndim = 2
        shape = (1, shape[-2], shape[-1])

    elif len(shape) == 3:

        ndim = 3

    else:

        raise ValueError('Incorrect dimension')

    out = np.zeros(shape, dtype=dtype)

    z, y, x = np.mgrid[-(shape[-3] // 2):((shape[-3] + 1) // 2),
                       -(shape[-2] // 2):((shape[-2] + 1) // 2),
                       -(shape[-1] // 2):((shape[-1] + 1) // 2)]

    coords = np.stack((x.ravel() / shape[-1] * 2,
                       y.ravel() / shape[-2] * 2,
                       z.ravel() / shape[-3] * 2))

    for amp, scale, offset, angle in zip(amps, scales, offsets, angles):

        ellipsoid(amp, scale, offset, angle, coords, out)

    if ndim == 2:

        return out[0, :, :]

    else:

        return out


def ellipsoid(amp, scale, offset, angle, coords, out):
    """
    Generate a cube containing an ellipsoid defined by its parameters.
    If out is given, fills the given cube instead of creating a new
    one.
    """
    R = rotation_matrix(angle)
    coords = (np.matmul(R, coords) - np.reshape(offset, (3, 1))) / \
        np.reshape(scale, (3, 1))

    r2 = np.sum(coords ** 2, axis=0).reshape(out.shape)

    out[r2 <= 1] += amp


def rotation_matrix(angle):
    cphi = np.cos(np.radians(angle[0]))
    sphi = np.sin(np.radians(angle[0]))
    ctheta = np.cos(np.radians(angle[1]))
    stheta = np.sin(np.radians(angle[1]))
    cpsi = np.cos(np.radians(angle[2]))
    spsi = np.sin(np.radians(angle[2]))
    alpha = [[cpsi * cphi - ctheta * sphi * spsi,
              cpsi * sphi + ctheta * cphi * spsi,
              spsi * stheta],
             [-spsi * cphi - ctheta * sphi * cpsi,
              -spsi * sphi + ctheta * cphi * cpsi,
              cpsi * stheta],
             [stheta * sphi,
              -stheta * cphi,
              ctheta]]
    return np.array(alpha)
