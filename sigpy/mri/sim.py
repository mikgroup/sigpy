# -*- coding: utf-8 -*-
"""MRI simulation functions.
"""
import numpy as np


__all__ = ['birdcage_maps']


def birdcage_maps(shape, r=1.5, nzz=8, dtype=np.complex):
    """Simulates birdcage coil sensitivies.

    Args:
        shape (tuple of ints): sensitivity maps shape,
            can be of length 3, and 4.
        r (float): relative radius of birdcage.
        nzz (int): number of coils per ring.
        dtype (Dtype): data type.

    Returns:
        array.
    """

    if len(shape) == 3:

        nc, ny, nx = shape
        c, y, x = np.mgrid[:nc, :ny, :nx]

        coilx = r * np.cos(c * (2 * np.pi / nc))
        coily = r * np.sin(c * (2 * np.pi / nc))
        coil_phs = -c * (2 * np.pi / nc)

        x_co = (x - nx / 2.0) / (nx / 2.0) - coilx
        y_co = (y - ny / 2.0) / (ny / 2.0) - coily
        rr = np.sqrt(x_co ** 2 + y_co ** 2)
        phi = np.arctan2(x_co, -y_co) + coil_phs
        out = (1.0 / rr) * np.exp(1j * phi)

    elif len(shape) == 4:
        nc, nz, ny, nx = shape
        c, z, y, x = np.mgrid[:nc, :nz, :ny, :nx]

        coilx = r * np.cos(c * (2 * np.pi / nzz))
        coily = r * np.sin(c * (2 * np.pi / nzz))
        coilz = np.floor(c / nzz) - 0.5 * (np.ceil(nc / nzz) - 1)
        coil_phs = -(c + np.floor(c / nzz)) * (2 * np.pi / nzz)

        x_co = (x - nx / 2.0) / (nx / 2.0) - coilx
        y_co = (y - ny / 2.0) / (ny / 2.0) - coily
        z_co = (z - nz / 2.0) / (nz / 2.0) - coilz
        rr = (x_co**2 + y_co**2 + z_co**2)**0.5
        phi = np.arctan2(x_co, -y_co) + coil_phs
        out = (1 / rr) * np.exp(1j * phi)
    else:
        raise ValueError('Can only generate shape with length 3 or 4')

    rss = sum(abs(out) ** 2, 0)**0.5
    out /= rss

    return out.astype(dtype)
