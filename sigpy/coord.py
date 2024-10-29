# -*- coding: utf-8 -*-
"""Methods for Coordinate Conversion

Reference:
    https://math.libretexts.org/Bookshelves/Calculus/Book%3A_Calculus_(OpenStax)/12%3A_Vectors_in_Space/12.07%3A_Cylindrical_and_Spherical_Coordinates#:~:text=To%20convert%20a%20point%20from,y2%2Bz2).

Author: Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import numpy as np


def cartes_to_spheri(x, y, z):

    r = (x**2 + y**2 + z**2)**0.5
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)

    return r, theta, phi


def spheri_to_cartes(r, theta, phi):

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    return x, y, z
