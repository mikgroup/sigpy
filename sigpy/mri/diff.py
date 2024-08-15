"""
Methods for Diffusion MRI.

Author: Zhengguo Tan <zhengguo.tan@gmail.com>
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# %% Convert Cartesian coordinates to polar coordinates
def cartes2polar(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    varphi = np.zeros_like(r)

    ind = np.where(x > 0)
    varphi[ind] = np.arctan(y[ind] / x[ind])

    ind = np.where((x < 0) & (y >= 0))
    varphi[ind] = np.arctan(y[ind] / x[ind]) + np.pi

    ind = np.where((x < 0) & (y < 0))
    varphi[ind] = np.arctan(y[ind] / x[ind]) - np.pi

    ind = np.where((x == 0) & (y > 0))
    varphi[ind] = np.pi / 2

    ind = np.where((x == 0) & (y < 0))
    varphi[ind] = - np.pi / 2

    return r, theta, varphi