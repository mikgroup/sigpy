# -*- coding: utf-8 -*-
"""MRI sampling functions.
"""
import numpy as np
import numba as nb


__all__ = ['poisson', 'radial', 'spiral']


def poisson(img_shape, accel, K=30, calib=[0, 0], dtype=np.complex,
            crop_corner=True, return_density=False, seed=0):
    """Generate Poisson-disc sampling pattern

    Args:
        img_shape (tuple of ints): length-2 image shape.
        accel (float): Target acceleration factor. Greater than 1.
        K (float): maximum number of samples to reject.
        calib (tuple of ints): length-2 calibration shape.
        dtype (Dtype): data type.
        crop_corner (bool): Toggle whether to crop sampling corners.
        return_density (bool): Toggle whether to return sampling density.
        seed (int): Random seed.

    Returns:
        array: Poisson-disc sampling mask.

    References:
        Bridson, Robert. "Fast Poisson disk sampling in arbitrary dimensions."
        SIGGRAPH sketches. 2007.

    """
    y, x = np.mgrid[:img_shape[-2], :img_shape[-1]]
    x = np.maximum(abs(x - img_shape[-1] / 2) - calib[-1] / 2, 0)
    x /= x.max()
    y = np.maximum(abs(y - img_shape[-2] / 2) - calib[-2] / 2, 0)
    y /= y.max()
    r = np.sqrt(x ** 2 + y ** 2)

    slope_max = 40
    slope_min = 0
    while slope_min < slope_max:
        slope = (slope_max + slope_min) / 2.0
        R = (1.0 + r * slope)
        mask = _poisson(img_shape[-1], img_shape[-2], K, R, calib, seed)
        if crop_corner:
            mask *= r < 1

        est_accel = img_shape[-1] * img_shape[-2] / np.sum(mask[:])

        if abs(est_accel - accel) < 0.1:
            break
        if est_accel < accel:
            slope_min = slope
        else:
            slope_max = slope

    mask = mask.reshape(img_shape).astype(dtype)
    if return_density:
        return mask, R
    else:
        return mask


def radial(coord_shape, img_shape, golden=True, dtype=np.float):
    """Generate radial trajectory.

    Args:
        coord_shape (tuple of ints): coordinates of shape [ntr, nro, ndim],
            where ntr is the number of TRs, nro is the number of readout,
            and ndim is the number of dimensions.
        img_shape (tuple of ints): image shape.
        golden (bool): golden angle ordering.
        dtype (Dtype): data type.

    Returns:
        array: radial coordinates.

    References:
        1. An Optimal Radial Profile Order Based on the Golden
        Ratio for Time-Resolved MRI
        Stefanie Winkelmann, Tobias Schaeffter, Thomas Koehler,
        Holger Eggers, and Olaf Doessel. TMI 2007.
        2. Temporal stability of adaptive 3D radial MRI using
        multidimensional golden means
        Rachel W. Chan, Elizabeth A. Ramsay, Charles H. Cunningham,
        and Donald B. Plewes. MRM 2009.

    """
    if len(img_shape) != coord_shape[-1]:
        raise ValueError(
            'coord_shape[-1] must match len(img_shape), '
            'got {} and {}'.format(coord_shape[-1], len(img_shape)))

    ntr, nro, ndim = coord_shape
    if ndim == 2:
        if golden:
            phi = np.pi * (3 - 5**0.5)
        else:
            phi = 2 * np.pi / ntr

        n, r = np.mgrid[:ntr, :0.5:0.5 / nro]

        theta = n * phi
        coord = np.zeros((ntr, nro, 2))
        coord[:, :, -1] = r * np.cos(theta)
        coord[:, :, -2] = r * np.sin(theta)

    elif ndim == 3:
        if golden:
            phi1 = 0.465571231876768
            phi2 = 0.682327803828019
        else:
            raise NotImplementedError

        n, r = np.mgrid[:ntr, :0.5:0.5 / nro]

        beta = np.arccos(2 * ((n * phi1) % 1) - 1)
        alpha = 2 * np.pi * ((n * phi2) % 1)

        coord = np.zeros((ntr, nro, 3))
        coord[:, :, -1] = r * np.sin(beta) * np.cos(alpha)
        coord[:, :, -2] = r * np.sin(beta) * np.sin(alpha)
        coord[:, :, -3] = r * np.cos(beta)
    else:
        raise ValueError(
            'coord_shape[-1] must be 2 or 3, got {}'.format(ndim))

    return (coord * img_shape[-ndim:]).astype(dtype)


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _poisson(nx, ny, K, R, calib, seed):

    mask = np.zeros((ny, nx))
    f = ny / nx

    if seed is not None:
        rand_state = np.random.RandomState(int(seed))
    else:
        rand_state = np.random

    pxs = np.empty(nx * ny, np.int32)
    pys = np.empty(nx * ny, np.int32)
    pxs[0] = rand_state.randint(0, nx)
    pys[0] = rand_state.randint(0, ny)
    m = 1
    while (m > 0):

        i = rand_state.randint(0, m)
        px = pxs[i]
        py = pys[i]
        rad = R[py, px]

        # Attempt to generate point
        done = False
        k = 0
        while not done and k < K:

            # Generate point randomly from R and 2R
            rd = rad * (rand_state.random() * 3 + 1)**0.5
            t = 2 * np.pi * rand_state.random()
            qx = px + rd * np.cos(t)
            qy = py + rd * f * np.sin(t)

            # Reject if outside grid or close to other points
            if qx >= 0 and qx < nx and qy >= 0 and qy < ny:

                startx = max(int(qx - rad), 0)
                endx = min(int(qx + rad + 1), nx)
                starty = max(int(qy - rad * f), 0)
                endy = min(int(qy + rad * f + 1), ny)

                done = True
                for x in range(startx, endx):
                    for y in range(starty, endy):
                        if (mask[y, x] == 1
                            and (((qx - x) / R[y, x]) ** 2 +
                                 ((qy - y) / (R[y, x] * f)) ** 2 < 1)):
                            done = False
                            break

                    if not done:
                        break

            k += 1

        # Add point if done else remove active
        if done:
            pxs[m] = qx
            pys[m] = qy
            mask[int(qy), int(qx)] = 1
            m += 1
        else:
            pxs[i] = pxs[m - 1]
            pys[i] = pys[m - 1]
            m -= 1

    # Add calibration region
    mask[int(ny / 2 - calib[-2] / 2):int(ny / 2 + calib[-2] / 2),
         int(nx / 2 - calib[-1] / 2):int(nx / 2 + calib[-1] / 2)] = 1

    return mask


def spiral(fov, N, f_sampling, R, ninterleaves, alpha, gm, sm, gamma=2.678e8):
    """Generate variable density spiral trajectory.

    Args:
        fov (float): field of view in meters.
        N (int): effective matrix shape.
        f_sampling (float): undersampling factor in freq encoding direction.
        R (float): undersampling factor.
        ninterleaves (int): number of spiral interleaves
        alpha (float): variable density factor
        gm (float): maximum gradient amplitude (T/m)
        sm (float): maximum slew rate (T/m/s)
        gamma (float): gyromagnetic ratio in rad/T/s

    Returns:
        array: spiral coordinates.

    References:
        Dong-hyun Kim, Elfar Adalsteinsson, and Daniel M. Spielman.
        'Simple Analytic Variable Density Spiral Design.' MRM 2003.

    """
    res = fov/N

    lam = .5 / res  # in m**(-1)
    n = 1 / (1 - (1 - ninterleaves * R / fov / lam) ** (1 / alpha))
    w = 2 * np.pi * n
    Tea = lam * w / gamma / gm / (alpha + 1)  # in s
    Tes = np.sqrt(lam * w ** 2 / sm / gamma) / (alpha / 2 + 1)  # in s
    Ts2a = (Tes ** ((alpha + 1) / (alpha / 2 + 1)) *
            (alpha / 2 + 1) / Tea / (alpha + 1)) ** (1 + 2 / alpha)  # in s

    if Ts2a < Tes:
        tautrans = (Ts2a / Tes) ** (1 / (alpha / 2 + 1))

        def tau(t):
            return (t / Tes) ** (1 / (alpha / 2 + 1)) * (0 <= t) * \
                (t <= Ts2a) + ((t - Ts2a) / Tea +
                               tautrans ** (alpha + 1)) ** (1 / (alpha + 1))\
                * (t > Ts2a) * (t <= Tea) * (Tes >= Ts2a)
        Tend = Tea
    else:

        def tau(t):
            return (t / Tes) ** (1 / (alpha / 2 + 1)) * (0 <= t) * (t <= Tes)
        Tend = Tes

    def k(t):
        return lam * tau(t) ** alpha * np.exp(w * tau(t) * 1j)
    dt = Tea * 1E-4  # in s

    Dt = dt * f_sampling / fov / abs(k(Tea) - k(Tea - dt))  # in s

    t = np.linspace(0, Tend, int(Tend / Dt))
    kt = k(t)  # in rad

    # generating cloned interleaves
    k = kt
    for i in range(1, ninterleaves):
        k = np.hstack((k, kt[0:] * np.exp(2 * np.pi * 1j * i / ninterleaves)))

    k = np.stack((np.real(k), np.imag(k)), axis=1)

    return k
