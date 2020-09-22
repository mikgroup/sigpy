# -*- coding: utf-8 -*-
"""MRI sampling functions.
"""
import numpy as np
import numba as nb


__all__ = ['poisson', 'radial', 'spiral']


def poisson(img_shape, accel, calib=(0, 0), dtype=np.complex,
            crop_corner=True, return_density=False, seed=0,
            max_attempts=30, tol=0.1):
    """Generate variable-density Poisson-disc sampling pattern.

    The function generates a variable density Poisson-disc sampling
    mask with density proportional to :math:`1 / (1 + s |r|)`,
    where :math:`r` represents the k-space radius, and :math:`s`
    represents the slope. A binary search is performed on the slope :math:`s`
    such that the resulting acceleration factor is close to the
    prescribed acceleration factor `accel`. The parameter `tol`
    determines how much they can deviate.

    Args:
        img_shape (tuple of ints): length-2 image shape.
        accel (float): Target acceleration factor. Must be greater than 1.
        calib (tuple of ints): length-2 calibration shape.
        dtype (Dtype): data type.
        crop_corner (bool): Toggle whether to crop sampling corners.
        seed (int): Random seed.
        max_attempts (float): maximum number of samples to reject in Poisson
           disc calculation.
        tol (float): Tolerance for how much the resulting acceleration can
            deviate form `accel`.

    Returns:
        array: Poisson-disc sampling mask.

    References:
        Bridson, Robert. "Fast Poisson disk sampling in arbitrary dimensions."
        SIGGRAPH sketches. 2007.

    """
    if accel <= 1:
        raise ValueError(f'accel must be greater than 1, got {accel}')

    if seed is not None:
        rand_state = np.random.get_state()

    ny, nx = img_shape
    y, x = np.mgrid[:ny, :nx]
    x = np.maximum(abs(x - img_shape[-1] / 2) - calib[-1] / 2, 0)
    x /= x.max()
    y = np.maximum(abs(y - img_shape[-2] / 2) - calib[-2] / 2, 0)
    y /= y.max()
    r = np.sqrt(x**2 + y**2)

    slope_max = max(nx, ny)
    slope_min = 0
    while slope_min < slope_max:
        slope = (slope_max + slope_min) / 2
        radius_x = np.clip((1 + r * slope) * nx / max(nx, ny), 1, None)
        radius_y = np.clip((1 + r * slope) * ny / max(nx, ny), 1, None)
        mask = _poisson(
            img_shape[-1], img_shape[-2], max_attempts,
            radius_x, radius_y, calib, seed)
        if crop_corner:
            mask *= r < 1

        actual_accel = img_shape[-1] * img_shape[-2] / np.sum(mask)

        if abs(actual_accel - accel) < tol:
            break
        if actual_accel < accel:
            slope_min = slope
        else:
            slope_max = slope

    if abs(actual_accel - accel) >= tol:
        raise ValueError(f'Cannot generate mask to satisfy accel={accel}.')

    mask = mask.reshape(img_shape).astype(dtype)

    if seed is not None:
        np.random.set_state(rand_state)

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
def _poisson(nx, ny, max_attempts, radius_x, radius_y, calib, seed=None):
    mask = np.zeros((ny, nx))

    if seed is not None:
        np.random.seed(int(seed))

    # initialize active list
    pxs = np.empty(nx * ny, np.int32)
    pys = np.empty(nx * ny, np.int32)
    pxs[0] = np.random.randint(0, nx)
    pys[0] = np.random.randint(0, ny)
    num_actives = 1
    while num_actives > 0:
        i = np.random.randint(0, num_actives)
        px = pxs[i]
        py = pys[i]
        rx = radius_x[py, px]
        ry = radius_y[py, px]

        # Attempt to generate point
        done = False
        k = 0
        while not done and k < max_attempts:
            # Generate point randomly from r and 2 * r
            v = (np.random.random() * 3 + 1)**0.5
            t = 2 * np.pi * np.random.random()
            qx = px + v * rx * np.cos(t)
            qy = py + v * ry * np.sin(t)

            # Reject if outside grid or close to other points
            if qx >= 0 and qx < nx and qy >= 0 and qy < ny:
                startx = max(int(qx - rx), 0)
                endx = min(int(qx + rx + 1), nx)
                starty = max(int(qy - ry), 0)
                endy = min(int(qy + ry + 1), ny)

                done = True
                for x in range(startx, endx):
                    for y in range(starty, endy):
                        if (mask[y, x] == 1
                            and (((qx - x) / radius_x[y, x])**2 +
                                 ((qy - y) / (radius_y[y, x]))**2 < 1)):
                            done = False
                            break

            k += 1

        # Add point if done else remove from active list
        if done:
            pxs[num_actives] = qx
            pys[num_actives] = qy
            mask[int(qy), int(qx)] = 1
            num_actives += 1
        else:
            pxs[i] = pxs[num_actives - 1]
            pys[i] = pys[num_actives - 1]
            num_actives -= 1

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
