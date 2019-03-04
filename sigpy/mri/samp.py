# -*- coding: utf-8 -*-
"""MRI sampling functions.
"""
import numpy as np
import numba as nb


__all__ = ['poisson', 'radial']


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
        return mask, r
    else:
        return mask


def radial(coord_shape, img_shape, golden=True, dtype=np.float):
    """Generate radial trajectory.

    Args:
        coord_shape (tuple of ints): coordinates of shape [ntr, nro, ndim]
        img_shape (tuple of ints): image shape.
        golden (bool): golden angle ordering.
        dtype (Dtype): data type.

    Returns:
        array: radial coordinates.

    """
    ntr, nro, ndim = coord_shape

    if ndim == 2:
        if golden:
            # 111.25 degrees in radians
            # from: Winkelmann, S. An Optimal Radial Profile Order Based on
            # the Golden Ratio for Time-Resolved MRI, IEEE TMI, 2007
            phi = 0.970839395
        else:
            phi = 1.0 / ntr

        n, r = np.mgrid[:ntr, :0.5:0.5 / nro]

        theta = 2.0 * np.pi * n * phi
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
        raise ValueError('Invalid ndim: {}'.format(ndim))

    return (coord * img_shape[-ndim:]).astype(dtype)


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _poisson(nx, ny, K, R, calib, seed):

    mask = np.zeros((ny, nx))
    f = ny / nx

    if seed is not None:
        np.random.seed(int(seed))

    pxs = np.empty(nx * ny, np.int32)
    pys = np.empty(nx * ny, np.int32)
    pxs[0] = np.random.randint(0, nx)
    pys[0] = np.random.randint(0, ny)
    m = 1
    while (m > 0):

        i = np.random.randint(0, m)
        px = pxs[i]
        py = pys[i]
        rad = R[py, px]

        # Attempt to generate point
        done = False
        k = 0
        while not done and k < K:

            # Generate point randomly from R and 2R
            rd = rad * (np.random.random() * 3 + 1)**0.5
            t = 2 * np.pi * np.random.random()
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
