import numpy as np

__all__ = ['birdcage_maps', 'shepp_logan']


def shepp_logan(shape, dtype=np.complex):
    """Generates a Shepp Logan phantom with a given shape and dtype.

    Args:
        shape (tuple of ints): shape, can be of length 2 or 3.
        dtype (Dtype): data type.

    Returns:
        array.
    """
    return phantom(shape, sl_amps, sl_scales, sl_offsets, sl_angles, dtype)


def birdcage_maps(shape, r=1.5, nzz=8, dtype=np.complex):
    """Simulates birdcage coil sensitivies.

    Args:
        shape (tuple of ints): sensitivity maps shape, can be of length 3, and 4.
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


sl_amps = [1, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
sl_t1_amps = [1000, -800, -200, -200, 100, 100, 100, 100, 100, 100]
sl_t2_amps = [1, 200, 0, 0, -100, -100, 100, 100, 100, 200]

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


oscillation_a = [0, 0, 0, 0, 0.0, 0.3, 0.3, 0.3, 0.3, 0.3]
oscillation_f = [0, 0, 0, 0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0]
enhancement_a = [0, 0, 0, 0, 0.6, 0.3, 0.3, 0.3, 0.3, 0.3]
enhancement_f = [0, 0, 0, 0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]


def dynamic_shepp_logan(shape, dtype=np.complex):
    """
    Generates a Shepp Logan phantom with a given shape
    """
    nt = shape[0]
    output = np.zeros(shape, dtype=dtype)

    for t in range(nt):

        amps = (np.array(sl_amps) +
                oscillation_a * np.sin(2 * np.pi * t / nt * np.array(oscillation_f)) +
                enhancement_a * (1 - np.exp(-np.array(enhancement_f) * t / nt)))

        output[t, ...] = phantom(
            shape[1:], amps, sl_scales, sl_offsets, sl_angles, dtype)

    return output


def quant_shepp_logan(shape, dtype=np.complex):
    """
    Generate Quantatiative t1, t2, offresonance maps
    """

    t1 = phantom(shape, sl_t1_amps, sl_scales, sl_offsets, sl_angles, dtype)
    t2 = phantom(shape, sl_t2_amps, sl_scales, sl_offsets, sl_angles, dtype)
    f = (1 - np.outer(np.hanning(shape[0]),
                      np.hanning(shape[1]))).astype(dtype) * 0.001
    proton = phantom(shape, sl_amps, sl_scales, sl_offsets, sl_angles, dtype)

    return t1 * (proton > 0), t2 * (proton > 0), f * (proton > 0), proton


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
