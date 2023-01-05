# -*- coding: utf-8 -*-
"""MRI simulation functions.
"""
import numpy as np
import math

from sigpy.mri import epi

__all__ = ['birdcage_maps', 'gradient_echoes', 'diffusion', 'get_subspace']


def birdcage_maps(shape, r=1.5, nzz=8, dtype=complex):
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
        rr = np.sqrt(x_co**2 + y_co**2)
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
        rr = (x_co**2 + y_co**2 + z_co**2) ** 0.5
        phi = np.arctan2(x_co, -y_co) + coil_phs
        out = (1 / rr) * np.exp(1j * phi)
    else:
        raise ValueError("Can only generate shape with length 3 or 4")

    rss = sum(abs(out) ** 2, 0) ** 0.5
    out /= rss

    return out.astype(dtype)


def _linspace_to_array(linspace_list):
    val_start = linspace_list[0]
    val_stop = linspace_list[1]
    num = linspace_list[2]

    if num == 1:
        return [val_start]
    else:
        step = (val_stop - val_start) / (num - 1)
        return val_start + step * np.arange(num)


def gradient_echoes(TE, B0_linspace=(-50, 50, 101),
                    T2star_linspace=(0.001, 0.200, 100),
                    rho0_linspace=(0.01, 1, 100)):
    """Compute gradient-echo signals

    signal = rho * exp(-TE/T2star) * exp(1i * 2 pi * B0 * TE)

    Args:
        TE (array): echo times [second]
        B0_linspace (list of floats): start, stop, num [Hz]
        T2star_linspace (list of floats): start, stop, num [second]
        rho0_linspace (list of floats): start, stop, num [a.u.]

    Returns:
        sig (array)

    Author:
        Zhengguo Tan <zhengguo.tan@gmail.com>
    """
    B0_array = _linspace_to_array(B0_linspace)
    T2star_array = _linspace_to_array(T2star_linspace)
    rho0_array = _linspace_to_array(rho0_linspace)

    sig = np.zeros((len(TE), 1, 1, 1, len(B0_array),
                    len(T2star_array), len(rho0_array)), dtype=complex)

    for B0_ind in np.arange(B0_linspace[2]):
        B0_val = B0_array[B0_ind]

        for T2star_ind in np.arange(T2star_linspace[2]):
            T2star_val = T2star_array[T2star_ind]
            z = (-1/T2star_val + 1j*2*math.pi*B0_val)

            for rho0_ind in np.arange(rho0_linspace[2]):
                rho0_val = rho0_array[rho0_ind]

                sig[:, 0, 0, 0, B0_ind, T2star_ind, rho0_ind] = \
                    rho0_val * np.exp(z * TE)

    return sig


def diffusion(b=None, g=None, D=(0, 0.004, 10)):
    """Compute diffusion signals

    signal = exp(B * D)

    Args:
        b (array): echo times [second]
        g (array): start, stop, num [Hz]

    Returns:
        sig (array)

    Author:
        Zhengguo Tan <zhengguo.tan@gmail.com>
    """
    assert (not ((b is None) ^ (g is None)))

    if b is None:
        None

    B = epi.get_B(b, g)

    D_start = D[0]
    D_end = D[1]
    D_num = D[2]

    D_grid = np.linspace(D_start, D_end, D_num)

    D = np.meshgrid(D_grid, D_grid, D_grid, D_grid, D_grid, D_grid)
    D = np.array(D).reshape((6, -1))

    return np.exp(np.matmul(B, D))


def get_subspace(sig, num_coeffs=5, error_bound=1e-5, prior_err=True):
    """Compute linear subspace coefficients of MR signal.

    Args:
        sig (array): MR signal
        num_coeffs (int): expected number of coefficients
        error_bound (float): relative error bound between the subspace signal
                            and the ground-truth signal
        prior_err (boolean): iteratively increase the number of coefficients
                            in order to reach the error_bound

    Returns:
        U_sub (array): truncated subspace matrix

    References:
        Huang C., Graff C. G., Clarkson E. W.,
        Bilgin A., Altbach M. I. (2012).
        T2 mapping from highly undersampled data by
        reconstruction of principal component coefficient maps
        using compressed sensing.
        Magn. Reson. Med., 67, 1355-1366.

        Tamir J. I., Uecker M., Chen W., Lai P., Alley M. T.,
        Vasanawala S. S., Lustig M. (2017).
        T2 shuffling: Sharp, multicontrast, volumetric fast spin-echo imaging.
        Magn. Reson. Med., 77, 180-195.

    Author:
        Zhengguo Tan <zhengguo.tan@gmail.com>
    """
    def get_rel_error(recov_sig, full_sig):
        return np.linalg.norm(full_sig - recov_sig).item() \
            / np.linalg.norm(full_sig).item()

    # reshape sig to (number of echoes, number of dictionary atoms)
    sig2 = np.reshape(sig, (sig.shape[-7], -1))

    # singular value decomposition
    U, S, VH = np.linalg.svd(sig2, full_matrices=False)

    while True:
        # truncate the U matrix
        U_sub = U[:, :num_coeffs]

        # recover from U_sub
        recov_sig = U_sub @ U_sub.T @ sig2

        err = get_rel_error(recov_sig, sig2)

        if (err > error_bound) and (prior_err):
            num_coeffs += 1
        else:
            print('Eventual number of subspace coefficients: '
                  + str(num_coeffs))
            print('Eventual error: ' + str(err))
            return U_sub
