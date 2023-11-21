"""Coil compression functions.

Author: Zhengguo Tan <zhengguo.tan@gmail.com>
"""
import sigpy as sp

__all__ = ['scc', 'gcc']


def scc(kdat, P=10, coil_dim=-2, device=sp.cpu_device):
    r"""Coil compression based on SVD.

    Args:
        kdat (array): raw k-space data.
        P (int): number of virtual coils to be kept. [Default: 10].
        coil_dim (int): the coil dimension. [Default: -2].
        device: use CPU or GPU device.

    Returns:
        coil compressed k-space data, and
        truncated eigen vectors.

    References:
        * Buehrer M., Pruessmann K. P., Boesiger P., Kozerke S. (2007).
          Array compression for MRI with large coil arrays.
          Magn. Reson. Med., 2007.

        * Huang F., Vijayakumar S., Li Y., Hertel S., Duensing G. R. (2008).
          A software channel compression technique for faster reconstruction with many channels.
          Magn. Reson. Imaging., 26, 133-141.
    """
    if P >= kdat.shape[coil_dim]:
        print('> return the original data')
        return kdat, None

    device = sp.Device(device)
    xp = device.xp

    with device:
        # move the dimension of coils to 0
        y1 = xp.swapaxes(sp.to_device(kdat, device=device), coil_dim, 0)
        y2 = xp.reshape(y1, (y1.shape[0], -1))

        # covariance matrix: [num_coil, num_coil]
        yc = xp.cov(y2)

        eigvals, eigvecs = xp.linalg.eigh(yc)

        # eigvals and eigvecs in descending order
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        print('> energy kept: ' +
              '%3.4f'%(xp.sum(eigvals[:P]) / xp.sum(eigvals)))

        S = eigvecs[:, :P]

        y3 = xp.conj(S.T) @ y2

        y4 = xp.reshape(y3, [P] + list(y1.shape[1:]))
        y5 = xp.swapaxes(y4, coil_dim, 0)

        return sp.to_device(y5, device=sp.get_device(kdat)), S


# TODO: geometric coil compression
def gcc():
    None
