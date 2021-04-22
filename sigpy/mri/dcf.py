"""Density compensation functions.

"""
import sigpy as sp
from tqdm.auto import tqdm


__all__ = ['pipe_menon_dcf']


def pipe_menon_dcf(coord, img_shape=None, device=sp.cpu_device, max_iter=30,
                   n=128, beta=8, width=4, show_pbar=True):
    r"""Compute Pipe Menon density compensation factor.

    Perform the following iteration:

    .. math::

        w = \frac{w}{|G^H G w|}

    with :math:`G` as the gridding operator.

    Args:
        coord (array): k-space coordinates.
        img_shape (None or list): Image shape.
        device (Device): computing device.
        max_iter (int): number of iterations.
        n (int): Kaiser-Bessel sampling numbers for gridding operator.
        beta (float): Kaiser-Bessel kernel parameter.
        width (float): Kaiser-Bessel kernel width.
        show_pbar (bool): show progress bar.

    Returns:
        array: density compensation factor.

    References:
        Pipe, James G., and Padmanabhan Menon.
        Sampling Density Compensation in MRI:
        Rationale and an Iterative Numerical Solution.
        Magnetic Resonance in Medicine 41, no. 1 (1999): 179–86.


    """
    device = sp.Device(device)
    xp = device.xp

    with device:
        w = xp.ones(coord.shape[:-1], dtype=coord.dtype)
        if img_shape is None:
            img_shape = sp.estimate_shape(coord)

        G = sp.linop.Gridding(img_shape, coord, param=beta,
                              width=width, kernel='kaiser_bessel')
        with tqdm(total=max_iter, desc="PipeMenonDCF",
                  disable=not show_pbar) as pbar:
            for it in range(max_iter):
                GHGw = G.H * G * w
                w /= xp.abs(GHGw)
                resid = xp.abs(GHGw - 1).max().item()

                pbar.set_postfix(resid='{0:.2E}'.format(resid))
                pbar.update()

    return w
