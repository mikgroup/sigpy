# -*- coding: utf-8 -*-
"""MRI applications.
"""
import numpy as np

import sigpy as sp
from sigpy.mri import linop

__all__ = [
    "SenseRecon",
    "L1WaveletRecon",
    "TotalVariationRecon",
    "JsenseRecon",
    "EspiritCalib",
]


def _estimate_weights(y, weights, coord):
    if weights is None and coord is None:
        with sp.get_device(y):
            weights = (sp.rss(y, axes=(0,)) > 0).astype(y.dtype)

    return weights


class SenseRecon(sp.app.LinearLeastSquares):
    r"""SENSE Reconstruction.

    Considers the problem

    .. math::
        \min_x \frac{1}{2} \| P F S x - y \|_2^2 +
        \frac{\lambda}{2} \| x \|_2^2

    where P is the sampling operator, F is the Fourier transform operator,
    S is the SENSE operator, x is the image, and y is the k-space measurements.

    Args:
        y (array): k-space measurements.
        mps (array): sensitivity maps.
        lamda (float): regularization parameter.
        weights (float or array): weights for data consistency.
        tseg (None or Dictionary): parameters for time-segmented off-resonance
            correction. Parameters are 'b0' (array), 'dt' (float),
            'lseg' (int), and 'n_bins' (int). Lseg is the number of
            time segments used, and n_bins is the number of histogram bins.
        coord (None or array): coordinates.
        device (Device): device to perform reconstruction.
        coil_batch_size (int): batch size to process coils.
            Only affects memory usage.
        comm (Communicator): communicator for distributed computing.
        **kwargs: Other optional arguments.

    References:
        Pruessmann, K. P., Weiger, M., Scheidegger, M. B., & Boesiger, P.
        (1999).
        SENSE: sensitivity encoding for fast MRI.
        Magnetic resonance in medicine, 42(5), 952-962.

        Pruessmann, K. P., Weiger, M., Bornert, P., & Boesiger, P. (2001).
        Advances in sensitivity encoding with arbitrary k-space trajectories.
        Magnetic resonance in medicine, 46(4), 638-651.

    """

    def __init__(
        self,
        y,
        mps,
        lamda=0,
        weights=None,
        tseg=None,
        coord=None,
        device=sp.cpu_device,
        coil_batch_size=None,
        comm=None,
        show_pbar=True,
        transp_nufft=False,
        **kwargs
    ):
        weights = _estimate_weights(y, weights, coord)
        if weights is not None:
            y = sp.to_device(y * weights**0.5, device=device)
        else:
            y = sp.to_device(y, device=device)

        A = linop.Sense(
            mps,
            coord=coord,
            weights=weights,
            tseg=tseg,
            coil_batch_size=coil_batch_size,
            comm=comm,
            transp_nufft=transp_nufft,
        )

        if comm is not None:
            show_pbar = show_pbar and comm.rank == 0

        super().__init__(A, y, lamda=lamda, show_pbar=show_pbar, **kwargs)


class L1WaveletRecon(sp.app.LinearLeastSquares):
    r"""L1 Wavelet regularized reconstruction.

    Considers the problem

    .. math::
        \min_x \frac{1}{2} \| P F S x - y \|_2^2 + \lambda \| W x \|_1

    where P is the sampling operator, F is the Fourier transform operator,
    S is the SENSE operator, W is the wavelet operator,
    x is the image, and y is the k-space measurements.

    Args:
        y (array): k-space measurements.
        mps (array): sensitivity maps.
        lamda (float): regularization parameter.
        weights (float or array): weights for data consistency.
        coord (None or array): coordinates.
        wave_name (str): wavelet name.
        device (Device): device to perform reconstruction.
        coil_batch_size (int): batch size to process coils.
        Only affects memory usage.
        comm (Communicator): communicator for distributed computing.
        **kwargs: Other optional arguments.

    References:
        Lustig, M., Donoho, D., & Pauly, J. M. (2007).
        Sparse MRI: The application of compressed sensing for rapid MR imaging.
        Magnetic Resonance in Medicine, 58(6), 1082-1195.

    """

    def __init__(
        self,
        y,
        mps,
        lamda,
        weights=None,
        coord=None,
        wave_name="db4",
        device=sp.cpu_device,
        coil_batch_size=None,
        comm=None,
        show_pbar=True,
        transp_nufft=False,
        **kwargs
    ):
        weights = _estimate_weights(y, weights, coord)
        if weights is not None:
            y = sp.to_device(y * weights**0.5, device=device)
        else:
            y = sp.to_device(y, device=device)

        A = linop.Sense(
            mps,
            coord=coord,
            weights=weights,
            comm=comm,
            coil_batch_size=coil_batch_size,
            transp_nufft=transp_nufft,
        )
        img_shape = mps.shape[1:]
        W = sp.linop.Wavelet(img_shape, wave_name=wave_name)
        proxg = sp.prox.UnitaryTransform(sp.prox.L1Reg(W.oshape, lamda), W)

        def g(input):
            device = sp.get_device(input)
            xp = device.xp
            with device:
                return lamda * xp.sum(xp.abs(W(input))).item()

        if comm is not None:
            show_pbar = show_pbar and comm.rank == 0

        super().__init__(A, y, proxg=proxg, g=g, show_pbar=show_pbar, **kwargs)


class TotalVariationRecon(sp.app.LinearLeastSquares):
    r"""Total variation regularized reconstruction.

    Considers the problem:

    .. math::
        \min_x \frac{1}{2} \| P F S x - y \|_2^2 + \lambda \| G x \|_1

    where P is the sampling operator, F is the Fourier transform operator,
    S is the SENSE operator, G is the gradient operator,
    x is the image, and y is the k-space measurements.

    Args:
        y (array): k-space measurements.
        mps (array): sensitivity maps.
        lamda (float): regularization parameter.
        weights (float or array): weights for data consistency.
        coord (None or array): coordinates.
        device (Device): device to perform reconstruction.
        coil_batch_size (int): batch size to process coils.
        Only affects memory usage.
        comm (Communicator): communicator for distributed computing.
        **kwargs: Other optional arguments.

    References:
        Block, K. T., Uecker, M., & Frahm, J. (2007).
        Undersampled radial MRI with multiple coils.
        Iterative image reconstruction using a total variation constraint.
        Magnetic Resonance in Medicine, 57(6), 1086-1098.

    """

    def __init__(
        self,
        y,
        mps,
        lamda,
        weights=None,
        coord=None,
        device=sp.cpu_device,
        coil_batch_size=None,
        comm=None,
        show_pbar=True,
        transp_nufft=False,
        **kwargs
    ):
        weights = _estimate_weights(y, weights, coord)
        if weights is not None:
            y = sp.to_device(y * weights**0.5, device=device)
        else:
            y = sp.to_device(y, device=device)

        A = linop.Sense(
            mps,
            coord=coord,
            weights=weights,
            comm=comm,
            coil_batch_size=coil_batch_size,
            transp_nufft=transp_nufft,
        )

        G = sp.linop.FiniteDifference(A.ishape)
        proxg = sp.prox.L1Reg(G.oshape, lamda)

        def g(x):
            device = sp.get_device(x)
            xp = device.xp
            with device:
                return lamda * xp.sum(xp.abs(x)).item()

        if comm is not None:
            show_pbar = show_pbar and comm.rank == 0

        super().__init__(
            A, y, proxg=proxg, g=g, G=G, show_pbar=show_pbar, **kwargs
        )


class JsenseRecon(sp.app.App):
    r"""JSENSE/NLINV reconstruction.

    Considers the problem

    .. math::
        \min_{l, r} \frac{1}{2} \| l \ast r - y \|_2^2 +
        \frac{\lambda}{2} (\| l \|_2^2 + \| r \|_2^2)

    where :math:`\ast` is the convolution operator.

    This formulation with regularization corresponds to the version
    described in the NLINV paper. Without regularization (which is the
    default) this corresponds to the version from the JSENSE paper but using a
    truncated Fourier representation of the coils (as in NLINV) instead
    of polynomials.

    Args:
        y (array): k-space measurements.
        mps_ker_width (int): sensitivity maps kernel width.
        ksp_calib_width (int): k-space calibration width.
        lamda (float): regularization parameter.
        device (Device): device to perform reconstruction.
        weights (float or array): weights for data consistency.
        coord (None or array): coordinates.
        img_shape (None or list): Image shape.
        grd_shape (None or list): Shape of grid.
        max_iter (int): Maximum number of iterations.
        max_inner_iter (int): Maximum number of inner iterations.

    References:
        Ying, L., & Sheng, J. (2007).
        Joint image reconstruction and sensitivity estimation in SENSE
        (JSENSE).
        Magnetic Resonance in Medicine, 57(6), 1196-1202.

        Uecker, M., Hohage, T., Block, K. T., & Frahm, J. (2008).
        Image reconstruction by regularized nonlinear inversion-
        joint estimation of coil sensitivities and image content.
        Magnetic Resonance in Medicine, 60(#), 674-682.

    """

    def __init__(
        self,
        y,
        mps_ker_width=16,
        ksp_calib_width=24,
        lamda=0,
        device=sp.cpu_device,
        comm=None,
        weights=None,
        coord=None,
        img_shape=None,
        grd_shape=None,
        max_iter=10,
        max_inner_iter=10,
        normalize=True,
        show_pbar=True,
    ):
        self.y = y
        self.mps_ker_width = mps_ker_width
        self.ksp_calib_width = ksp_calib_width
        self.lamda = lamda
        self.weights = weights
        self.coord = coord
        self.img_shape = img_shape
        self.grd_shape = grd_shape
        self.max_iter = max_iter
        self.max_inner_iter = max_inner_iter
        self.normalize = normalize

        self.device = sp.Device(device)
        self.comm = comm
        self.dtype = y.dtype
        self.num_coils = len(y)
        if comm is not None:
            show_pbar = show_pbar and comm.rank == 0

        self._get_data()
        self._get_vars()
        self._get_alg()
        super().__init__(self.alg, show_pbar=show_pbar)

    def _get_data(self):
        if self.coord is None:
            self.img_shape = list(self.y.shape[1:])
            ndim = len(self.img_shape)

            self.y = sp.resize(
                self.y, [self.num_coils] + ndim * [self.ksp_calib_width]
            )

            if self.weights is not None:
                self.weights = sp.resize(
                    self.weights, ndim * [self.ksp_calib_width]
                )

        else:
            if self.img_shape is None:
                self.img_shape = sp.estimate_shape(self.coord)
            else:
                self.img_shape = list(self.img_shape)

            calib_idx = (
                np.amax(np.abs(self.coord), axis=-1) < self.ksp_calib_width / 2
            )

            self.coord = self.coord[calib_idx]
            self.y = self.y[:, calib_idx]

            if self.weights is not None:
                self.weights = self.weights[calib_idx]

        if self.weights is None:
            self.y = sp.to_device(self.y, self.device)
        else:
            self.y = sp.to_device(self.weights**0.5 * self.y, self.device)

        if self.coord is not None:
            self.coord = sp.to_device(self.coord, self.device)
        if self.weights is not None:
            self.weights = sp.to_device(self.weights, self.device)

        self.weights = _estimate_weights(self.y, self.weights, self.coord)

        if self.normalize:
            xp = self.device.xp
            with self.device:
                self.y = self.y / xp.linalg.norm(self.y)

    def _get_vars(self):
        ndim = len(self.img_shape)

        mps_ker_shape = [self.num_coils] + [self.mps_ker_width] * ndim
        if self.coord is None:
            img_ker_shape = [
                i + self.mps_ker_width - 1 for i in self.y.shape[1:]
            ]
        else:
            if self.grd_shape is None:
                self.grd_shape = sp.estimate_shape(self.coord)

            img_ker_shape = [
                i + self.mps_ker_width - 1 for i in self.grd_shape
            ]

        self.img_ker = sp.dirac(
            img_ker_shape, dtype=self.dtype, device=self.device
        )
        with self.device:
            self.mps_ker = self.device.xp.zeros(
                mps_ker_shape, dtype=self.dtype
            )

    def _get_alg(self):
        def min_mps_ker():
            self.A_mps_ker = linop.ConvImage(
                self.mps_ker.shape,
                self.img_ker,
                coord=self.coord,
                weights=self.weights,
            )
            sp.app.LinearLeastSquares(
                self.A_mps_ker,
                self.y,
                self.mps_ker,
                lamda=self.lamda,
                max_iter=self.max_inner_iter,
                show_pbar=False,
            ).run()

        def min_img_ker():
            self.A_img_ker = linop.ConvSense(
                self.img_ker.shape,
                self.mps_ker,
                coord=self.coord,
                weights=self.weights,
                comm=self.comm,
            )
            sp.app.LinearLeastSquares(
                self.A_img_ker,
                self.y,
                self.img_ker,
                lamda=self.lamda,
                max_iter=self.max_inner_iter,
                show_pbar=False,
            ).run()

        self.alg = sp.alg.AltMin(
            min_mps_ker, min_img_ker, max_iter=self.max_iter
        )

    def _output(self):
        xp = self.device.xp
        # Normalize by root-sum-of-squares.
        with self.device:
            rss = 0
            mps = np.empty([self.num_coils] + self.img_shape, dtype=self.dtype)
            for c in range(self.num_coils):
                mps_c = sp.ifft(sp.resize(self.mps_ker[c], self.img_shape))
                rss += xp.abs(mps_c) ** 2
                sp.copyto(mps[c], mps_c)

            rss = sp.to_device(rss)
            if self.comm is not None:
                self.comm.allreduce(rss)

            rss = rss**0.5
            mps /= rss
            return mps


class EspiritCalib(sp.app.App):
    """ESPIRiT calibration.

    Currently only supports outputting one set of maps.

    Args:
        ksp (array): k-space array of shape [num_coils, n_ndim, ..., n_1]
        calib (tuple of ints): length-2 image shape.
        thresh (float): threshold for the calibration matrix.
        kernel_width (int): kernel width for the calibration matrix.
        max_power_iter (int): maximum number of power iterations.
        device (Device): computing device.
        crop (int): cropping threshold.

    Returns:
        array: ESPIRiT maps of the same shape as ksp.

    References:
        Martin Uecker, Peng Lai, Mark J. Murphy, Patrick Virtue, Michael Elad,
        John M. Pauly, Shreyas S. Vasanawala, and Michael Lustig
        ESPIRIT - An Eigenvalue Approach to Autocalibrating Parallel MRI:
        Where SENSE meets GRAPPA.
        Magnetic Resonance in Medicine, 71:990-1001 (2014)

    """

    def __init__(
        self,
        ksp,
        calib_width=24,
        thresh=0.02,
        kernel_width=6,
        crop=0.95,
        max_iter=100,
        device=sp.cpu_device,
        output_eigenvalue=False,
        show_pbar=True,
    ):
        self.device = sp.Device(device)
        self.output_eigenvalue = output_eigenvalue
        self.crop = crop

        img_ndim = ksp.ndim - 1
        num_coils = len(ksp)
        with sp.get_device(ksp):
            # Get calibration region
            calib_shape = [num_coils] + [calib_width] * img_ndim
            calib = sp.resize(ksp, calib_shape)
            calib = sp.to_device(calib, device)

        xp = self.device.xp
        with self.device:
            # Get calibration matrix.
            # Shape [num_coils] + num_blks + [kernel_width] * img_ndim
            mat = sp.array_to_blocks(
                calib, [kernel_width] * img_ndim, [1] * img_ndim
            )
            mat = mat.reshape([num_coils, -1, kernel_width**img_ndim])
            mat = mat.transpose([1, 0, 2])
            mat = mat.reshape([-1, num_coils * kernel_width**img_ndim])

            # Perform SVD on calibration matrix
            _, S, VH = xp.linalg.svd(mat, full_matrices=False)
            VH = VH[S > thresh * S.max(), :]

            # Get kernels
            num_kernels = len(VH)
            kernels = VH.reshape(
                [num_kernels, num_coils] + [kernel_width] * img_ndim
            )
            img_shape = ksp.shape[1:]

            # Get covariance matrix in image domain
            AHA = xp.zeros(
                img_shape[::-1] + (num_coils, num_coils), dtype=ksp.dtype
            )
            for kernel in kernels:
                img_kernel = sp.ifft(
                    sp.resize(kernel, ksp.shape), axes=range(-img_ndim, 0)
                )
                aH = xp.expand_dims(img_kernel.T, axis=-1)
                a = xp.conj(aH.swapaxes(-1, -2))
                AHA += aH @ a

            AHA *= sp.prod(img_shape) / kernel_width**img_ndim
            self.mps = xp.ones(ksp.shape[::-1] + (1,), dtype=ksp.dtype)

            def forward(x):
                with sp.get_device(x):
                    return AHA @ x

            def normalize(x):
                with sp.get_device(x):
                    return (
                        xp.sum(xp.abs(x) ** 2, axis=-2, keepdims=True) ** 0.5
                    )

            alg = sp.alg.PowerMethod(
                forward, self.mps, norm_func=normalize, max_iter=max_iter
            )

        super().__init__(alg, show_pbar=show_pbar)

    def _output(self):
        xp = self.device.xp
        with self.device:
            # Normalize phase with respect to first channel
            mps = self.mps.T[0]
            mps *= xp.conj(mps[0] / xp.abs(mps[0]))

            # Crop maps by thresholding eigenvalue
            max_eig = self.alg.max_eig.T[0]
            mps *= max_eig > self.crop

        if self.output_eigenvalue:
            return mps, max_eig
        else:
            return mps
