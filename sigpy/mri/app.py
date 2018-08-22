import pickle
import numpy as np
import sigpy as sp

from sigpy.mri import linop


if sp.config.mpi4py_enabled:
    from mpi4py import MPI


def _estimate_weights(ksp, weights, coord):
    if np.isscalar(weights) and coord is None:
        with sp.util.get_device(ksp):
            weights = sp.util.rss(ksp, axes=(0, )) > 0
    return weights


def _move_to_device(ksp, mps, weights, coord, device):
    ksp = sp.util.move(ksp, device=device)
    mps = sp.util.move(mps, device=device)

    if not np.isscalar(weights):
        weights = sp.util.move(weights, device=device)

    if coord is not None:
        coord = sp.util.move(coord, device=device)

    return ksp, mps, weights, coord


class SenseRecon(sp.app.LinearLeastSquares):
    r"""SENSE Reconstruction.

    Considers the problem

    .. math:: \min_x \frac{1}{2} \| P F S x - y \|_2^2 + \frac{\lambda}{2} \| x \|_2^2
    where P is the sampling operator, F is the Fourier transform operator,
    S is the SENSE operator, x is the image, and y is the k-space measurements.

    Args:
        ksp (array): k-space measurements.
        mps (array): sensitivity maps.
        lamda (float): regularization parameter.
        weights (float or array): weights for data consistency.
        coord (None or array): coordinates.
        device (Device): device to perform reconstruction.
        **kwargs: Other optional arguments.

    References:
        Pruessmann, K. P., Weiger, M., Scheidegger, M. B., & Boesiger, P. (1999). 
        SENSE: sensitivity encoding for fast MRI. 
        Magnetic resonance in medicine, 42(5), 952-962.

        Pruessmann, K. P., Weiger, M., Bornert, P., & Boesiger, P. (2001).
        Advances in sensitivity encoding with arbitrary k-space trajectories.
        Magnetic resonance in medicine, 46(4), 638-651.
       
    """

    def __init__(self, ksp, mps, lamda=0, weights=1,
                 coord=None, device=sp.util.cpu_device, **kwargs):
        ksp, mps, weights, coord = _move_to_device(
            ksp, mps, weights, coord, device)

        weights = _estimate_weights(ksp, weights, coord)
        A = linop.Sense(mps, coord=coord)
        self.img = sp.util.zeros(mps.shape[1:], dtype=ksp.dtype, device=device)

        super().__init__(A, ksp, self.img, lamda=lamda, weights=weights, **kwargs)


class SenseConstrainedRecon(sp.app.L2ConstrainedMinimization):
    r"""SENSE constrained reconstruction.

    Considers the problem

    .. math::
        \min_x &\| x \|_2^2 \\
        \text{s.t.} &\| P F S x - y \|_2^2 \le \epsilon
    where P is the sampling operator, F is the Fourier transform operator,
    S is the SENSE operator, x is the image, and y is the k-space measurements.

    Args:
        ksp (array): k-space measurements.
        mps (array): sensitivity maps.
        eps (float): constraint parameter.
        weights (float or array): weights for data consistency.
        coord (None or array): coordinates.
        device (Device): device to perform reconstruction.
        **kwargs: Other optional arguments.

    See also:
       SenseRecon
    """

    def __init__(self, ksp, mps, eps,
                 weights=1, coord=None,
                 device=sp.util.cpu_device, **kwargs):
        ksp, mps, weights, coord = _move_to_device(
            ksp, mps, weights, coord, device)
        weights = _estimate_weights(ksp, weights, coord)

        A = linop.Sense(mps, coord=coord)
        proxg = sp.prox.L2Reg(A.ishape, 1)
        self.img = sp.util.zeros(mps.shape[1:], dtype=ksp.dtype, device=device)

        super().__init__(A, ksp, self.img, proxg, eps, weights=weights, **kwargs)


class L1WaveletRecon(sp.app.LinearLeastSquares):
    r"""L1 Wavelet regularized reconstruction.

    Considers the problem

    .. math:: \min_x \frac{1}{2} \| P F S x - y \|_2^2 + \lambda \| W x \|_1
    where P is the sampling operator, F is the Fourier transform operator,
    S is the SENSE operator, W is the wavelet operator,
    x is the image, and y is the k-space measurements.

    Args:
        ksp (array): k-space measurements.
        mps (array): sensitivity maps.
        lamda (float): regularization parameter.
        weights (float or array): weights for data consistency.
        coord (None or array): coordinates.
        wave_name (str): wavelet name.
        device (Device): device to perform reconstruction.
        **kwargs: Other optional arguments.

    References:
        Lustig, M., Donoho, D., & Pauly, J. M. (2007). 
        Sparse MRI: The application of compressed sensing for rapid MR imaging. 
        Magnetic Resonance in Medicine, 58(6), 1182-1195.

    """

    def __init__(self, ksp, mps, lamda,
                 weights=1, coord=None,
                 wave_name='db4', device=sp.util.cpu_device, **kwargs):
        ksp, mps, weights, coord = _move_to_device(
            ksp, mps, weights, coord, device)
        weights = _estimate_weights(ksp, weights, coord)

        A = linop.Sense(mps, coord=coord)
        self.img = sp.util.zeros(mps.shape[1:], dtype=ksp.dtype, device=device)

        W = sp.linop.Wavelet(A.ishape, wave_name=wave_name)
        proxg = sp.prox.UnitaryTransform(sp.prox.L1Reg(W.oshape, lamda), W)

        def g(input):
            device = sp.util.get_device(input)
            xp = device.xp

            with device:
                return lamda * xp.sum(abs(W(input)))

        super().__init__(A, ksp, self.img, proxg=proxg, g=g, weights=weights, **kwargs)


class L1WaveletConstrainedRecon(sp.app.L2ConstrainedMinimization):
    r"""L1 wavelet regularized constrained reconstruction.

    Considers the problem

    .. math::
        \min_x &\| W x \|_1 \\
        \text{s.t.} &\| P F S x - y \|_2^2 \le \epsilon
    where P is the sampling operator, F is the Fourier transform operator,
    S is the SENSE operator, W is the wavelet operator, 
    x is the image, and y is the k-space measurements.

    Args:
        ksp (array): k-space measurements.
        mps (array): sensitivity maps.
        eps (float): constraint parameter.
        wave_name (str): wavelet name.
        weights (float or array): weights for data consistency.
        coord (None or array): coordinates.
        device (Device): device to perform reconstruction.
        **kwargs: Other optional arguments.

    See also:
       :func:`sigpy.mri.app.WaveletRecon`

    """

    def __init__(
            self, ksp, mps, eps,
            wave_name='db4', weights=1, coord=None, device=sp.util.cpu_device, **kwargs):
        ksp, mps, weights, coord = _move_to_device(
            ksp, mps, weights, coord, device)
        weights = _estimate_weights(ksp, weights, coord)

        A = linop.Sense(mps, coord=coord)
        self.img = sp.util.zeros(mps.shape[1:], dtype=ksp.dtype, device=device)
        W = sp.linop.Wavelet(A.ishape, wave_name=wave_name)
        proxg = sp.prox.UnitaryTransform(sp.prox.L1Reg(W.oshape, 1), W)

        super().__init__(A, ksp, self.img, proxg, eps, weights=weights, **kwargs)


class TotalVariationRecon(sp.app.LinearLeastSquares):
    r"""Total variation regularized reconstruction.

    Considers the problem:
    .. math:: \min_x \frac{1}{2} \| P F S x - y \|_2^2 + \lambda \| G x \|_1
    where P is the sampling operator, F is the Fourier transform operator,
    S is the SENSE operator, G is the gradient operator,
    x is the image, and y is the k-space measurements.

    Args:
        ksp (array): k-space measurements.
        mps (array): sensitivity maps.
        lamda (float): regularization parameter.
        weights (float or array): weights for data consistency.
        coord (None or array): coordinates.
        device (Device): device to perform reconstruction.
        **kwargs: Other optional arguments.

    References:
        Block, K. T., Uecker, M., & Frahm, J. (2007). 
        Undersampled radial MRI with multiple coils. 
        Iterative image reconstruction using a total variation constraint. 
        Magnetic Resonance in Medicine, 57(6), 1086-1098.
    """

    def __init__(self, ksp, mps, lamda,
                 weights=1, coord=None, device=sp.util.cpu_device, **kwargs):
        ksp, mps, weights, coord = _move_to_device(
            ksp, mps, weights, coord, device)
        weights = _estimate_weights(ksp, weights, coord)

        A = linop.Sense(mps, coord=coord)
        self.img = sp.util.zeros(mps.shape[1:], dtype=ksp.dtype, device=device)

        G = sp.linop.Gradient(A.ishape)
        proxg = sp.prox.L1Reg(G.oshape, lamda)

        def g(input):
            xp = device.xp

            with device:
                return lamda * xp.sum(abs(G(input)))

        super().__init__(A, ksp, self.img, proxg=proxg, g=g, G=G, weights=weights, **kwargs)


class TotalVariationConstrainedRecon(sp.app.L2ConstrainedMinimization):
    r"""Total variation regularized constrained reconstruction.

    Considers the problem

    .. math::
        \min_x &\| G x \|_1 \\
        \text{s.t.} &\| P F S x - y \|_2^2 \le \epsilon
    where P is the sampling operator, F is the Fourier transform operator,
    S is the SENSE operator, G is the gradient operator,
    x is the image, and y is the k-space measurements.

    Args:
        ksp (array): k-space measurements.
        mps (array): sensitivity maps.
        eps (float): constraint parameter.
        weights (float or array): weights for data consistency.
        coord (None or array): coordinates.
        device (Device): device to perform reconstruction.
        **kwargs: Other optional arguments.

    See also:
       :func:`sigpy.mri.app.TotalVariationRecon`

    """

    def __init__(
            self, ksp, mps, eps,
            weights=1, coord=None, device=sp.util.cpu_device, **kwargs):
        ksp, mps, weights, coord = _move_to_device(
            ksp, mps, weights, coord, device)
        weights = _estimate_weights(ksp, weights, coord)

        A = linop.Sense(mps, coord=coord)
        self.img = sp.util.zeros(mps.shape[1:], dtype=ksp.dtype, device=device)
        G = sp.linop.Gradient(A.ishape)
        proxg = sp.prox.L1Reg(G.oshape, 1)

        super().__init__(A, ksp, self.img, proxg, eps, G=G, weights=weights, **kwargs)


class JsenseRecon(sp.app.App):
    r"""JSENSE reconstruction.

    Considers the problem 

    .. math:: 
        \min_{l, r} \frac{1}{2} \| l \ast r - y ||_2^2 + 
        \frac{\lambda}{2} (\| l \|_2^2 + \| r \|_2^2)
    where \ast is the convolution operator.

    Args:
        ksp (array): k-space measurements.
        mps_ker_width (int): sensitivity maps kernel width.
        ksp_calib_width (int): k-space calibration width.
        lamda (float): regularization parameter.
        device (Device): device to perform reconstruction.
        weights (float or array): weights for data consistency.
        coord (None or array): coordinates.
        max_iter (int): Maximum number of iterations.
        max_inner_iter (int): Maximum number of inner iterations.
        thresh (float): threshold parameter for output, from 0 to 1.

    References:
        Ying, L., & Sheng, J. (2007). 
        Joint image reconstruction and sensitivity estimation in SENSE (JSENSE). 
        Magnetic Resonance in Medicine, 57(6), 1196-1202.

        Uecker, M., Hohage, T., Block, K. T., & Frahm, J. (2008). 
        Image reconstruction by regularized nonlinear inversion—joint 
        estimation of coil sensitivities and image content. 
        Magnetic Resonance in Medicine, 60(3), 674-682.

    """

    def __init__(self, ksp,
                 mps_ker_width=12, ksp_calib_width=24,
                 lamda=0, device=sp.util.cpu_device,
                 weights=1, coord=None, max_iter=5,
                 max_inner_iter=5, thresh=0):

        self.ksp = ksp
        self.mps_ker_width = mps_ker_width
        self.ksp_calib_width = ksp_calib_width
        self.lamda = lamda
        self.weights = weights
        self.coord = coord
        self.max_iter = max_iter
        self.max_inner_iter = max_inner_iter
        self.thresh = thresh

        self.device = sp.util.Device(device)
        self.dtype = ksp.dtype
        self.num_coils = len(ksp)

        self._init_data()
        self._init_vars()
        self._init_model()
        self._init_alg()

    def _init_data(self):
        if self.coord is None:
            self.img_shape = self.ksp.shape[1:]
            ndim = len(self.img_shape)

            self.ksp = sp.util.resize(
                self.ksp, [self.num_coils] + ndim * [self.ksp_calib_width])

            if not np.isscalar(self.weights):
                self.weights = sp.util.resize(
                    self.weights, ndim * [self.ksp_calib_width])

        else:
            self.img_shape = sp.nufft.estimate_shape(self.coord)
            calib_idx = np.amax(np.abs(self.coord), axis=-
                                1) < self.ksp_calib_width / 2

            self.coord = self.coord[calib_idx]
            self.ksp = self.ksp[:, calib_idx]

            if not np.isscalar(self.weights):
                self.weights = self.weights[calib_idx]

        self.ksp = self.ksp / np.abs(self.ksp).max()
        self.ksp = sp.util.move(self.ksp, self.device)
        if self.coord is not None:
            self.coord = sp.util.move(self.coord, self.device)
        if not np.isscalar(self.weights):
            self.weights = sp.util.move(self.weights, self.device)

        self.weights = _estimate_weights(self.ksp, self.weights, self.coord)

    def _init_vars(self):
        ndim = len(self.img_shape)

        mps_ker_shape = [self.num_coils] + [self.mps_ker_width] * ndim
        if self.coord is None:
            img_ker_shape = [i + self.mps_ker_width -
                             1 for i in self.ksp.shape[1:]]
        else:
            grd_shape = sp.nufft.estimate_shape(self.coord)
            img_ker_shape = [i + self.mps_ker_width - 1 for i in grd_shape]

        self.img_ker = sp.util.dirac(
            img_ker_shape, dtype=self.dtype, device=self.device)
        self.mps_ker = sp.util.zeros(
            mps_ker_shape, dtype=self.dtype, device=self.device)

    def _init_model(self):
        self.A_img_ker = linop.ConvSense(
            self.img_ker.shape, self.mps_ker, coord=self.coord)

        self.A_mps_ker = linop.ConvImage(
            self.mps_ker.shape, self.img_ker, coord=self.coord)

    def _init_alg(self):
        self.app_mps = sp.app.LinearLeastSquares(
            self.A_mps_ker, self.ksp, self.mps_ker, weights=self.weights,
            lamda=self.lamda, max_iter=self.max_inner_iter)

        self.app_img = sp.app.LinearLeastSquares(
            self.A_img_ker, self.ksp, self.img_ker, weights=self.weights,
            lamda=self.lamda, max_iter=self.max_inner_iter)

        _alg = sp.alg.AltMin(
            self.app_mps.run, self.app_img.run, max_iter=self.max_iter)

        super().__init__(_alg)

    def _output(self):
        xp = self.device.xp
        # Coil by coil to save memory
        with self.device:
            mps_rss = 0
            mps = []
            for mps_ker_c in self.mps_ker:
                mps_c = sp.fft.ifft(sp.util.resize(mps_ker_c, self.img_shape))
                mps.append(sp.util.move(mps_c))
                mps_rss += xp.abs(mps_c)**2

            mps_rss = sp.util.move(mps_rss**0.5)
            mps = np.stack(mps)
            mps /= mps_rss

        return mps
