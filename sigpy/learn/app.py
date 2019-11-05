# -*- coding: utf-8 -*-
"""Machine learning applications.
"""
import pickle
import pathlib
import sigpy as sp


class ConvSparseDecom(sp.app.LinearLeastSquares):
    r"""Convolutional sparse decomposition app.

    Considers the convolutional sparse linear model
    :math:`y = \sum_j L_j * R_j`,
    with :math:`L` fixed, and the problem,

    .. math::
        \min_{R} \frac{1}{2}\|y - \sum_j L_j * R_j\|_2^2 + \lambda \|R\|_1

    Args:
        y (array): data array, the first dimension is the number of data.
            If multi_channel is True,
            then the second dimension should be the number of channels.
        L (array): filter array. If multi_channel is True,
            the first dimension is the number of filters.
            Otherwise, the first dimension
            is the number of channels, and second dimension
            is the number of filters.
        lamda (float): regularization parameter.
        mode (str): convolution mode in forward model. {'full', 'valid'}.
            If 'full', then R is smaller than y. If 'valid',
            then R is larger than y.
        multi_channel (bool): whether data is multi-channel or not.
        **kwargs: other LinearLeastSquares arguments.

    Returns:
        array: Coefficients. First two dimensions are number of data
        and number of filters.

    See Also:
        :func:`sigpy.app.LinearLeastSquares`

    """

    def __init__(
            self,
            y,
            L,
            lamda=0.005,
            mode='full',
            multi_channel=False,
            device=sp.cpu_device,
            **kwargs):
        self.y = sp.to_device(y, device)
        self.L = sp.to_device(L, device)
        self.lamda = lamda
        self.mode = mode
        self.multi_channel = multi_channel
        self.device = device

        self._get_params()
        self.A_R = sp.linop.ConvolveInput(
            self.R_shape,
            self.L,
            mode=self.mode,
            input_multi_channel=True,
            output_multi_channel=self.multi_channel)

        proxg_R = sp.prox.L1Reg(self.R_shape, lamda)
        super().__init__(self.A_R, self.y, proxg=proxg_R, **kwargs)

    def _get_params(self):
        self.device = sp.Device(self.device)
        self.dtype = self.y.dtype
        self.num_data = len(self.y)
        self.filt_width = self.L.shape[-1]
        self.num_filters = self.L.shape[self.multi_channel]
        self.data_ndim = self.y.ndim - self.multi_channel - 1

        if self.mode == 'full':
            self.R_shape = (
                [self.num_data, self.num_filters] +
                [i - self.filt_width + 1
                 for i in self.y.shape[-self.data_ndim:]])
        else:
            self.R_shape = (
                [self.num_data, self.num_filters] +
                [i + self.filt_width - 1
                 for i in self.y.shape[-self.data_ndim:]])


class ConvSparseCoding(sp.app.App):
    r"""Convolutional sparse coding application.

    Considers the convolutional sparse model
    :math:`y_t = \sum_j L_j * R_{tj}`,
    and the objective function

    .. math::
        f(L, R) = \sum_t \frac{1}{2} \|y_t - \sum_j L_j * R_{tj}\|_2^2 +
        \lambda \|R\|_1

    where :math:`y_t` is the tth data,
    :math:`L_j` is the jth filter constrained to have unit norm,
    and :math:`R_{tj}` is the jth coefficient for t th data.

    Args:
        y (array): data array, the first dimension is the number of data.
            If multi_channel is True,
            then the second dimension should be the number of channels.
        num_filters (int): number of filters.
        filt_width (int): filter width.
        batch_size (int): batch size.
        lamda (float): regularization parameter.
        alpha (float): step-size.
        max_inner_iter (int): maximum number of iteration for inner-loop.
        max_power_iter (int): maximum number of iteration for power method.
        mode (str): convolution mode in forward model. {'full', 'valid'}.
        multi_channel (bool): whether data is multi-channel or not.
        **kwargs: other LinearLeastSquares arguments.

    Returns:
        array: Filters.

    See Also:
        :func:`sigpy.learn.app.ConvSparseDecom`

    References:
        TODO

    """

    def __init__(self, y, num_filters, filt_width, batch_size,
                 lamda=0.001, alpha=0.5,
                 max_inner_iter=100,
                 max_power_iter=10,
                 max_iter=10,
                 mode='full',
                 multi_channel=False,
                 device=sp.cpu_device,
                 checkpoint_path=None, show_pbar=True):
        self.y = y
        self.num_filters = num_filters
        self.filt_width = filt_width
        self.batch_size = batch_size
        self.lamda = lamda
        self.alpha = alpha
        self.max_inner_iter = max_inner_iter
        self.max_power_iter = max_power_iter
        self.max_iter = max_iter
        self.mode = mode
        self.multi_channel = multi_channel
        self.device = device
        self.checkpoint_path = checkpoint_path

        self._get_params()
        self._get_vars()
        self._get_alg()

        super().__init__(self.alg, show_pbar=show_pbar)

    def _get_params(self):
        self.device = sp.Device(self.device)
        self.dtype = self.y.dtype
        self.data_ndim = self.y.ndim - self.multi_channel - 1
        if self.checkpoint_path is not None:
            self.checkpoint_path = pathlib.Path(self.checkpoint_path)
            self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        self.batch_size = min(len(self.y), self.batch_size)
        self.num_batches = len(self.y) // self.batch_size

        self.L_shape = [self.num_filters] + [self.filt_width] * self.data_ndim
        if self.multi_channel:
            self.L_shape = [self.y.shape[1]] + self.L_shape

        if self.mode == 'full':
            self.R_t_shape = (
                [self.batch_size, self.num_filters] +
                [i - self.filt_width + 1
                 for i in self.y.shape[-self.data_ndim:]])
        else:
            self.R_t_shape = (
                [self.batch_size, self.num_filters] +
                [i + self.filt_width - 1
                 for i in self.y.shape[-self.data_ndim:]])

    def _get_vars(self):
        self.t_idx = sp.ShuffledNumbers(self.num_batches)
        xp = self.device.xp
        with self.device:
            self.y_t = xp.empty((self.batch_size, ) +
                                self.y.shape[1:], dtype=self.dtype)
            self.L = sp.randn(self.L_shape, dtype=self.dtype,
                              device=self.device)
            if self.multi_channel:
                self.L /= xp.sum(xp.abs(self.L)**2, axis=(0, ) +
                                 tuple(range(-self.data_ndim, 0)),
                                 keepdims=True)**0.5
            else:
                self.L /= xp.sum(xp.abs(self.L)**2,
                                 axis=tuple(range(-self.data_ndim,
                                                  0)),
                                 keepdims=True)**0.5

            self.L_old = xp.empty(self.L_shape, dtype=self.dtype)
            self.R = ConvSparseCoefficients(
                self.y,
                self.L,
                lamda=self.lamda,
                multi_channel=self.multi_channel,
                mode=self.mode,
                max_iter=self.max_inner_iter,
                max_power_iter=self.max_power_iter)

    def _get_alg(self):
        def min_R_t():
            self.R_t = ConvSparseDecom(
                self.y_t,
                self.L,
                lamda=self.lamda,
                mode=self.mode,
                multi_channel=self.multi_channel,
                max_power_iter=self.max_power_iter,
                max_iter=self.max_inner_iter,
                device=self.device).run()

        def min_L():
            self.A_L = sp.linop.ConvolveFilter(
                self.L_shape,
                self.R_t,
                mode=self.mode,
                input_multi_channel=True,
                output_multi_channel=self.multi_channel)

            mu = (1 - self.alpha) / self.alpha
            if self.multi_channel:
                proxg_L = sp.prox.L2Proj(self.L.shape, 1, axes=[
                                         0] + list(range(-self.data_ndim, 0)))
            else:
                proxg_L = sp.prox.L2Proj(
                    self.L.shape, 1, axes=range(-self.data_ndim, 0))

            sp.app.LinearLeastSquares(self.A_L, self.y_t, x=self.L,
                                      mu=mu, z=self.L_old,
                                      proxg=proxg_L,
                                      max_power_iter=self.max_power_iter,
                                      max_iter=self.max_inner_iter).run()

        self.alg = sp.alg.AltMin(min_R_t, min_L, max_iter=self.max_iter)

    def _pre_update(self):
        t = self.t_idx.next()
        t_start = t * self.batch_size
        t_end = (t + 1) * self.batch_size

        sp.copyto(self.y_t, self.y[t_start:t_end])
        sp.copyto(self.L_old, self.L)

    def _summarize(self):
        if self.checkpoint_path is not None:
            xp = self.device.xp
            with self.device:
                xp.save(str(self.checkpoint_path / 'L.npy'), self.L)
                xp.save(str(self.checkpoint_path / 'R_t.npy'), self.R_t)
                x = sp.convolve(
                    self.R_t,
                    self.L,
                    mode=self.mode,
                    input_multi_channel=True,
                    output_multi_channel=self.multi_channel)
                xp.save(str(self.checkpoint_path / 'x_t.npy'), x)
                xp.save(str(self.checkpoint_path / 'y_t.npy'), self.y_t)

    def _output(self):
        R = ConvSparseCoefficients(
            self.y,
            self.L,
            lamda=self.lamda,
            multi_channel=self.multi_channel,
            mode=self.mode,
            max_iter=self.max_inner_iter,
            max_power_iter=self.max_power_iter)
        return self.L, R


class LinearRegression(sp.app.App):
    r"""Performs linear regression to fit input to output.

    Considers the linear model :math:`y_t = M x_t`, and the problem,

    .. math::
        \min_M \sum_t \frac{1}{2} \| y_t - M x_t \|_2^2

    where :math:`y_t` is the tth output, :math:`M` is the learned matrix,
    and :math:`x_t` is the tth input.

    It uses the randomized block Kaczmarz method.

    Args:
        input (array): input data of shape (num_data, ...).
        output (array): output data of shape (num_data, ...).
        batch_size (int): batch size.
        mu (float): step size.

    Returns:
       array: matrix of shape input.shape[1:] + output.shape[1:].

    References:
        Needell, Deanna, Ran Zhao, and Anastasios Zouzias.
        Randomized block Kaczmarz method with projection
        for solving least squares.
        Linear Algebra and its Applications 484 (2015): 322-343.

    """

    def __init__(self, input, output, batch_size, mu,
                 lamda=0,
                 max_iter=100, max_inner_iter=100, device=sp.cpu_device,
                 checkpoint_path=None):
        dtype = output.dtype

        num_data = len(output)
        num_batches = num_data // batch_size
        self.device = device
        self.lamda = lamda
        self.batch_size = batch_size
        self.input = input
        self.output = output
        self.checkpoint_path = checkpoint_path

        self.device = sp.Device(device)
        xp = self.device.xp
        with self.device:
            self.mat = xp.zeros(
                input.shape[1:] + output.shape[1:], dtype=dtype)
            self.input_t = xp.empty(
                (batch_size, ) + input.shape[1:], dtype=dtype)
            self.output_t = xp.empty(
                (batch_size, ) + output.shape[1:], dtype=dtype)
            self.t_idx = sp.ShuffledNumbers(num_batches)

        self._get_A()

        def proxf(mu, x):
            return sp.app.LinearLeastSquares(self.A, self.output_t, x=x,
                                             lamda=self.lamda / num_batches,
                                             mu=1 / mu, z=x,
                                             max_iter=max_inner_iter).run()

        alg = sp.alg.ProximalPointMethod(
            proxf, mu, self.mat, max_iter=max_iter)
        super().__init__(alg)

    def _pre_update(self):
        t = self.t_idx.next()
        t_start = t * self.batch_size
        t_end = (t + 1) * self.batch_size

        sp.copyto(self.input_t, self.input[t_start:t_end])
        sp.copyto(self.output_t, self.output[t_start:t_end])

    def _summarize(self):
        xp = self.device.xp
        if self.checkpoint_path is not None:
            xp.save(self.checkpoint_path, self.mat)

    def _output(self):
        return self.mat

    def _get_A(self):
        input_t_size = sp.prod(self.input_t.shape[1:])
        output_t_size = sp.prod(self.output_t.shape[1:])

        Ri = sp.linop.Reshape([input_t_size, output_t_size], self.mat.shape)
        M = sp.linop.MatMul([input_t_size, output_t_size],
                            self.input_t.reshape([self.batch_size, -1]))
        Ro = sp.linop.Reshape(self.output_t.shape, [
                              self.batch_size, output_t_size])
        self.A = Ro * M * Ri


class ConvSparseCoefficients(object):
    r"""Convolutional sparse coefficients.

    Generates coefficients on the fly using convolutional
    sparse decomposition.
    ConvSparseCoefficients can be sliced like arrays.

    Args:
        data (array): data array, the first dimension is the number of data.
            If multi_channel is True, then the second dimension
            should be the number of channels.
        L (array): filter. If multi_channel is True,
            the first dimension is the number of filters.
            Otherwise, the first dimension
            is the number of channels,
            and second dimension is the number of filters.
        lamda (float): regularization parameter.
        mode (str): convolution mode in forward model. {'full', 'valid'}.
        multi_channel (bool): whether data is multi-channel or not.
        max_iter (bool): maximum number of iterations.
        max_power_iter (bool): maximum number of power iterations.

    Attributes:
        shape (tuple of ints): coefficient shape.
        ndim (int): number of dimensions of coefficient.
        dtype (Dtype): Data type.

    """

    def __init__(self, y, L,
                 lamda=1, multi_channel=False, mode='full',
                 max_iter=100, max_power_iter=10,
                 device=sp.cpu_device):

        self.y = y
        self.L = L
        self.lamda = lamda
        self.multi_channel = multi_channel
        self.mode = mode
        self.max_iter = max_iter
        self.max_power_iter = max_power_iter

        self._get_params()
        self.ndim = len(self.shape)
        self.dtype = y.dtype
        self.use_device(device)

    def use_device(self, device):
        self.device = sp.Device(device)

    def __getitem__(self, index):
        if isinstance(index, int):
            y_t = self.y[index:(index + 1)]
            index = 0
        elif isinstance(index, slice):
            y_t = self.y[index]
            index = slice(None)
        elif isinstance(index, tuple) or isinstance(index, list):
            if isinstance(index[0], int):
                y_t = self.y[index[0]:(index[0] + 1)]
                index = [0] + list(index[1:])
            else:
                y_t = self.y[index[0]]
                index = [slice(None)] + list(index[1:])

        with self.device:
            return ConvSparseDecom(
                y_t,
                self.L,
                lamda=self.lamda,
                multi_channel=self.multi_channel,
                mode=self.mode,
                max_iter=self.max_iter,
                max_power_iter=self.max_power_iter,
                device=self.device).run()[index]

    def __len__(self):
        return self.num_data

    def _get_params(self):
        self.num_data = len(self.y)
        if self.multi_channel:
            self.num_filters = self.L.shape[1]
            self.data_ndim = self.y.ndim - 2
        else:
            self.num_filters = self.L.shape[0]
            self.data_ndim = self.y.ndim - 1

        if self.mode == 'full':
            self.shape = tuple(
                [self.num_data, self.num_filters] + [
                    i - min(i, f) + 1 for i, f in zip(
                        self.y.shape[-self.data_ndim:],
                        self.L.shape[-self.data_ndim:])])
        else:
            self.shape = tuple(
                [self.num_data, self.num_filters] + [
                    i + min(i, f) - 1 for i, f in zip(
                        self.y.shape[-self.data_ndim:],
                        self.L.shape[-self.data_ndim:])])

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
