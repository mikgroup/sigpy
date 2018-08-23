"""Machine Learning Apps.
"""
import pickle
import logging
import sigpy as sp
        

class ConvSparseDecom(sp.app.LinearLeastSquares):
    r"""Convolutional sparse decomposition app.

    Considers the convolutional sparse linear model :math:`y_j = \sum_i l_i * r_{ij}`, 
    with :math:`l` fixed, and the problem,

    .. math:: 
        \min_{r_{ij}} \sum_j \frac{1}{2}\|y_j - \sum_i l_i * r_{ij}\|_2^2 + \lambda \|r_{ij}\|_1^2
    where :math:`y_j` is the jth data, :math:`l_i` is the ith filter, 
    :math:`r_{ij}` is the ith coefficient for jth data.

    Args:
        y_j (array): data array, the first dimension is the number of data.
            If multi_channel is True, then the second dimension should be the number of channels.
        l (array): filter array. If multi_channel is True,
            the first dimension is the number of filters. Otherwise, the first dimension
            is the number of channels, and second dimension is the number of filters.
        lamda (float): regularization parameter.
        mode (str): convolution mode in forward model. {'full', 'valid'}.
            If 'full', then r_j is smaller than y. If 'valid', then r_j is larger than y.
        multi_channel (bool): whether data is multi-channel or not.
        **kwargs: other LinearLeastSquares arguments.

    Returns:
        array: Coefficients. First two dimensions are number of data and number of filters.

    See Also:
        :func:`sigpy.app.LinearLeastSquares`

    """
    def __init__(self, y_j, l, lamda=1,
                 mode='full', multi_channel=False, device=sp.util.cpu_device, **kwargs):
        self.y_j = y_j
        self.l = l
        self.lamda = lamda
        self.mode = mode
        self.multi_channel = multi_channel
        self.device = device

        self._get_params()
        self.r_j = sp.util.empty(self.r_j_shape, dtype=self.dtype, device=device)
        self.A_r_j = sp.linop.ConvolveInput(self.r_j.shape, self.l, mode=self.mode,
                                            input_multi_channel=True,
                                            output_multi_channel=self.multi_channel)
        proxg_r_j = sp.prox.L1Reg(self.A_r_j.ishape, lamda)
        
        super().__init__(self.A_r_j, self.y_j, self.r_j, proxg=proxg_r_j, **kwargs)

    def _init(self):
        with self.device:
            self.r_j.fill(0)
            
        super()._init()

    def _get_params(self):
        self.device = sp.util.Device(self.device)
        self.dtype = self.y_j.dtype
        self.batch_size = len(self.y_j)
        self.num_filters = self.l.shape[self.multi_channel]
        self.data_ndim = self.y_j.ndim - self.multi_channel - 1

        if self.mode == 'full':
            self.r_j_shape = ([self.batch_size, self.num_filters] +
                              [i - min(i, f) + 1 for i, f in zip(self.y_j.shape[-self.data_ndim:],
                                                                 self.l.shape[-self.data_ndim:])])
        else:
            self.r_j_shape = ([self.batch_size, self.num_filters] +
                              [i + min(i, f) - 1 for i, f in zip(self.y_j.shape[-self.data_ndim:],
                                                                 self.l.shape[-self.data_ndim:])])

        self.r_j_shape = tuple(self.r_j_shape)


class ConvSparseCoding(sp.app.App):
    r"""Convolutional sparse coding application.

    Considers the convolutional sparse bilinear model :math:`y_j = \sum_i l_i * r_{ij}`,
    and the objective function

    .. math:: 
        f(l, c) = 
        \sum_j \frac{1}{2} \|y_j - \sum_i l_i * r_{ij}\|_2^2 
        + \frac{\lambda}{2} \sum_i (\|l_i\|_2^2 + \|r_{ij}\|_1^2)
    where :math:`y_j` is the jth data, :math:`l_i` is the ith filter, 
    :math:`r_{ij}` is the ith coefficient for jth data.

    Args:
        data (array): data array, the first dimension is the number of data.
            If multi_channel is True, then the second dimension should be the number of channels.
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
                 lamda=1, alpha=1, init_scale=1e-5,
                 max_l_iter=30, max_r_j_iter=50, max_power_iter=10, max_epoch=1,
                 mode='full', multi_channel=False, device=sp.util.cpu_device,
                 checkpoint_filepath=None):
        self.y = y
        self.num_filters = num_filters
        self.filt_width = filt_width
        self.batch_size = batch_size
        self.lamda = lamda
        self.alpha = alpha
        self.init_scale = init_scale
        self.max_l_iter = max_l_iter
        self.max_r_j_iter = max_r_j_iter
        self.max_power_iter = max_power_iter
        self.max_epoch = max_epoch
        self.mode = mode
        self.multi_channel = multi_channel
        self.device = device
        self.checkpoint_filepath = checkpoint_filepath

        self._get_params()
        self._get_batch_vars()
        self.l = sp.util.empty(self.l_shape, dtype=self.dtype, device=self.device)
        self.l_old = sp.util.empty(self.l_shape, dtype=self.dtype, device=self.device)
        self._get_alg()

    def _init(self):
        sp.util.move_to(self.l, sp.util.randn_like(self.l))
        xp = self.device.xp
        with self.device:
            if self.multi_channel:
                l_norm = sp.util.norm(self.l, axes=[0] + list(range(-self.data_ndim, 0)),
                                      keepdims=True)
            else:
                l_norm = sp.util.norm(self.l, axes=range(-self.data_ndim, 0), keepdims=True)

            self.l /= l_norm

    def _pre_update(self):
        j = self.j_idx.next()
        j_start = j * self.batch_size
        j_end = (j + 1) * self.batch_size

        sp.util.move_to(self.y_j, self.y[j_start:j_end])
        sp.util.move_to(self.l_old, self.l)

    def _summarize(self):
        xp = self.device.xp
        if self.checkpoint_filepath is not None:
            xp.save(self.checkpoint_filepath, self.l)

    def _output(self):
        r = ConvSparseCoefficients(self.y, self.l, lamda=self.lamda,
                                   multi_channel=self.multi_channel,
                                   mode=self.mode, max_iter=self.max_r_j_iter,
                                   max_power_iter=self.max_power_iter)
        return self.l, r

    def _get_params(self):
        self.device = sp.util.Device(self.device)
        self.dtype = self.y.dtype
        self.num_batches = len(self.y) // self.batch_size
        self.data_ndim = self.y.ndim - self.multi_channel - 1

        self.l_shape = ([self.num_filters] +
                        [min(d, self.filt_width) for d in self.y.shape[-self.data_ndim:]])
        
        if self.multi_channel:
            self.l_shape = [self.y.shape[1]] + self.l_shape

        self.l_shape = tuple(self.l_shape)

    def _get_batch_vars(self):
        self.j_idx = sp.index.ShuffledIndex(self.num_batches)
        self.y_j = sp.util.empty((self.batch_size, ) + self.y.shape[1:],
                                 dtype=self.dtype, device=self.device)

    def _get_alg(self):
        min_r_j_app = ConvSparseDecom(self.y_j, self.l, lamda=self.lamda,
                                      mode=self.mode, multi_channel=self.multi_channel,
                                      max_power_iter=self.max_power_iter,
                                      max_iter=self.max_r_j_iter, device=self.device)
        self.r_j = min_r_j_app.r_j

        self.A_l = sp.linop.ConvolveFilter(self.l_shape, self.r_j, mode=self.mode,
                                           input_multi_channel=True,
                                           output_multi_channel=self.multi_channel)
        if self.multi_channel:
            proxg_l = sp.prox.L2Proj(self.l_shape, 1, axes=[0] + list(range(-self.data_ndim, 0)))
        else:
            proxg_l = sp.prox.L2Proj(self.l_shape, 1, axes=range(-self.data_ndim, 0))
            
        min_l_app = sp.app.LinearLeastSquares(self.A_l, self.y_j, self.l,
                                              mu=1 / (self.alpha * self.num_batches), z=self.l_old,
                                              max_power_iter=self.max_power_iter,
                                              max_iter=self.max_l_iter, proxg=proxg_l)

        max_iter = self.max_epoch * self.num_batches
        self.alg = sp.alg.AltMin(min_r_j_app.run, min_l_app.run, max_iter=max_iter)

    
class LinearRegression(sp.app.App):
    r"""Performs linear regression to fit input to output.

    Considers the linear model :math:`y_j = M x_j`, and the problem,

    .. math::
        \min_M \sum_j \frac{1}{2} \| y_j - M x_j \|_2^2
    where :math:`y_j` is the jth output, :math:`M` is the learned matrix,
    and :math:`x_j` is the jth input.

    It uses the randomized block Kaczmarz method.

    Args:
        input (array): input data of shape (num_data, ...).
        output (array): output data of shape (num_data, ...).
        batch_size (int): batch size.
        alpha (float): step size.

    Returns:
       array: matrix of shape input.shape[1:] + output.shape[1:].

    References:
        Needell, Deanna, Ran Zhao, and Anastasios Zouzias. 
        Randomized block Kaczmarz method with projection for solving least squares.
        Linear Algebra and its Applications 484 (2015): 322-343.

    """
    def __init__(self, input, output, batch_size, alpha,
                 max_epoch=1, max_inner_iter=100, device=sp.util.cpu_device,
                 checkpoint_filepath=None):
        dtype = output.dtype

        num_data = len(output)
        num_batches = num_data // batch_size
        self.batch_size = batch_size
        self.input = input
        self.output = output
        self.checkpoint_filepath = checkpoint_filepath

        self.mat = sp.util.zeros(input.shape[1:] + output.shape[1:], dtype=dtype, device=device)
        
        self.j_idx = sp.index.ShuffledIndex(num_batches)
        self.input_j = sp.util.empty((batch_size, ) + input.shape[1:],
                                     dtype=dtype, device=device)
        self.output_j = sp.util.empty((batch_size, ) + output.shape[1:],
                                      dtype=dtype, device=device)
        
        self._get_A()
        def proxf(alpha, x):
            app = sp.app.LinearLeastSquares(self.A, self.output_j, x,
                                            mu=1 / (alpha * num_batches), z=x,
                                            max_iter=max_inner_iter)
            return app.run()

        max_iter = max_epoch * num_batches
        alg = sp.alg.ProximalPointMethod(proxf, alpha, self.mat, max_iter=max_iter)
        super().__init__(alg)
        
    def _pre_update(self):
        j = self.j_idx.next()
        j_start = j * self.batch_size
        j_end = (j + 1) * self.batch_size
        
        sp.util.move_to(self.input_j, self.input[j_start:j_end])
        sp.util.move_to(self.output_j, self.output[j_start:j_end])

    def _summarize(self):
        xp = self.device.xp
        if self.checkpoint_filepath is not None:
            xp.save(self.checkpoint_filepath, self.mat)

    def _output(self):
        return self.mat
    
    def _get_A(self):
        input_j_size = sp.util.prod(self.input_j.shape[1:])
        output_j_size = sp.util.prod(self.output_j.shape[1:])

        Ri = sp.linop.Reshape([input_j_size, output_j_size], self.mat.shape)
        M = sp.linop.MatMul([input_j_size, output_j_size],
                            self.input_j.reshape([self.batch_size, -1]))
        Ro = sp.linop.Reshape(self.output_j.shape, [self.batch_size, output_j_size])
        self.A = Ro * M * Ri


class ConvSparseCoefficients(object):
    r"""Convolutional sparse coefficients.

    Generates coefficients on the fly using convolutional sparse decomposition.
    ConvSparseCoefficients can be sliced like arrays.

    Args:
        data (array): data array, the first dimension is the number of data.
            If multi_channel is True, then the second dimension should be the number of channels.
        l (array): filter. If multi_channel is True,
            the first dimension is the number of filters. Otherwise, the first dimension
            is the number of channels, and second dimension is the number of filters.
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
    def __init__(self, y, l,
                 lamda=1, multi_channel=False, mode='full',
                 max_iter=100, max_power_iter=10, device=sp.util.cpu_device):
        
        self.y = y
        self.l = l
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
        self.device = sp.util.Device(device)
        
    def __getitem__(self, index):
        if isinstance(index, int):
            y_j = self.y[index:(index + 1)]
            index = 0
        elif isinstance(index, slice):
            y_j = self.y[index]
            index = slice(None)
        elif isinstance(index, tuple) or isinstance(index, list):
            if isinstance(index[0], int):
                y_j = self.y[index[0]:(index[0] + 1)]
                index = [0] + list(index[1:])
            else:
                y_j = self.y[index[0]]
                index = [slice(None)] + list(index[1:])

        y_j = sp.util.move(y_j, self.device)
        l = sp.util.move(self.l, self.device)
        app_j = ConvSparseDecom(y_j, l, lamda=self.lamda,
                                multi_channel=self.multi_channel, mode=self.mode,
                                max_iter=self.max_iter, max_power_iter=self.max_power_iter,
                                device=self.device)
        r_j = app_j.run()

        with self.device:
            return r_j[index]

    def __len__(self):
        return self.num_data
        
    def _get_params(self):
        self.num_data = len(self.y)
        if self.multi_channel:
            self.num_filters = self.l.shape[1]
            self.data_ndim = self.y.ndim - 2
        else:
            self.num_filters = self.l.shape[0]
            self.data_ndim = self.y.ndim - 1

        if self.mode == 'full':
            self.shape = tuple([self.num_data, self.num_filters] +
                               [i - min(i, f) + 1 for i, f in zip(self.y.shape[-self.data_ndim:],
                                                                  self.l.shape[-self.data_ndim:])])
        else:
            self.shape = tuple([self.num_data, self.num_filters] +
                               [i + min(i, f) - 1 for i, f in zip(self.y.shape[-self.data_ndim:],
                                                                  self.l.shape[-self.data_ndim:])])

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
