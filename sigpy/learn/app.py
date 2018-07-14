'''
Machine Learning Apps.
'''
import pickle
import logging
import sigpy as sp
        

class ConvSparseDecom(sp.app.LinearLeastSquares):
    r"""Convolutional sparse decomposition app.

    Considers the convolutional sparse linear model :math:`y_j = \sum_i c_{ij} * \psi_i`, 
    with $\psi$ fixed, and the problem,

    .. math:: 
        \min_{c_{ij}} \sum_j \frac{1}{2}\|y_j - \sum_i c_{ij} * \psi_i\|_2^2 
        + \lambda \|c_{ij}\|_1
    where :math:`y_j` is the jth data, :math:`\psi_i` is the ith filter, 
    :math:`c_{ij}` is the ith coefficient for jth data.

    Args:
        data (array): data array, the first dimension is the number of data.
            If multi_channel is True, then the second dimension should be the number of channels.
        filt (array): filter. If multi_channel is True,
            the first dimension is the number of filters. Otherwise, the first dimension
            is the number of channels, and second dimension is the number of filters.
        lamda (float): regularization parameter.
        mode (str): convolution mode in forward model. {'full', 'valid'}.
        multi_channel (bool): whether data is multi-channel or not.
        **kwargs: other LinearLeastSquares arguments.

    Returns:
        array: Coefficients. First two dimensions are number of data and number of filters.

    See Also:
        :func:`sigpy.app.LinearLeastSquares`

    """

    def __init__(self, data, filt, lamda=0.001,
                 mode='full', multi_channel=False,
                 device=sp.util.cpu_device, **kwargs):

        if multi_channel:
            num_filters = filt.shape[1]
        else:
            num_filters = filt.shape[0]
            
        filt_width = filt.shape[-1]

        if multi_channel:
            ndim = len(data.shape) - 2
        else:
            ndim = len(data.shape) - 1
        
        num_data = len(data)
        coef_shape = _get_csc_coef_shape(data.shape, num_data, num_filters,
                                         filt_width, mode, multi_channel)
        self.coef = sp.util.zeros(coef_shape, dtype=data.dtype, device=device)
        
        A_coef = _get_csc_A_coef(self.coef, filt, mode, multi_channel)
        proxg = sp.prox.L1Reg(A_coef.ishape, lamda)
        
        super().__init__(A_coef, data, self.coef, proxg=proxg, **kwargs)


class ConvSparseCoding(sp.app.App):
    r"""Convolutional sparse coding application.

    Considers the convolutional sparse bi-linear model :math:`y_j = \sum_i c_{ij} * \psi_i`,
    and the objective function

    .. math:: 
        f(\psi, c) = 
        \sum_j \frac{1}{2} \|y_j - \sum_i c_{ij} * \psi_i\|_2^2 + \lambda \|c_{ij}\|_1
        + 1\{\| \psi_i \|_2 \leq 1\}
    where :math:`y_j` is the jth data, :math:`\psi_i` is the ith filter, 
    :math:`c_{ij}` is the ith coefficient for jth data.

    Args:
        data (array): data array, the first dimension is the number of data.
            If multi_channel is True, then the second dimension should be the number of channels.
        num_filters (int): number of filters.
        filt_width (int): filter widith.
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

    """

    def __init__(self, data, num_filters, filt_width, batch_size,
                 lamda=0.001, alpha=1, output_iter=False,
                 max_inner_iter=100, max_power_iter=10, max_iter=100,
                 mode='full', multi_channel=False, device=sp.util.cpu_device):
    
        dtype = data.dtype
        self.data = data
        self.output_iter = output_iter
        if output_iter:
            self.filt_iter = []
        
        self.batch_size = batch_size
        num_batches = len(data) // batch_size
        self.j_idx = sp.index.ShuffledIndex(num_batches)
        self.data_j = sp.util.empty((batch_size, ) + data.shape[1:], dtype=dtype, device=device)
        
        filt_shape = _get_csc_filt_shape(data.shape, num_filters, filt_width, mode, multi_channel)
        self.filt = sp.util.empty(filt_shape, dtype=dtype, device=device)

        self.proxg = sp.prox.L2Proj(self.filt.shape, 1, axes=tuple(range(1, self.filt.ndim)))

        min_coef_j_app = ConvSparseDecom(self.data_j, self.filt, lamda=lamda,
                                         mode=mode, multi_channel=multi_channel,
                                         max_power_iter=max_power_iter,
                                         max_iter=max_inner_iter)
        self.coef_j = min_coef_j_app.x
        
        A_filt = _get_csc_A_filt(self.coef_j, self.filt, mode, multi_channel)
        update_filt = _get_csc_update_filt(A_filt, self.filt, self.data_j,
                                           num_batches, alpha, self.proxg)

        alg = sp.alg.AltMin(min_coef_j_app.run, update_filt, max_iter=max_iter)

        super().__init__(alg)

    def _init(self):
        sp.util.move_to(self.filt, self.proxg(1, sp.util.randn_like(self.filt)))

    def _pre_update(self):
        j = self.j_idx.next()
        j_start = j * self.batch_size
        j_end = (j + 1) * self.batch_size
        
        sp.util.move_to(self.data_j, self.data[j_start:j_end])
        
        with sp.util.get_device(self.coef_j):
            self.coef_j.fill(0)

    def _post_update(self):
        if self.output_iter:
            self.filt_iter.append(sp.util.move(self.filt).copy())

    def _output(self):
        if self.output_iter:
            return self.filt_iter
        else:
            return self.filt

    
class LinearRegression(sp.app.LinearLeastSquares):
    r"""Performs linear regression to fit input to output.

    Considers the linear model :math:`y_j = M x_j`, and the problem,

    .. math::
        \min_M \sum_j \frac{1}{2} \| y_j - M x_j \|_2^2
    where :math:`y_j` is the jth output, :math:`M` is the learned matrix,
    and :math:`x_j` is the jth input.

    Args:
        input (array): input data of shape (num_data, ...).
        output (array): output data of shape (num_data, ...).
        batch_size (int): batch size.
        alpha (float): step size.

    Returns:
       array: matrix of shape input.shape[1:] + output.shape[1:].

    """

    def __init__(self, input, output, batch_size, alpha,
                 max_iter=100, device=sp.util.cpu_device, **kwargs):
        
        dtype = output.dtype

        num_data = len(output)
        num_batches = num_data // batch_size
        self.batch_size = batch_size
        self.input = input
        self.output = output

        mat = sp.util.zeros(input.shape[1:] + output.shape[1:], dtype=dtype, device=device)
        
        self.j_idx = sp.index.ShuffledIndex(num_batches)
        self.input_j = sp.util.empty((batch_size, ) + input.shape[1:],
                                     dtype=dtype, device=device)
        self.output_j = sp.util.empty((batch_size, ) + output.shape[1:],
                                      dtype=dtype, device=device)
        
        A = _get_lr_A(self.input_j, self.output_j, mat, batch_size)
        
        super().__init__(A, self.output_j, mat, alg_name='GradientMethod', accelerate=False,
                         alpha=alpha, max_iter=max_iter)
        
    def _pre_update(self):
        j = self.j_idx.next()
        j_start = j * self.batch_size
        j_end = (j + 1) * self.batch_size
        
        sp.util.move_to(self.input_j, self.input[j_start:j_end])
        sp.util.move_to(self.output_j, self.output[j_start:j_end])

        
def _get_lr_A(input_j, output_j, mat, batch_size):
    input_j_size = sp.util.prod(input_j.shape[1:])
    output_j_size = sp.util.prod(output_j.shape[1:])
    
    Ri = sp.linop.Reshape([input_j_size, output_j_size], mat.shape)
    M = sp.linop.MatMul([input_j_size, output_j_size], input_j.reshape([batch_size, -1]))
    Ro = sp.linop.Reshape(output_j.shape, [batch_size, output_j_size])

    A = Ro * M * Ri

    return A


class ConvSparseCoefficients(object):
    r"""Convolutional sparse coefficients.

    Generates coefficients on the fly using convolutional sparse decomposition.
    ConvSparseCoefficients can be sliced like arrays.

    Args:
        data (array): data array, the first dimension is the number of data.
            If multi_channel is True, then the second dimension should be the number of channels.
        filt (array): filter. If multi_channel is True,
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

    """

    def __init__(self, data, filt,
                 lamda=0.001, multi_channel=False, mode='full',
                 max_iter=100, max_power_iter=10, device=sp.util.cpu_device):
        
        self.data = data
        self.filt = filt
        self.lamda = lamda
        self.multi_channel = multi_channel
        self.mode = mode
        self.max_iter = max_iter
        self.max_power_iter = max_power_iter

        if multi_channel:
            num_filters = filt.shape[1]
        else:
            num_filters = filt.shape[0]
            
        filt_width = filt.shape[-1]
        self.shape = _get_csc_coef_shape(
            data.shape, len(data), num_filters, filt_width, mode, multi_channel)
        self.ndim = len(self.shape)
        self.device = sp.util.Device(device)
        
    def __getitem__(self, slc):
        if isinstance(slc, int):
            data_j = self.data[slc:(slc + 1)]
            slc = 0

        elif isinstance(slc, slice):
            data_j = self.data[slc]
            slc = slice(None)

        elif isinstance(slc, tuple) or isinstance(slc, list):
            if isinstance(slc[0], int):
                data_j = self.data[slc[0]:(slc[0] + 1)]
                slc = [0] + list(slc[1:])
            else:
                data_j = self.data[slc[0]]
                slc = [slice(None)] + list(slc[1:])

        data_j = sp.util.move(data_j, self.device)
        filt = sp.util.move(self.filt, self.device)
        app_j = ConvSparseDecom(data_j, filt, lamda=self.lamda,
                                multi_channel=self.multi_channel, mode=self.mode,
                                max_iter=self.max_iter, max_power_iter=self.max_power_iter)

        fea = app_j.run()

        with self.device:
            return fea[slc]

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def save(self, filename):
        self.use_device(sp.util.cpu_device)
        with open(filename, "wb") as f:
            pickle.dump(self, f)


def _get_csc_update_filt(A_filt, filt, data_j, num_batches, alpha, proxg):

    device = sp.util.get_device(filt)
    def update_filt():
        with device:
            gradf_filt = A_filt.H(A_filt(filt) - data_j)
            gradf_filt *= num_batches

            sp.util.axpy(filt, -alpha, gradf_filt)

        sp.util.move_to(filt, proxg(alpha, filt))

    return update_filt


def _get_csc_coef_shape(data_shape, num_data, num_filters, filt_width, mode, multi_channel):
    if multi_channel:
        ndim = len(data_shape) - 2
    else:
        ndim = len(data_shape) - 1

    if mode == 'full':
        coef_shape = tuple([num_data, num_filters] +
                          [i - min(i, filt_width) + 1 for i in data_shape[-ndim:]])
    else:
        coef_shape = tuple([num_data, num_filters] +
                          [i + min(i, filt_width) - 1 for i in data_shape[-ndim:]])

    return coef_shape


def _get_csc_A_coef(coef, filt, mode, multi_channel):
    
    if sp.util.get_device(filt) != sp.util.cpu_device and sp.config.cudnn_enabled:
        if multi_channel:
            filt_cudnn_shape = filt.shape
        else:
            filt_cudnn_shape = (1, ) + filt.shape
            
        A_coef = sp.linop.CudnnConvolveData(coef.shape, filt.reshape(filt_cudnn_shape), mode=mode)
        
        if not multi_channel:
            data_shape = [A_coef.oshape[0]] + A_coef.oshape[2:]
            R_data = sp.linop.Reshape(data_shape, A_coef.oshape)
        
            A_coef = R_data * A_coef
    else:
        if multi_channel:
            ndim = filt.ndim - 2
        else:
            ndim = filt.ndim - 1
            
        C_coef = sp.linop.Convolve(coef.shape, filt, axes=range(-ndim, 0), mode=mode)
        S_coef = sp.linop.Sum(C_coef.oshape, axes=[-(ndim + 1)])

        A_coef = S_coef * C_coef

    return A_coef


def _get_csc_filt_shape(data_shape, num_filters, filt_width, mode, multi_channel):
    if multi_channel:
        ndim = len(data_shape) - 2
    else:
        ndim = len(data_shape) - 1

    if multi_channel:
        num_channels = data_shape[1]
        filt_shape = tuple([num_channels, num_filters] +
                          [min(d, filt_width) for d in data_shape[-ndim:]])
    else:
        filt_shape = tuple([num_filters] + [min(d, filt_width) for d in data_shape[-ndim:]])

    return filt_shape


def _get_csc_A_filt(coef, filt, mode, multi_channel):

    if sp.util.get_device(filt) != sp.util.cpu_device and sp.config.cudnn_enabled:
        if multi_channel:
            filt_cudnn_shape = filt.shape
        else:
            filt_cudnn_shape = (1, ) + filt.shape
            
        R_filt = sp.linop.Reshape(filt_cudnn_shape, filt.shape)
        C_filt = sp.linop.CudnnConvolveFilter(filt_cudnn_shape, coef, mode=mode)
        
        A_filt = C_filt * R_filt

        if not multi_channel:
            data_shape = [A_filt.oshape[0]] + A_filt.oshape[2:]
            R_data = sp.linop.Reshape(data_shape, A_filt.oshape)

            A_filt = R_data * A_filt
    else:
        ndim = filt.ndim - 1
        C_filt = sp.linop.Convolve(filt.shape, coef, axes=range(-ndim, 0), mode=mode)
        S_filt = sp.linop.Sum(C_filt.oshape, axes=[1])

        A_filt = S_filt * C_filt

    return A_filt
