'''
Machine Learning Apps.
'''
import logging
import sigpy as sp
        

class ConvSparseDecom(sp.app.LinearLeastSquares):
    """Convolutional sparse decomposition app.

    Considers the model :math:`y_j = \sum_i d_i \ast f_{ij}`, and the problem

    .. math:: 
        \min_{f_{ij}\frac{1}{2}\|y_j - \sum_i c_i \ast f_{ij}\|_2^2 + \lambda \|f_{ij}\|_1
    where :math:`y_j` is the jth data, :math:`f_i` is the ith filter, 
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
        array: Coefficient array f. First two dimensions are number of data and number of filters.

    See Also:
        :func:`sigpy.app.LinearLeastSquares`

    """

    def __init__(self, data, filt, lamda=0.001, mode='full', multi_channel=False, **kwargs):

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
        coef_shape = _get_csc_coef_shape(
            data.shape, num_data, num_filters, filt_width, mode, multi_channel)

        self.coef = sp.util.zeros(coef_shape, dtype=data.dtype, device=sp.util.get_device(data))
            
        A_coef = _get_csc_A_coef(self.coef, filt, mode, multi_channel)
        proxg = sp.prox.L1Reg(A_coef.ishape, lamda)
        
        super().__init__(A_coef, data, self.coef, proxg=proxg, **kwargs)


class ConvSparseCoding(sp.app.App):
    """Convolutional sparse coding application.

    Args:
        data (array): data array y, the first dimension is the number of data.
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
        array: Coefficient array. First two dimensions are number of data and number of filters.

    """

    def __init__(self, data, num_filters, filt_width, batch_size,
                 lamda=0.001, alpha=1,
                 max_inner_iter=100, max_power_iter=10, max_iter=100,
                 mode='full', multi_channel=False, device=sp.util.cpu_device):
    
        dtype = data.dtype
        self.data = data
        
        self.batch_size = batch_size
        num_batches = len(data) // batch_size
        self.j_idx = sp.index.ShuffledIndex(num_batches)
        self.data_j = sp.util.empty((batch_size, ) + data.shape[1:], dtype=dtype, device=device)
        
        filt_shape = _get_csc_filt_shape(data.shape, num_filters, filt_width, mode, multi_channel)
        self.filt = sp.util.empty(filt_shape, dtype=dtype, device=device)
        self.proxg = sp.prox.L2Proj(self.filt.shape, 1, axes=tuple(range(1, self.filt.ndim)))

        min_coef_j_app = ConvSparseDecom(self.data_j, self.filt, lamda=lamda,
                                        mode=mode, multi_channel=multi_channel,
                                        max_power_iter=max_power_iter, max_iter=max_inner_iter)
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

    def _output(self):
        return self.filt

    
class LinearRegression(sp.app.LinearLeastSquares):
    """Performs linear regression to fit coefficients to data.

    Considers the model data = coef * mat, and solves

    .. math::
        \min_mat || coef * mat - data ||_2^2

    Args:
        coef (array): coefficient of shape (num_data, ...).
        data (array): data of shape (num_data, ...).
        batch_size (int): batch size.
        alpha (float): step size.

    Returns:
       array: matrix of shape coef.shape[1:] + data.shape[1:].

    """

    def __init__(self, coef, data, batch_size, alpha,
                 max_iter=100, device=sp.util.cpu_device, **kwargs):
        
        dtype = data.dtype

        num_data = len(data)
        num_batches = num_data // batch_size
        self.batch_size = batch_size
        self.coef = coef
        self.data = data

        mat = sp.util.zeros(coef.shape[1:] + data.shape[1:], dtype=dtype, device=device)
        
        self.j_idx = sp.index.ShuffledIndex(num_batches)
        self.coef_j = sp.util.empty(
            (batch_size, ) + coef.shape[1:], dtype=dtype, device=device)
        self.data_j = sp.util.empty(
            (batch_size, ) + data.shape[1:], dtype=dtype, device=device)
        
        A = _get_lr_A(self.coef_j, self.data_j, mat, batch_size)
        
        super().__init__(A, self.data_j, mat, alg_name='GradientMethod', accelerate=False,
                         alpha=alpha, max_iter=max_iter)
        
    def _pre_update(self):
        j = self.j_idx.next()
        j_start = j * self.batch_size
        j_end = (j + 1) * self.batch_size
        
        sp.util.move_to(self.coef_j, self.coef[j_start:j_end])
        sp.util.move_to(self.data_j, self.data[j_start:j_end])

        
def _get_lr_A(coef_j, data_j, mat, batch_size):
    coef_j_size = sp.util.prod(coef_j.shape[1:])
    data_j_size = sp.util.prod(data_j.shape[1:])
    
    Ri = sp.linop.Reshape([coef_j_size, data_j_size], mat.shape)
    M = sp.linop.MatMul([coef_j_size, data_j_size], coef_j.reshape([batch_size, -1]))
    Ro = sp.linop.Reshape(data_j.shape, [batch_size, data_j_size])

    A = Ro * M * Ri

    return A


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
