'''
Machine Learning Apps.
'''
import logging
import sigpy as sp
        

class ConvSparseDecom(sp.app.LinearLeastSquares):
    """Convolutional sparse decomposition app.

    Considers the model :math:`y_j = \sum_i d_i \ast f_{ij}`, and the problem

    .. math:: 
        \min_{f_{ij}\frac{1}{2}\|y_j - \sum_i d_i \ast f_{ij}\|_2^2 + \lambda \|f_{ij}\|_1
    where :math:`y_j` is the jth data, :math:`d_i` is the ith dictionary, 
    :math:`f_{ij}` is the ith feature for jth data.

    Args:
        dat (array): data array y, the first dimension is the number of data.
            If multi_channel is True, then the second dimension should be the number of channels.
        dic (array): dictionary d. If multi_channel is True,
            the first dimension is the number of atoms. Otherwise, the first dimension
            is the number of channels, and second dimension is the number of atoms.
        lamda (float): regularization parameter.
        mode (str): convolution mode in forward model. {'full', 'valid'}.
        multi_channel (bool): whether data is multi-channel or not.
        **kwargs: other LinearLeastSquares arguments.

    Returns:
        array: Feature array f. First two dimensions are number of data and number of atoms.

    See Also:
        :func:`sigpy.app.LinearLeastSquares`
    """

    def __init__(self, dat, dic, lamda=0.001, mode='full', multi_channel=False, **kwargs):

        if multi_channel:
            num_atoms = dic.shape[1]
        else:
            num_atoms = dic.shape[0]
            
        dic_width = dic.shape[-1]

        if multi_channel:
            ndim = len(dat.shape) - 2
        else:
            ndim = len(dat.shape) - 1
        
        num_dat = len(dat)
        fea_shape = _get_csc_fea_shape(
            dat.shape, num_dat, num_atoms, dic_width, mode, multi_channel)

        self.fea = sp.util.zeros(fea_shape, dtype=dat.dtype, device=sp.util.get_device(dat))
            
        A_fea = _get_csc_A_fea(self.fea, dic, mode, multi_channel)
        proxg = sp.prox.L1Reg(A_fea.ishape, lamda)
        
        super().__init__(A_fea, dat, self.fea, proxg=proxg, **kwargs)


class ConvSparseCoding(sp.app.App):
    """Convolutional sparse coding application.

    Args:
        dat (array): data array y, the first dimension is the number of data.
            If multi_channel is True, then the second dimension should be the number of channels.
        num_atoms (int): number of atoms in dictionary.
        dic_width (int): dictionary widith.
        batch_size (int): batch size.
        lamda (float): regularization parameter.
        alpha (float): step-size.
        max_inner_iter (int): maximum number of iteration for inner-loop.
        max_power_iter (int): maximum number of iteration for power method.
        mode (str): convolution mode in forward model. {'full', 'valid'}.
        multi_channel (bool): whether data is multi-channel or not.
        **kwargs: other LinearLeastSquares arguments.

    Returns:
        array: Feature array f. First two dimensions are number of data and number of atoms.

    """

    def __init__(self, dat, num_atoms, dic_width, batch_size,
                 lamda=0.001, alpha=1,
                 max_inner_iter=100, max_power_iter=10, max_iter=100,
                 mode='full', multi_channel=False, device=sp.util.cpu_device):
    
        dtype = dat.dtype
        self.dat = dat
        
        self.batch_size = batch_size
        num_batches = len(dat) // batch_size
        self.j_idx = sp.index.ShuffledIndex(num_batches)
        self.dat_j = sp.util.empty((batch_size, ) + dat.shape[1:], dtype=dtype, device=device)
        
        dic_shape = _get_csc_dic_shape(dat.shape, num_atoms, dic_width, mode, multi_channel)
        self.dic = sp.util.empty(dic_shape, dtype=dtype, device=device)
        self.proxg = sp.prox.L2Proj(self.dic.shape, 1, axes=tuple(range(1, self.dic.ndim)))

        min_fea_j_app = ConvSparseDecom(self.dat_j, self.dic, lamda=lamda,
                                        mode=mode, multi_channel=multi_channel,
                                        max_power_iter=max_power_iter, max_iter=max_inner_iter)
        self.fea_j = min_fea_j_app.x
        
        A_dic = _get_csc_A_dic(self.fea_j, self.dic, mode, multi_channel)
        update_dic = _get_csc_update_dic(A_dic, self.dic, self.dat_j,
                                         num_batches, alpha, self.proxg)

        alg = sp.alg.AltMin(min_fea_j_app.run, update_dic, max_iter=max_iter)

        super().__init__(alg)

    def _init(self):
        sp.util.move_to(self.dic, self.proxg(1, sp.util.randn_like(self.dic)))

    def _pre_update(self):
        j = self.j_idx.next()
        j_start = j * self.batch_size
        j_end = (j + 1) * self.batch_size
        
        sp.util.move_to(self.dat_j, self.dat[j_start:j_end])
        
        with sp.util.get_device(self.fea_j):
            self.fea_j.fill(0)

    def _output(self):
        return self.dic


def _get_csc_update_dic(A_dic, dic, dat_j, num_batches, alpha, proxg):

    device = sp.util.get_device(dic)
    def update_dic():
        with device:
            gradf_dic = A_dic.H(A_dic(dic) - dat_j)
            gradf_dic *= num_batches

            sp.util.axpy(dic, -alpha, gradf_dic)
            
        sp.util.move_to(dic, proxg(alpha, dic))

    return update_dic


def _get_csc_fea_shape(dat_shape, num_dat, num_atoms, dic_width, mode, multi_channel):
    if multi_channel:
        ndim = len(dat_shape) - 2
    else:
        ndim = len(dat_shape) - 1

    if mode == 'full':
        fea_shape = tuple([num_dat, num_atoms] +
                          [i - min(i, dic_width) + 1 for i in dat_shape[-ndim:]])
    else:
        fea_shape = tuple([num_dat, num_atoms] +
                          [i + min(i, dic_width) - 1 for i in dat_shape[-ndim:]])

    return fea_shape


def _get_csc_A_fea(fea, dic, mode, multi_channel):
    
    if sp.util.get_device(dic) != sp.util.cpu_device and sp.config.cudnn_enabled:
        if multi_channel:
            dic_cudnn_shape = dic.shape
        else:
            dic_cudnn_shape = (1, ) + dic.shape
            
        A_fea = sp.linop.CudnnConvolveData(fea.shape, dic.reshape(dic_cudnn_shape), mode=mode)
        
        if not multi_channel:
            dat_shape = [A_fea.oshape[0]] + A_fea.oshape[2:]
            R_dat = sp.linop.Reshape(dat_shape, A_fea.oshape)
        
            A_fea = R_dat * A_fea
    else:
        if multi_channel:
            ndim = dic.ndim - 2
        else:
            ndim = dic.ndim - 1
            
        C_fea = sp.linop.Convolve(fea.shape, dic, axes=range(-ndim, 0), mode=mode)
        S_fea = sp.linop.Sum(C_fea.oshape, axes=[-(ndim + 1)])

        A_fea = S_fea * C_fea

    return A_fea


def _get_csc_dic_shape(dat_shape, num_atoms, dic_width, mode, multi_channel):
    if multi_channel:
        ndim = len(dat_shape) - 2
    else:
        ndim = len(dat_shape) - 1

    if multi_channel:
        num_channels = dat_shape[1]
        dic_shape = tuple([num_channels, num_atoms] +
                          [min(d, dic_width) for d in dat_shape[-ndim:]])
    else:
        dic_shape = tuple([num_atoms] + [min(d, dic_width) for d in dat_shape[-ndim:]])

    return dic_shape


def _get_csc_A_dic(fea, dic, mode, multi_channel):

    if sp.util.get_device(dic) != sp.util.cpu_device and sp.config.cudnn_enabled:
        if multi_channel:
            dic_cudnn_shape = dic.shape
        else:
            dic_cudnn_shape = (1, ) + dic.shape
            
        R_dic = sp.linop.Reshape(dic_cudnn_shape, dic.shape)
        C_dic = sp.linop.CudnnConvolveFilter(dic_cudnn_shape, fea, mode=mode)
        
        A_dic = C_dic * R_dic

        if not multi_channel:
            dat_shape = [A_dic.oshape[0]] + A_dic.oshape[2:]
            R_dat = sp.linop.Reshape(dat_shape, A_dic.oshape)

            A_dic = R_dat * A_dic
    else:
        ndim = dic.ndim - 1
        C_dic = sp.linop.Convolve(dic.shape, fea, axes=range(-ndim, 0), mode=mode)
        S_dic = sp.linop.Sum(C_dic.oshape, axes=[1])

        A_dic = S_dic * C_dic

    return A_dic
