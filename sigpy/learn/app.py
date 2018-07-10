'''
Machine Learning Apps.
'''
import logging
import sigpy as sp
        

class ConvSparseDecom(sp.app.LinearLeastSquares):
    '''Convolutional sparse decomposition app.

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
    '''

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

        fea = sp.util.zeros(fea_shape, dtype=dat.dtype, device=sp.util.get_device(dat))
            
        A_fea = _get_csc_A_fea(fea, dic, mode, multi_channel)
        proxg = sp.prox.L1Reg(A_fea.ishape, lamda)
        
        super().__init__(A_fea, dat, fea, proxg=proxg, **kwargs)


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
