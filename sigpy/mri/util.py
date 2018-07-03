import numpy as np
from scipy.linalg import solve_triangular


def get_cov(noise):
    '''
    noise - [num_coils, ...]
    '''
    num_coils = noise.shape[0]
    X = noise.reshape([num_coils, -1])
    X -= np.mean(X, axis=-1, keepdims=True)
    cov = np.matmul(X, X.T.conjugate())

    return cov


def whiten(ksp, cov):
    '''
    ksp - [num_coils, ...]
    L - [num_coils, num_coils]
    '''
    num_coils = ksp.shape[0]
    
    x = ksp.reshape([num_coils, -1])
    
    L = np.linalg.cholesky(cov)
    x_w = solve_triangular(L, x, lower=True)
    ksp_w = x_w.reshape(ksp.shape)

    return ksp_w
