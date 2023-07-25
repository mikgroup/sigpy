"""Functions for Dynamic Contrast Enhanced (DCE) MRI

Author: Zhengguo Tan <zhengguo.tan@gmail.com>
"""
import numpy as np
import sigpy as sp

from sigpy import backend


def arterial_input_function(sample_time,
                            A = [0.7668, 0.3309],
                            T = [0.1710, 0.4079],
                            sigma = [0.0583, 0.1359],
                            alpha = 1.0259,
                            beta = 0.1608,
                            s = 22.1275,
                            tau = 0.503,
                            Hct = 0.4):
    """
    Args:
        sample_time (array): sampling time array for AIF calculation. [unit: minutes]

        Please refer to the following references for the definition of other parameters.

    References:
        * Parker GJM, Roberts C, Macdonald A, Buonaccorsi GA, Cheung S, Buckley DL, Jackson A, Watson Y, Davies K, Jayson GC.
          Experimentally-derived functional form for a population-averaged high-temporal-resolution arterial input function for dynamic contrast-enhanced MRI.
          Magnetic Resonance in Medicine 56:993-1000 (2006).

        * Tofts PS, Berkowitz B, Schnall MD.
          Quantitative analysis of dynamic Gd-DTPA enhancement in breast tumors using a permeability model.
          Magnetic Resonance in Medicine 33:564-568 (1995).

        * https://mriquestions.com/uploads/3/4/5/7/34572113/dce-mri_siemens.pdf
    """

    sample_mask = sample_time > 0

    Cp = np.zeros_like(sample_time)

    # sigmoid function
    exp_b = np.exp(-beta * sample_time)
    exp_s = np.exp(-s * (sample_time - tau))
    sigmoid_vals = alpha * exp_b / (1 + exp_s)

    # Gaussian functions
    for n in range(len(A)):
        scale = A[n] / (sigma[n] * (2*np.pi)**0.5)
        exp_t = np.exp(-((sample_time - T[n])**2.)/(2 * sigma[n]**2))
        Cp += scale * exp_t

    Cp += sigmoid_vals

    Cp *= sample_mask

    # Cp /= 3  # scaling

    # Cp /= (1 - Hct)

    return Cp


def Patlak(ishape, sample_time, device=sp.cpu_device):
    """
    Args:
        ishape (tuple or list of int): input parameter maps shape [Np, Ny, Nx].
        sample_time (array): sampling time array of the AIF.

    Output:
        linop of C_time: contrast agent signal along time.
        mult (array):

    References:
        * Patlak C, Blasberg RG, Fenstermacher JD.
          Graphical Evaluation of Blood-to-Brain Transfer Constants from Multiple-Time Uptake Data.
          Journal of Cerebral Blood Flow & Metabolism (1983).
    """

    Np = ishape[0]
    Ny, Nx = ishape[-2:]
    # Np, Ny, Nx = ishape  # parameter shape (K, V).^T
    assert(2 == Np)

    sample_time = np.squeeze(sample_time)
    assert(1 == sample_time.ndim)

    t0_idx = np.nonzero(sample_time == 0)
    dt0 = sample_time[t0_idx]

    t1_idx = np.nonzero(sample_time)
    dt1 = np.diff(sample_time[t1_idx], prepend=0)

    dt = np.concatenate((dt0, dt1))

    Cp = arterial_input_function(sample_time)
    K_time = np.cumsum(Cp) * dt[19]
    mult = np.array([K_time, Cp]).T

    mult_dev = sp.to_device(mult, device=device)

    R = sp.linop.Reshape([Np, Ny*Nx], ishape)
    M = sp.linop.MatMul(R.oshape, mult_dev)
    B = sp.linop.Reshape([len(sample_time), 1, 1, 1, Ny, Nx], M.oshape)

    return B * M * R, mult_dev


def _array_to_device(input, device=sp.cpu_device):

    if isinstance(input, backend.get_array_module(input).ndarray):
        output = backend.to_device(input, device=device)
    elif np.isscalar(input):
        output = input

    return output


class DCE(sp.nlop.Nlop):
    """Tracer Kinetic Modeling

    This non-linear operator maps CA (contrast agent concentration) to
    acquired MR image. i.e.

        input:  DCE parameters [Np, 1, 1, Ny, Nx]
                -->
                CA [Ntime, 1, 1, 1, Ny, Nx]
                -->
        output: MR image of the same shape as CA.

    Args:
        ishape (tuple of list of int): input shape.
        sample_time (array): sampling time array of the AIF.
        R1 (float scalar or array): inverse of baseline T1 values [default: 1].
        M0 (float scalar or array): baseline M0 magnetization [default: 5].
        R1CA (float scalar or array): inverse of CA T1 values [default: 4.39].
        FA (float): flip angle in degree [default: 15].
        TR (float): repetition time in second [default 0.006].

    References:
        * Guo Y, Lingala SG, Zhu Y, Lebel RM, Nayak KS.
          Direct estimation of tracer-kinetic parameter maps from highly undersampled brain dynamic contrast enhanced MRI.
          Magnetic Resonance in Medicine 78:1566-1578 (2017).
    """

    def __init__(self, ishape,
                 sample_time,
                 R1 = 1,
                 M0 = 5,
                 R1CA = 4.39,
                 FA = 15,
                 TR = 0.006,   # second
                 rvc = True,
                 verbose = False,
                 device = sp.cpu_device,
                 repr_str = None):

        Np = ishape[-5]
        Nz, Ny, Nx = ishape[-3:]
        assert(1 == ishape[-4])


        sample_time = np.squeeze(sample_time)

        P, mult = Patlak(ishape, sample_time)

        self.Patlak = P

        oshape = P.oshape

        xp = device.xp
        with device:

            self.mult = _array_to_device(mult, device=device)

            self.R1 = _array_to_device(R1, device=device)
            self.M0 = _array_to_device(M0, device=device)
            self.R1CA = _array_to_device(R1CA, device=device)

            FA_radian = FA * xp.pi / 180

            self.M0_trans = self.M0 * xp.sin(FA_radian)

            E1 = xp.exp(-TR * self.R1)
            self.M_steady = self.M0_trans * (1 - E1) / (1 - E1 * xp.cos(FA_radian))

        self.FA = FA
        self.TR = TR

        self.rvc = rvc

        self.device = device

        self.verbose = verbose

        super().__init__(oshape, ishape, repr_str=repr_str)

    def _forward(self, input):
        self.x = _array_to_device(input, device=self.device)
        xp = self.device.xp

        with self.device:
            CA = self.Patlak(self.x)
            x0 = 1.

            FA_radian = self.FA * xp.pi / 180

            E1CA = xp.exp(-self.TR * (self.R1 + self.R1CA * CA))

            CA_trans = self.M0_trans * (1 - E1CA) / (1 - E1CA * xp.cos(FA_radian))

            output = CA_trans + x0 - self.M_steady

        return output

    def _get_Jacobian(self, x):
        self.x = x
        device = backend.get_device(self.x)
        xp = device.xp

        FA_radian = self.FA * xp.pi / 180

        with device:
            CA = self.Patlak(self.x)

            cosFA = xp.cos(FA_radian)

            E1CA = xp.exp(-self.TR * (self.R1 + self.R1CA * CA))

            dCA = self.R1CA * self.M0_trans *\
                    ((self.TR * E1CA * (1 - E1CA * cosFA)) - \
                     (1 - E1CA) * self.TR * E1CA * cosFA) \
                    / (1 - cosFA * E1CA)**2

            Jaco = ((self.mult.T) * (dCA.T)).T

            if self.verbose:
                print('> dCA shape: ', dCA.shape)
                print('> Jaco shape: ', Jaco.shape)

            return Jaco

    def _derivative(self, x, dx):
        device = backend.get_device(dx)
        xp = device.xp

        with device:
            self.Jacobian = self._get_Jacobian(x)
            return xp.sum(self.Jacobian * dx, axis=1, keepdims=True)

    def _adjoint(self, x, dy):
        device = backend.get_device(dy)
        xp = device.xp

        with device:
            self.Jacobian = self._get_Jacobian(x)
            JH = xp.conjugate(self.Jacobian)
            dx = xp.sum(JH * dy, axis=0)

            if self.rvc:
                dx = dx.real + 0. * 1j

            return dx
