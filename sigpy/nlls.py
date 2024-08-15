"""
This module implements non-linear least square (nlls) algorithms.
Author: Zhengguo Tan <zhengguo.tan@gmail.com>
"""
import numpy as np
import sigpy as sp

from sigpy import backend, alg, prox, util

class NonLinearLeastSquares(alg.Alg):
    """Abstraction for non-linear least square (nlls) algorithms.

    Args:
        A (nlop): non-linear operator.
        y (array): measurements.
        x (None or array): initial guess of unknown. Default None.
        x0 (None or array): previous estimate of unknown. Default None.
        outer_iter (int): outer iteration steps. Default 6.
        alpha (float): regularization strength. Default 1.
        redu (float): reduction rate along outer iteration. Default 2.
        trafos (None or linop): transformation on x. Default None.
        proxf (None or prox): proximal operator. Default None.
        inner_iter (int): inner iteration steps. Default 100.
        inner_tol (float): inner iteration tolerance. Default 0.01.

    References:
        Bauer F., Kannengiesser S. (2007).
        An alternative approach to the image reconstruction for parallel data acquisition in MRI.
        Math. Meth. Appl. Sci. 30, 1437-1451.

        Uecker M., Hohage T., Block K. T., Frahm J. (2008)
        Image reconstruction by regularized nonlinear inversion -- joint estimation of coil sensitivities and image content.
        Magn. Reson. Med. 60, 674-682.

        Tan Z., Roeloffs V., Voit D., Joseph A. A., Untenberger M., Merboldt K. D., Frahm J. (2016).
        Model-based reconstruction for real-time phase-contrast flow MRI: Improved spatiotemporal accuracy.
        Magn. Reson. Med. 77, 1082-1093.
    """
    def __init__(self, A, y, x=None, x0=None,
                 outer_iter=6, alpha=1., redu=2.,
                 trafos=None, proxf=None,
                 inner_iter=100, inner_tol=0.01):
        self.A = A
        self.y = y
        self.device = backend.get_device(y)
        xp = self.device.xp

        self.trafos = trafos
        self.proxf = proxf

        # outer iteration
        self.outer_iter = outer_iter  # max_iter
        self.alpha = alpha
        self.redu = redu

        # inner iteration
        self.inner_iter = inner_iter
        self.inner_tol = inner_tol

        with self.device:
            # initialize x
            self.x = xp.zeros(A.ishape, dtype=y.dtype)
            if x is not None:
                self.x = sp.to_device(x, device=self.device)

            # initialize x0
            self.x0 = xp.zeros(A.ishape, dtype=y.dtype)
            if x0 is not None:
                self.x0 = sp.to_device(x0, device=self.device)

        # FIXME: splitx
        if self.A.repr_str == 'Nlinv':
            self.splitx = 1
        elif self.A.repr_str == 'Diffusion':
            self.splitx = self.A.ishape[0]

        super().__init__(outer_iter)

    def _update(self):
        """Perfom outer iteration steps of the nonlinear problem.
        """
        xp = self.device.xp

        with self.device:
            self.dx = xp.zeros_like(self.x)

            self.r = self.y - self.A(self.x)

            # residual
            resid = xp.linalg.norm(self.r).item()

            print("iter: " + "%2d"%(self.iter) + "; alpha: " + "%.2f"%(self.alpha) + "; resid: " + "%4.3f"%(resid))

            self.p = self.A.adjoint(self.x, self.r)

            self.p += self.alpha * (self.x0 - self.x)

            # update dx
            self.lls()

            self.x += 1. * self.dx

            self.alpha /= self.redu


    def _done(self):
        """Stopping criteria for the outer iteration.
        """
        return self.iter >= self.outer_iter


    def lls(self):
        """Inner linear least square (lls) solver 
        for the linearized non-linear problem.
        """
        xp = self.device.xp

        with self.device:
            def AHA(x):
                return self.A.adjoint(self.x, self.A.derivative(self.x, x))

            # tolerance
            tol = self.inner_tol * xp.linalg.norm(self.p).item()
            
            if self.trafos is None and self.proxf is None:
                """Conjugate gradient method
                """

                AHA_L2 = lambda x: AHA(x) + self.alpha * x
                alg_lls = alg.ConjugateGradient(AHA_L2, self.p, self.dx, max_iter=self.inner_iter, tol=tol)
                while not alg_lls.done():
                    alg_lls.update()

            else:
                """Primal dual method
                """
                einit = util.randn(self.x.shape,
                                   dtype=self.x.dtype,
                                   device=self.device)

                alg_eig = alg.PowerMethod(AHA, einit, max_iter=10)
                while not alg_eig.done():
                    alg_eig.update()

                sigma = 1
                tau = 1 / alg_eig.max_eig
                theta = 1
                self.inner_tol *= self.alpha
                inner_iter = int(min(self.inner_iter, 
                                     10 * 2**(np.log(1/self.alpha))))

                proxf1c = prox.L2Reg(self.y.shape, 1, y=-self.r)

                self.proxf.lamda *= self.alpha
                proxf2c = prox.Conj(self.proxf)

                proxg = prox.L2Reg(self.x.shape, self.alpha, y=self.x0 - self.x)


                # initialization
                dx_ext = self.x.copy()
                u_F1 = xp.zeros_like(self.y)

                u_F2 = xp.zeros_like(self.y, shape=proxf2c.shape)

                for ninner in range(inner_iter):

                    # Update dual 1
                    util.axpy(u_F1, sigma, self.A.derivative(self.x, dx_ext))
                    backend.copyto(u_F1, proxf1c(sigma, u_F1))

                    # Update dual 2
                    util.axpy(u_F2, sigma, self.trafos(dx_ext[:self.splitx, ...]))
                    backend.copyto(u_F2, proxf2c(sigma, u_F2))

                    # Update primal 1
                    dx_old = self.dx.copy()
                    util.axpy(self.dx, -tau, self.A.adjoint(self.x, u_F1))
                    util.axpy(self.dx[:self.splitx, ...], tau, self.trafos.H(u_F2))
                    backend.copyto(self.dx, proxg(tau, self.dx))

                    dx_dif = self.dx - dx_old
                    self.inner_resid =  xp.linalg.norm(dx_dif / tau**0.5).item()
                    backend.copyto(dx_ext, self.dx + theta * dx_dif)

                    print("  iter: " + "%3d"%(ninner) +
                          "; resid: " + "%4.6f"%(self.inner_resid))

                    if self.inner_resid < self.inner_tol:
                        break

            return self.dx
