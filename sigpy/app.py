# -*- coding: utf-8 -*-
"""This module contains an abstract class App for iterative signal processing,
and provides a few general Apps, including a linear least squares App,
and a maximum eigenvalue estimation App.
"""
import numpy as np
import time

from tqdm.auto import tqdm
from sigpy import backend, linop, prox, util
from sigpy.alg import (PowerMethod, GradientMethod, ADMM,
                       ConjugateGradient, PrimalDualHybridGradient)


class App(object):
    """Abstraction for iterative signal reconstruction applications.

    An App is the final deliverable for each signal reconstruction application.
    The standard way to run an App object, say app, is as follows:

        >>> app.run()

    Each App must have a core Alg object. The run() function runs the Alg,
    with additional convenient features, such as a progress bar, which
    can be toggled with the show_pbar option.

    When creating a new App class, the user should supply an Alg object.
    The user can also optionally define a _pre_update and a _post_update
    function to performs tasks before and after the Alg.update.

    Similar to Alg, an App object is meant to be run once. Different from Alg,
    App is higher level can should use Linop and Prox whenever possible.

    Args:
        alg (Alg): Alg object.
        show_pbar (bool): toggle whether show progress bar.
        leave_pbar (bool): toggle whether to leave progress bar after finished.

    Attributes:
        alg (Alg)
        show_pbar (bool)
        leave_pbar (bool)

    """

    def __init__(self, alg, show_pbar=True, leave_pbar=True,
                 record_time=True):
        self.alg = alg
        self.show_pbar = show_pbar
        self.leave_pbar = leave_pbar
        self.record_time = record_time
        if self.record_time:
            self.time = [0]

    def _pre_update(self):
        return

    def _post_update(self):
        return

    def _summarize(self):
        return

    def _output(self):
        return

    def run(self):
        """Run the App.

        """
        if self.show_pbar:
            if self.__class__.__name__ == 'App':
                name = self.alg.__class__.__name__
            else:
                name = self.__class__.__name__

            self.pbar = tqdm(
                total=self.alg.max_iter, desc=name, leave=self.leave_pbar)

        while not self.alg.done():
            if self.record_time:
                start_time = time.time()

            self._pre_update()
            self.alg.update()
            self._post_update()

            if self.record_time:
                self.time.append(self.time[-1] + time.time() - start_time)

            self._summarize()
            if self.show_pbar:
                self.pbar.update()
                self.pbar.refresh()

        if self.show_pbar:
            self.pbar.close()

        return self._output()


class MaxEig(App):
    """Computes maximum eigenvalue of a Linop.

    Args:
        A (Linop): Hermitian linear operator.
        dtype (Dtype): Data type.
        device (Device): Device.

    Attributes:
        x (int): Eigenvector with largest eigenvalue.

    Output:
        max_eig (int): Largest eigenvalue of A.

    """

    def __init__(self, A, dtype=np.float, device=backend.cpu_device,
                 max_iter=30, show_pbar=True, leave_pbar=True):
        self.x = util.randn(A.ishape, dtype=dtype, device=device)
        alg = PowerMethod(A, self.x, max_iter=max_iter)
        super().__init__(alg, show_pbar=show_pbar, leave_pbar=leave_pbar)

    def _summarize(self):
        if self.show_pbar:
            self.pbar.set_postfix(max_eig='{0:.2E}'.format(self.alg.max_eig))

    def _output(self):
        return self.alg.max_eig


class LinearLeastSquares(App):
    r"""Linear least squares application.

    Solves for the following problem, with optional regularizations:

    .. math::
        \min_x \frac{1}{2} \| A x - y \|_2^2 + g(G x) +
        \frac{\lambda}{2} \| x - z \|_2^2

    Four solvers can be used: :class:`sigpy.alg.ConjugateGradient`,
    :class:`sigpy.alg.GradientMethod`, :class:`sigpy.alg.ADMM`,
    and :class:`sigpy.alg.PrimalDualHybridGradient`.
    If ``solver`` is None, :class:`sigpy.alg.ConjugateGradient` is used
    when ``proxg`` is not specified. If ``proxg`` is specified,
    then :class:`sigpy.alg.GradientMethod` is used when ``G`` is specified,
    and :class:`sigpy.alg.PrimalDualHybridGradient` is used otherwise.

    Args:
        A (Linop): Forward linear operator.
        y (array): Observation.
        x (array): Solution.
        proxg (Prox): Proximal operator of g.
        lamda (float): l2 regularization parameter.
        g (None or function): Regularization function.
            Only used for when `save_objective_values` is true.
        G (None or Linop): Regularization linear operator.
        z (float or array): Bias for l2 regularization.
        solver (str): {`'ConjugateGradient'`, `'GradientMethod'`,
            `'PrimalDualHybridGradient'`, `'ADMM'`}.
        max_iter (int): Maximum number of iterations.
        P (Linop): Preconditioner for ConjugateGradient.
        alpha (None or float): Step size for `GradientMethod`.
        accelerate (bool): Toggle Nesterov acceleration for `GradientMethod`.
        max_power_iter (int): Maximum number of iterations for power method.
            Used for `GradientMethod` when `alpha` is not specified,
            and for `PrimalDualHybridGradient` when `tau` or `sigma` is not
            specified.
        tau (float): Primal step-size for `PrimalDualHybridGradient`.
        sigma (float): Dual step-size for `PrimalDualHybridGradient`.
        rho (float): Augmented Lagrangian parameter for `ADMM`.
        max_cg_iter (int): Maximum number of iterations for conjugate gradient
            in ADMM.
        save_objective_values (bool): Toggle saving objective value.

    """
    def __init__(self, A, y, x=None, proxg=None,
                 lamda=0, G=None, g=None, z=None,
                 solver=None, max_iter=100,
                 P=None, alpha=None, max_power_iter=30, accelerate=True,
                 tau=None, sigma=None,
                 rho=1, max_cg_iter=10, tol=0,
                 save_objective_values=False,
                 show_pbar=True, leave_pbar=True):
        self.A = A
        self.y = y
        self.x = x
        self.proxg = proxg
        self.lamda = lamda
        self.G = G
        self.g = g
        self.z = z
        self.solver = solver
        self.max_iter = max_iter
        self.P = P
        self.alpha = alpha
        self.max_power_iter = max_power_iter
        self.accelerate = accelerate
        self.tau = tau
        self.sigma = sigma
        self.rho = rho
        self.max_cg_iter = max_cg_iter
        self.tol = tol
        self.save_objective_values = save_objective_values
        self.show_pbar = show_pbar
        self.leave_pbar = leave_pbar

        self.y_device = backend.get_device(y)
        if self.x is None:
            with self.y_device:
                self.x = self.y_device.xp.zeros(A.ishape, dtype=y.dtype)

        self.x_device = backend.get_device(self.x)
        self._get_alg()
        if self.save_objective_values:
            self.objective_values = [self.objective()]

        super().__init__(self.alg, show_pbar=show_pbar, leave_pbar=leave_pbar)

    def _summarize(self):
        if self.save_objective_values:
            self.objective_values.append(self.objective())

        if self.show_pbar:
            if self.save_objective_values:
                self.pbar.set_postfix(
                    obj='{0:.2E}'.format(self.objective_values[-1]))
            else:
                self.pbar.set_postfix(resid='{0:.2E}'.format(
                    backend.to_device(self.alg.resid, backend.cpu_device)))

    def _output(self):
        return self.x

    def _get_alg(self):
        if self.solver is None:
            if self.proxg is None:
                self.solver = 'ConjugateGradient'
            elif self.G is None:
                self.solver = 'GradientMethod'
            else:
                self.solver = 'PrimalDualHybridGradient'

        if self.solver == 'ConjugateGradient':
            if self.proxg is not None:
                raise ValueError(
                    'ConjugateGradient cannot have proxg specified.')

            self._get_ConjugateGradient()
        elif self.solver == 'GradientMethod':
            if self.G is not None:
                raise ValueError('GradientMethod cannot have G specified.')

            self._get_GradientMethod()
        elif self.solver == 'PrimalDualHybridGradient':
            self._get_PrimalDualHybridGradient()
        elif self.solver == 'ADMM':
            self._get_ADMM()
        else:
            raise ValueError('Invalid solver: {solver}.'.format(
                solver=self.solver))

    def _get_ConjugateGradient(self):
        I = linop.Identity(self.x.shape)
        AHA = self.A.H * self.A
        AHy = self.A.H(self.y)

        if self.lamda != 0:
            AHA += self.lamda * I
            if self.z is not None:
                util.axpy(AHy, self.lamda, self.z)

        self.alg = ConjugateGradient(
            AHA, AHy, self.x, P=self.P, max_iter=self.max_iter,
            tol=self.tol)

    def _get_GradientMethod(self):
        def gradf(x):
            with self.y_device:
                r = self.A(x)
                r -= self.y

            with self.x_device:
                gradf_x = self.A.H(r)
                if self.lamda != 0:
                    if self.z is None:
                        util.axpy(gradf_x, self.lamda, x)
                    else:
                        util.axpy(gradf_x, self.lamda, x - self.z)

                return gradf_x

        if self.alpha is None:
            I = linop.Identity(self.x.shape)
            AHA = self.A.H * self.A
            if self.lamda != 0:
                AHA += self.lamda * I

            max_eig = MaxEig(AHA, dtype=self.x.dtype, device=self.x_device,
                             max_iter=self.max_power_iter,
                             show_pbar=self.show_pbar).run()
            if max_eig == 0:
                self.alpha = 1
            else:
                self.alpha = 1 / max_eig

        self.alg = GradientMethod(
            gradf,
            self.x,
            self.alpha,
            proxg=self.proxg,
            max_iter=self.max_iter,
            accelerate=self.accelerate, tol=self.tol)

    def _get_PrimalDualHybridGradient(self):
        with self.y_device:
            A = self.A

        if self.lamda > 0:
            gamma_primal = self.lamda
            proxg = prox.L2Reg(self.x.shape, self.lamda,
                               y=self.z, proxh=self.proxg)
        else:
            gamma_primal = 0
            if self.proxg is None:
                proxg = prox.NoOp(self.x.shape)
            else:
                proxg = self.proxg

        with self.y_device:
            if self.G is None:
                proxfc = prox.L2Reg(self.y.shape, 1, y=-self.y)
                gamma_dual = 1
            else:
                A = linop.Vstack([A, self.G])
                proxf1c = prox.L2Reg(self.y.shape, 1, y=-self.y)
                proxf2c = prox.Conj(proxg)
                proxfc = prox.Stack([proxf1c, proxf2c])
                proxg = prox.NoOp(self.x.shape)
                gamma_dual = 0

        if self.tau is None:
            if self.sigma is None:
                self.sigma = 1

            S = linop.Multiply(A.oshape, self.sigma)
            AHA = A.H * S * A
            max_eig = MaxEig(
                AHA,
                dtype=self.x.dtype,
                device=self.x_device,
                max_iter=self.max_power_iter,
                show_pbar=self.show_pbar).run()

            self.tau = 1 / max_eig
        elif self.sigma is None:
            T = linop.Multiply(A.ishape, self.tau)
            AAH = A * T * A.H

            max_eig = MaxEig(
                AAH,
                dtype=self.x.dtype,
                device=self.x_device,
                max_iter=self.max_power_iter,
                show_pbar=self.show_pbar).run()

            self.sigma = 1 / max_eig

        with self.y_device:
            u = self.y_device.xp.zeros(A.oshape, dtype=self.y.dtype)

        self.alg = PrimalDualHybridGradient(
            proxfc,
            proxg,
            A,
            A.H,
            self.x,
            u,
            self.tau,
            self.sigma,
            gamma_primal=gamma_primal,
            gamma_dual=gamma_dual,
            max_iter=self.max_iter,
            tol=self.tol)

    def _get_ADMM(self):
        r"""Considers the formulation:

        .. math::
            \min_{x, v: G x = v} \frac{1}{2} \|A x - y\|_2^2 +
            \frac{\lambda}{2} \| x - z \|_2^2 + g(v)

        """
        xp = self.x_device.xp
        with self.x_device:
            if self.G is None:
                v = self.x.copy()
            else:
                v = self.G(self.x)

            u = xp.zeros_like(v)

        def minL_x():
            AHy = self.A.H * self.y
            if self.G is None:
                AHy += self.rho * (v - u)
            else:
                AHy += self.rho * self.G.H(v - u)

            if self.z is not None:
                AHy += self.lamda * self.z

            AHA = self.A.H * self.A
            I = linop.Identity(self.x.shape)
            if self.G is None:
                AHA += (self.lamda + self.rho) * I
            else:
                if self.lamda > 0:
                    AHA += self.lamda * I

                AHA += self.rho * self.G.H * self.G

            App(ConjugateGradient(AHA, AHy, self.x, P=self.P,
                                  max_iter=self.max_cg_iter),
                show_pbar=False).run()

        def minL_v():
            if self.G is None:
                backend.copyto(v, self.x + u)
            else:
                backend.copyto(v, self.G(self.x) + u)

            if self.proxg is not None:
                backend.copyto(v, self.proxg(1 / self.rho, v))

        I_v = linop.Identity(v.shape)
        if self.G is None:
            I_x = linop.Identity(self.x.shape)
            G = I_x
        else:
            G = self.G

        self.alg = ADMM(minL_x, minL_v, self.x, v, u,
                        G, -I_v, 0, max_iter=self.max_iter)

    def objective(self):
        with self.y_device:
            r = self.A(self.x) - self.y

            obj = 1 / 2 * self.y_device.xp.linalg.norm(r).item()**2
            if self.lamda > 0:
                if self.z is None:
                    obj += self.lamda / 2 * self.x_device.xp.linalg.norm(
                        self.x).item()**2
                else:
                    obj += self.lamda / 2 * self.x_device.xp.linalg.norm(
                        self.x - self.z).item()**2

            if self.proxg is not None:
                if self.g is None:
                    raise ValueError(
                        'Cannot compute objective when proxg is specified,'
                        'but g is not.')

                if self.G is None:
                    obj += self.g(self.x)
                else:
                    obj += self.g(self.G(self.x))

            return obj


class L2ConstrainedMinimization(App):
    r"""L2 contrained minimization application.

    Solves for problem:

    .. math::
        &\min_x g(G x) \\
        &\text{s.t.} \| A x - y \|_2 \leq \epsilon

    Args:
        A (Linop): Forward model linear operator.
        y (array): Observation.
        proxg (Prox): Proximal operator of objective.
        eps (float): Residual.

    """

    def __init__(self, A, y, proxg, eps, x=None, G=None,
                 max_iter=100, tau=None, sigma=None,
                 show_pbar=True):
        self.y = y
        self.x = x
        self.y_device = backend.get_device(y)
        if self.x is None:
            with self.y_device:
                self.x = self.y_device.xp.zeros(A.ishape, dtype=self.y.dtype)

        self.x_device = backend.get_device(self.x)
        if G is None:
            self.max_eig_app = MaxEig(
                A.H * A, dtype=self.x.dtype, device=self.x_device,
                show_pbar=show_pbar)

            proxfc = prox.Conj(prox.L2Proj(A.oshape, eps, y=y))
        else:
            proxf1 = prox.L2Proj(A.oshape, eps, y=y)
            proxf2 = proxg
            proxfc = prox.Conj(prox.Stack([proxf1, proxf2]))
            proxg = prox.NoOp(A.ishape)
            A = linop.Vstack([A, G])

        if tau is None or sigma is None:
            max_eig = MaxEig(A.H * A, dtype=self.x.dtype,
                             device=self.x_device,
                             show_pbar=show_pbar).run()
            tau = 1
            sigma = 1 / max_eig

        with self.y_device:
            self.u = self.y_device.xp.zeros(A.oshape, dtype=self.y.dtype)

        alg = PrimalDualHybridGradient(proxfc, proxg, A, A.H, self.x, self.u,
                                       tau, sigma, max_iter=max_iter)

        super().__init__(alg, show_pbar=show_pbar)

    def _summarize(self):
        if self.show_pbar:
            self.pbar.set_postfix(resid='{0:.2E}'.format(self.alg.resid))

    def _output(self):
        return self.x
