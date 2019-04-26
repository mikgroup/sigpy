# -*- coding: utf-8 -*-
"""This module contains an abstract class App for iterative signal processing,
and provides a few general Apps, including a linear least squares App,
and a maximum eigenvalue estimation App.
"""
import numpy as np

from tqdm import tqdm
from sigpy import backend, linop, prox, util
from sigpy.alg import (PowerMethod, GradientMethod,
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

    Attributes:
        alg (Alg)
        show_pbar (bool)

    """

    def __init__(self, alg, show_pbar=True):
        self.alg = alg
        self.show_pbar = show_pbar

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

            self.pbar = tqdm(total=self.alg.max_iter, desc=name)

        while(not self.alg.done()):
            self._pre_update()
            self.alg.update()
            self._post_update()
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
                 max_iter=30, show_pbar=True):
        self.x = util.randn(A.ishape, dtype=dtype, device=device)
        alg = PowerMethod(A, self.x, max_iter=max_iter)
        super().__init__(alg, show_pbar=show_pbar)

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
        \frac{\lambda}{2} \| R x \|_2^2 + \frac{\mu}{2} \| x - z \|_2^2

    Three algorithms can be used: :class:`sigpy.alg.ConjugateGradient`,
    :class:`sigpy.alg.GradientMethod`,
    and :class:`sigpy.alg.PrimalDualHybridGradient`.
    If ``alg_name`` is None, :class:`sigpy.alg.ConjugateGradient` is used
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
        R (None or Linop): l2 regularization linear operator.
        mu (float): l2 bias regularization parameter.
        z (float or array): Bias for l2 regularization.
        alg_name (str): {`'ConjugateGradient'`, `'GradientMethod'`,
            `'PrimalDualHybridGradient'`}.
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
        save_objective_values (bool): Toggle saving objective value.

    """

    def __init__(self, A, y, x=None, proxg=None,
                 lamda=0, G=None, g=None, R=None, mu=0, z=0,
                 alg_name=None, max_iter=100,
                 P=None, alpha=None, max_power_iter=30, accelerate=True,
                 tau=None, sigma=None,
                 save_objective_values=False, show_pbar=True):
        self.A = A
        self.y = y
        self.x = x
        self.proxg = proxg
        self.lamda = lamda
        self.G = G
        self.g = g
        self.R = R
        self.mu = mu
        self.z = z
        self.alg_name = alg_name
        self.max_iter = max_iter
        self.P = P
        self.alpha = alpha
        self.max_power_iter = max_power_iter
        self.accelerate = accelerate
        self.tau = tau
        self.sigma = sigma
        self.save_objective_values = save_objective_values
        self.show_pbar = show_pbar

        self.y_device = backend.get_device(y)
        if self.x is None:
            with self.y_device:
                self.x = self.y_device.xp.zeros(A.ishape, dtype=y.dtype)

        self.x_device = backend.get_device(self.x)
        self._get_alg()
        if self.save_objective_values:
            self.objective_values = []

    def _summarize(self):
        if self.save_objective_values:
            self.objective_values.append(self.objective())

        if self.show_pbar:
            if self.save_objective_values:
                self.pbar.set_postfix(
                    obj='{0:.2E}'.format(self.objective_values[-1]))
            else:
                self.pbar.set_postfix(resid='{0:.2E}'.format(self.alg.resid))

    def _output(self):
        return self.x

    def _get_alg(self):
        if self.alg_name is None:
            if self.proxg is None:
                self.alg_name = 'ConjugateGradient'
            elif self.G is None:
                self.alg_name = 'GradientMethod'
            else:
                self.alg_name = 'PrimalDualHybridGradient'

        if self.alg_name == 'ConjugateGradient':
            if self.proxg is not None:
                raise ValueError(
                    'ConjugateGradient cannot have proxg specified.')

            self._get_ConjugateGradient()
        elif self.alg_name == 'GradientMethod':
            if self.G is not None:
                raise ValueError('GradientMethod cannot have G specified.')

            self._get_GradientMethod()
        elif self.alg_name == 'PrimalDualHybridGradient':
            self._get_PrimalDualHybridGradient()
        else:
            raise ValueError('Invalid alg_name: {alg_name}.'.format(
                alg_name=self.alg_name))

    def _get_ConjugateGradient(self):
        I = linop.Identity(self.x.shape)
        AHA = self.A.H * self.A
        AHy = self.A.H(self.y)

        if self.lamda != 0:
            if self.R is None:
                AHA += self.lamda * I
            else:
                AHA += self.lamda * self.R.H * self.R

        if self.mu != 0:
            AHA += self.mu * I
            util.axpy(AHy, self.mu, self.z)

        self.alg = ConjugateGradient(
            AHA, AHy, self.x, P=self.P, max_iter=self.max_iter)

    def _get_GradientMethod(self):
        def gradf(x):
            with self.y_device:
                r = self.A(x)
                r -= self.y

            with self.x_device:
                gradf_x = self.A.H(r)
                if self.lamda != 0:
                    if self.R is None:
                        util.axpy(gradf_x, self.lamda, x)
                    else:
                        util.axpy(gradf_x, self.lamda, self.R.H(self.R(x)))

                if self.mu != 0:
                    util.axpy(gradf_x, self.mu, x - self.z)

                return gradf_x

        I = linop.Identity(self.x.shape)
        AHA = self.A.H * self.A

        if self.lamda != 0:
            if self.R is None:
                AHA += self.lamda * I
            else:
                AHA += self.lamda * self.R.H * self.R

        if self.mu != 0:
            AHA += self.mu * I

        max_eig = MaxEig(AHA, dtype=self.x.dtype,
                         device=self.x_device, max_iter=self.max_power_iter,
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
            accelerate=self.accelerate)

    def _get_PrimalDualHybridGradient(self):
        with self.y_device:
            y = -self.y
            A = self.A

        if self.proxg is None:
            proxg = prox.NoOp(self.x.shape)
        else:
            proxg = self.proxg

        if self.lamda > 0 or self.mu > 0:
            def gradh(x):
                with backend.get_device(self.x):
                    gradh_x = 0
                    if self.lamda > 0:
                        if self.R is None:
                            gradh_x += self.lamda * x
                        else:
                            gradh_x += self.lamda * self.R.H(self.R(x))

                    if self.mu > 0:
                        gradh_x += self.mu * (x - self.z)

                    return gradh_x

            if self.R is None:
                gamma_primal = self.lamda + self.mu
            else:
                gamma_primal = self.mu

        else:
            gradh = None
            gamma_primal = 0

        if self.G is None:
            proxfc = prox.L2Reg(y.shape, 1, y=y)
            gamma_dual = 1
        else:
            A = linop.Vstack([A, self.G])
            proxf1c = prox.L2Reg(self.y.shape, 1, y=y)
            proxf2c = prox.Conj(self.proxg)
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

            self.tau = 1 / (max_eig + self.lamda + self.mu)
        else:
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
            gradh=gradh,
            max_iter=self.max_iter)

    def objective(self):
        with self.y_device:
            r = self.A(self.x) - self.y

            obj = 1 / 2 * util.norm2(r)
            if self.lamda > 0:
                if self.R is None:
                    obj += self.lamda / 2 * util.norm2(self.x)
                else:
                    obj += self.lamda / 2 * util.norm2(self.R(self.x))

            if self.mu != 0:
                obj += self.mu / 2 * util.norm2(self.x - self.z)

            if self.proxg is not None:
                if self.g is None:
                    raise ValueError(
                        'Cannot compute objective when proxg is specified,'
                        'but g is not.')

                if self.G is None:
                    obj += self.g(self.x)
                else:
                    obj += self.g(self.G(self.x))

            obj = util.asscalar(obj)
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
