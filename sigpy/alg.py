# -*- coding: utf-8 -*-
"""This module provides an abstract class Alg for iterative algorithms,
and implements commonly used methods, such as gradient methods,
Newton's method, and the augmented Lagrangian method.
"""
import numpy as np
from sigpy import backend, util


class Alg(object):
    """Abstraction for iterative algorithms.

    The standard way of using an :class:`Alg` object, say alg, is as follows:

    >>> while not alg.done():
    >>>     alg.update()

    The user is free to run other things in the while loop.
    An :class:`Alg` object is meant to run once.
    Once done, the object should not be run again.

    When creating a new :class:`Alg` class, the user should supply
    an _update() function
    to perform the iterative update, and optionally a _done() function
    to determine when to terminate the iteration. The default _done() function
    simply checks whether the number of iterations has reached the maximum.

    The interface for each :class:`Alg` class should not depend on
    Linop or Prox explicitly.
    For example, if the user wants to design an
    :class:`Alg` class to accept a Linop, say A,
    as an argument, then it should also accept any function that can be called
    to compute x -> A(x). Similarly, to accept a Prox, say proxg,
    as an argument,
    the Alg class should accept any function that can be called to compute
    alpha, x -> proxg(x).

    Args:
        max_iter (int): Maximum number of iterations.

    Attributes:
        max_iter (int): Maximum number of iterations.
        iter (int): Current iteration.

    """

    def __init__(self, max_iter):
        self.max_iter = max_iter
        self.iter = 0

    def _update(self):
        raise NotImplementedError

    def _done(self):
        return self.iter >= self.max_iter

    def update(self):
        """Perform one update step.

        Call the user-defined _update() function and increment iter.
        """
        self._update()
        self.iter += 1

    def done(self):
        """Return whether the algorithm is done.

        Call the user-defined _done() function.
        """
        return self._done()


class PowerMethod(Alg):
    """Power method to estimate maximum eigenvalue and eigenvector.

    Args:
        A (Linop or function): Function to a hermitian linear mapping.
        x (array): Variable to optimize over.
        max_iter (int): Maximum number of iterations.

    Attributes:
        max_eig (float): Maximum eigenvalue of `A`.

    """

    def __init__(self, A, x, norm_func=None, max_iter=30):
        self.A = A
        self.x = x
        self.max_eig = np.infty
        self.norm_func = norm_func
        super().__init__(max_iter)

    def _update(self):
        y = self.A(self.x)
        device = backend.get_device(y)
        xp = device.xp
        with device:
            if self.norm_func is None:
                self.max_eig = xp.linalg.norm(y).item()
            else:
                self.max_eig = self.norm_func(y)

            backend.copyto(self.x, y / self.max_eig)

    def _done(self):
        return self.iter >= self.max_iter


class GradientMethod(Alg):
    r"""First order gradient method.

    For the simplest setting when proxg is not specified,
    the method considers the objective function:

    .. math:: \min_x f(x)

    where :math:`f` is (sub)-differentiable and performs the update:

    .. math:: x_\text{new} = x - \alpha \nabla f(x)

    When proxg is specified, the method considers the composite
    objective function:

    .. math:: f(x) + g(x)

    where :math:`f` is (sub)-differentiable and :math:`g` is simple,
    and performs the update:

    .. math:: x_\text{new} = \text{prox}_{\alpha g}(x - \alpha \nabla f(x))

    Nesterov's acceleration is supported by toggling the `accelerate`
    input option.

    Args:
        gradf (function): function to compute :math:`\nabla f`.
        x (array): variable to optimize over.
        alpha (float or None): step size, or initial step size
             if backtracking line-search is on.
        proxg (Prox, function or None): Prox or function to compute
            proximal operator of :math:`g`.
        accelerate (bool): toggle Nesterov acceleration.
        max_iter (int): maximum number of iterations.
        tol (float): Tolerance for stopping condition.

    References:
        Nesterov, Y. E. (1983).
        A method for solving the convex programming problem
        with convergence rate O (1/k^ 2).
        In Dokl. Akad. Nauk SSSR (Vol. 269, pp. 543-547).

        Beck, A., & Teboulle, M. (2009).
        A fast iterative shrinkage-thresholding algorithm
        for linear inverse problems.
        SIAM journal on imaging sciences, 2(1), 183-202.

    """

    def __init__(self, gradf, x, alpha, proxg=None,
                 accelerate=False, max_iter=100,
                 tol=0):
        self.gradf = gradf
        self.alpha = alpha
        self.accelerate = accelerate
        self.proxg = proxg
        self.x = x
        self.tol = tol

        self.device = backend.get_device(x)
        with self.device:
            if self.accelerate:
                self.z = self.x.copy()
                self.t = 1

        self.resid = np.infty
        super().__init__(max_iter)

    def _update(self):
        xp = self.device.xp
        with self.device:
            x_old = self.x.copy()

            if self.accelerate:
                backend.copyto(self.x, self.z)

            # Perform update
            util.axpy(self.x, -self.alpha, self.gradf(self.x))
            if self.proxg is not None:
                backend.copyto(self.x, self.proxg(self.alpha, self.x))

            if self.accelerate:
                t_old = self.t
                self.t = (1 + (1 + 4 * t_old**2)**0.5) / 2
                backend.copyto(self.z, self.x +
                               ((t_old - 1) / self.t) * (self.x - x_old))

            self.resid = xp.linalg.norm((self.x - x_old) / self.alpha).item()

    def _done(self):
        return (self.iter >= self.max_iter) or self.resid <= self.tol


class ConjugateGradient(Alg):
    r"""Conjugate gradient method.

    Solves for:

    .. math:: A x = b

    where A is a Hermitian linear operator.

    Args:
        A (Linop or function): Linop or function to compute A.
        b (array): Observation.
        x (array): Variable.
        P (function or None): Preconditioner.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for stopping condition.

    """

    def __init__(self, A, b, x, P=None, max_iter=100, tol=0):
        self.A = A
        self.b = b
        self.P = P
        self.x = x
        self.tol = tol
        self.device = backend.get_device(x)
        with self.device:
            xp = self.device.xp
            self.r = b - self.A(self.x)

            if self.P is None:
                z = self.r
            else:
                z = self.P(self.r)

            if max_iter > 1:
                self.p = z.copy()
            else:
                self.p = z

            self.not_positive_definite = False
            self.rzold = xp.real(xp.vdot(self.r, z))
            self.resid = self.rzold.item()**0.5

        super().__init__(max_iter)

    def _update(self):
        with self.device:
            xp = self.device.xp
            Ap = self.A(self.p)
            pAp = xp.real(xp.vdot(self.p, Ap)).item()
            if pAp <= 0:
                self.not_positive_definite = True
                return

            self.alpha = self.rzold / pAp
            util.axpy(self.x, self.alpha, self.p)
            if self.iter < self.max_iter - 1:
                util.axpy(self.r, -self.alpha, Ap)
                if self.P is not None:
                    z = self.P(self.r)
                else:
                    z = self.r

                rznew = xp.real(xp.vdot(self.r, z))
                beta = rznew / self.rzold
                util.xpay(self.p, beta, z)
                self.rzold = rznew

            self.resid = self.rzold.item()**0.5

    def _done(self):
        return (self.iter >= self.max_iter or
                self.not_positive_definite or self.resid <= self.tol)


class PrimalDualHybridGradient(Alg):
    r"""Primal dual hybrid gradient.

    Considers the problem:

    .. math:: \min_x \max_u - f^*(u) + g(x) + \left<Ax, u\right>

    Or equivalently:

    .. math:: \min_x f(A x) + g(x)

    where f, and g are simple.

    Args:
        proxfc (function): Function to compute proximal operator of f^*.
        proxg (function): Function to compute proximal operator of g.
        A (function): Function to compute a linear mapping.
        AH (function): Function to compute the adjoint linear mapping of `A`.
        x (array): Primal solution.
        u (array): Dual solution.
        tau (float or array): Primal step-size.
        sigma (float or array): Dual step-size.
        gamma_primal (float): Strong convexity parameter of g.
        gamma_dual (float): Strong convexity parameter of f^*.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for stopping condition.

    References:
       Chambolle, A., & Pock, T. (2011).
       A first-order primal-dual algorithm for convex problems with
       applications to imaging.
       Journal of mathematical imaging and vision, 40(1), 120-145.

    """

    def __init__(self, proxfc, proxg, A, AH, x, u,
                 tau, sigma, theta=1,
                 gamma_primal=0, gamma_dual=0,
                 max_iter=100, tol=0):
        self.proxfc = proxfc
        self.proxg = proxg
        self.tol = tol

        self.A = A
        self.AH = AH

        self.u = u
        self.x = x

        self.tau = tau
        self.sigma = sigma
        self.theta = theta
        self.gamma_primal = gamma_primal
        self.gamma_dual = gamma_dual

        self.x_device = backend.get_device(x)
        self.u_device = backend.get_device(u)

        with self.x_device:
            self.x_ext = self.x.copy()

        if self.gamma_primal > 0:
            xp = self.x_device.xp
            with self.x_device:
                self.tau_min = xp.amin(xp.abs(tau)).item()

        if self.gamma_dual > 0:
            xp = self.u_device.xp
            with self.u_device:
                self.sigma_min = xp.amin(xp.abs(sigma)).item()

        self.resid = np.infty

        super().__init__(max_iter)

    def _update(self):
        # Update dual.
        util.axpy(self.u, self.sigma, self.A(self.x_ext))
        backend.copyto(self.u, self.proxfc(self.sigma, self.u))

        # Update primal.
        with self.x_device:
            x_old = self.x.copy()
            util.axpy(self.x, -self.tau, self.AH(self.u))
            backend.copyto(self.x, self.proxg(self.tau, self.x))

        # Update step-size if neccessary.
        if self.gamma_primal > 0 and self.gamma_dual == 0:
            with self.x_device:
                xp = self.x_device.xp
                theta = 1 / (1 + 2 * self.gamma_primal * self.tau_min)**0.5
                self.tau *= theta
                self.tau_min *= theta

            with self.u_device:
                self.sigma /= theta
        elif self.gamma_primal == 0 and self.gamma_dual > 0:
            with self.u_device:
                xp = self.u_device.xp
                theta = 1 / (1 + 2 * self.gamma_dual * self.sigma_min)**0.5
                self.sigma *= theta
                self.sigma_min *= theta

            with self.x_device:
                self.tau /= theta
        else:
            theta = self.theta

        # Extrapolate primal.
        with self.x_device:
            xp = self.x_device.xp
            x_diff = self.x - x_old
            self.resid = xp.linalg.norm(x_diff / self.tau**0.5).item()
            backend.copyto(self.x_ext, self.x + theta * x_diff)

    def _done(self):
        return (self.iter >= self.max_iter) or (self.resid <= self.tol)


class AltMin(Alg):
    """Alternating minimization.

    Args:
        min1 (function): Function to minimize over variable 1.
        min2 (function): Funciton to minimize over variable 2.
        max_iter (int): Maximum number of iterations.

    """

    def __init__(self, min1, min2, max_iter=30):
        self.min1 = min1
        self.min2 = min2
        super().__init__(max_iter)

    def _update(self):
        self.min1()
        self.min2()


class AugmentedLagrangianMethod(Alg):
    r"""Augmented Lagrangian method for constrained optimization.

    Consider the equality and inequality constrained problem:

    .. math:: \min_{x: g(x) \leq 0, h(x) = 0} f(x)

    And perform the following update steps:

    .. math::
        x \in \text{argmin}_{x} L(x, u, v, \mu)\\
        u = [u + \mu g(x)]_+
        v = v + \mu h(x)

    where :math:`L(x, u, v, \mu)`: is the augmented Lagrangian function:

    .. math::
        L(x, u, v, \mu) = f(x) + \frac{\mu}{2}(
        \|[g(x) + \frac{u}{\mu}]_+\|_2^2 + \|h(x) + \frac{v}{\mu}\|_2^2)

    Args:
        minL (function): a function that minimizes the augmented Lagrangian.
        g (None or function): a function that takes :math:`x` as input,
            and outputs :math:`g(x)`, the inequality constraints.
        h (None or function): a function that takes :math:`x` as input,
            and outputs :math:`h(x)`, the equality constraints.
        x (array): primal variable.
        u (array): dual variable for inequality constraints.
        v (array): dual variable for equality constraints.
        mu (scalar): step size.
        max_iter (int): maximum number of iterations.

    """
    def __init__(self, minL, g, h, x, u, v, mu, max_iter=30):
        self.minL = minL
        self.g = g
        self.h = h
        self.x = x
        self.u = u
        self.v = v
        self.mu = mu
        super().__init__(max_iter)

    def _update(self):
        self.minL()
        if self.g is not None:
            device = backend.get_device(self.u)
            xp = device.xp
            with device:
                util.axpy(self.u, self.mu, self.g(self.x))
                backend.copyto(self.u, xp.clip(self.u, 0, np.infty))

        if self.h is not None:
            util.axpy(self.v, self.mu, self.h(self.x))


class ADMM(Alg):
    r"""Alternating Direction Method of Multipliers.

    Consider the equality constrained problem:

    .. math:: \min_{x: A x + B z = c} f(x) + g(z)

    And perform the following update steps:

    .. math::
        x = \text{argmin}_{x} L_\mu(x, z, u)\\
        z = \text{argmin}_{z} L_\mu(x, z, u)\\
        u = u + A x + B z - c

    where :math:`L(x, u, v, \mu)`: is the augmented Lagrangian function:

    .. math::
        L_\rho(x, z, u) = f(x) + g(z) + \frac{\rho}{2}\|A x + B z - c + u\|_2^2

    Args:
        minL_x (function): a function that minimizes L w.r.t. x.
        minL_z (function): a function that minimizes L w.r.t. z.
        x (array): primal variable 1.
        z (array): primal variable 2.
        u (array): scaled dual variable.
        max_iter (int): maximum number of iterations.

    """
    def __init__(self, minL_x, minL_z, x, z, u, A, B, c, max_iter=30):
        self.minL_x = minL_x
        self.minL_z = minL_z
        self.x = x
        self.z = z
        self.u = u
        self.A = A
        self.B = B
        self.c = c
        super().__init__(max_iter)

    def _update(self):
        self.minL_x()
        self.minL_z()
        self.u += self.A(self.x) + self.B(self.z) - self.c


class NewtonsMethod(Alg):
    """Newton's Method.

    Args:
        gradf (function) - A function gradf(x): x -> gradient of f at x.
        inv_hessf (function) - A function H(x): x -> inverse Hessian of f at x,
            which is another function: y -> inverse Hessian of f at x times y.
        x (function) - solution.
        beta (scalar): backtracking linesearch factor.
             Enables backtracking when beta < 1.
        f (function or None): function to compute :math:`f`
             for backtracking line-search.
        max_iter (int): maximum number of iterations.
        tol (float): Tolerance for stopping condition.

    """
    def __init__(self, gradf, inv_hessf, x,
                 beta=1, f=None, max_iter=10, tol=0):
        if beta < 1 and f is None:
            raise TypeError(
                "Cannot do backtracking linesearch without specifying f.")

        self.gradf = gradf
        self.inv_hessf = inv_hessf
        self.x = x
        self.lamda = np.infty
        self.beta = beta
        self.f = f
        self.residual = np.infty
        self.tol = tol

        super().__init__(max_iter)

    def _update(self):
        device = backend.get_device(self.x)
        xp = device.xp
        with device:
            gradf_x = self.gradf(self.x)
            p = -self.inv_hessf(self.x)(gradf_x)
            self.lamda2 = -xp.real(xp.vdot(p, gradf_x)).item()
            if self.lamda2 < 0:
                raise ValueError(
                    'Direction is not descending. Got lamda2={}. '
                    'inv_hessf might not be defined correctly.'.
                    format(self.lamda2))

            x_new = self.x + p
            if self.beta < 1:
                fx = self.f(self.x)
                alpha = 1
                while self.f(x_new) > fx - alpha / 2 * self.lamda2:
                    alpha *= self.beta
                    x_new = self.x + alpha * p

            backend.copyto(self.x, x_new)
            self.residual = self.lamda2**0.5

    def _done(self):
        return self.iter >= self.max_iter or self.residual <= self.tol
