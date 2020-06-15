# -*- coding: utf-8 -*-
"""This module provides an abstract class Alg for iterative algorithms,
and implements commonly used methods, such as gradient methods,
Newton's method, and the augmented Lagrangian method.
"""
import numpy as np
import sigpy as sp
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

            self.resid = xp.linalg.norm(self.x - x_old).item() / self.alpha

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


class SDMM(Alg):
    r"""Simultaneous Direction Method of Multipliers. Can be used for
    unconstrained or constrained optimization with several constraints.

    Solves the problem of form:

    .. math::

        \min_{x} \frac{1}{2}\left\|Ax-d\right\|_2^2
        + \frac{\lambda}{2}\left\|x\right\|_2^2

        s.t. \left\|L_{i}x\right\|_2^2 < c_{i}

    In SDMM, constraints are typically specified as a list of L linear
    operators and c constraints. This algorithm gives the user the option to
    provide either a list of L's and c's or a single maximum (c_max) and/or
    norm constraint (c_norm) with implicit L.

    Solution variable x can be found in alg_method.x.

    Args:
        A (Linop): a system matrix Linop.
        d (array): observation.
        lam (float): scaling parameter.
        L (list of arrays): list of constraint linear operator arrays. If not
            used, provide empty list {}.
        c (list of floats): list of constraints, constraining the L2 norm of
            :math:`L_{i}x`
        mu (float): proximal scaling factor.
        rho (list of floats): list of L scaling parameters, which each one
            corresponding to one L constraint linear array.
        rho_max (float): max constraint scaling parameter (if c_max provided).
        rho_norm (float): norm constraint scaling parameter (if c_norm
            provided).
        eps_pri (float): primal variable error tolerance.
        eps_dual (float): dual variable error tolerance.
        c_max (float): maximum value constraint.
        c_norm (float): norm constraint.
        max_cg_iter (int): maximum number of unconstrained CG iterations per
            SDMM iteration.
        max_iter (int): maximum number of SDMM iterations.

    References:
        Moolekamp, F. and Melchior, P. (2017). 'Block-Simultaneous Direction
        Method of Multipliers: A proximal primal-dual splitting algorithm for
        nonconvex problems with multiple constraints.' arXiv.

    """
    def __init__(self, A, d, lam, L, c, mu, rho, rho_max, rho_norm,
                 eps_pri=10**-5, eps_dual=10**-2, c_max=None, c_norm=None,
                 max_cg_iter=30, max_iter=1000):
        self.A = A
        self.d = d
        self.lam = lam
        self.L = L
        self.c = c
        self.mu = mu
        self.rho = rho
        self.rho_max = rho_max
        self.rho_norm = rho_norm
        self.eps_pri = eps_pri
        self.eps_dual = eps_dual
        self.c_max = c_max
        self.c_norm = c_norm
        self. max_cg_iter = max_cg_iter
        self.stop = False  # stop criterion collector variable
        self.device = backend.get_device(d)
        super().__init__(max_iter)

        M = len(self.L)
        with self.device:
            xp = self.device.xp
            self.x = xp.zeros(self.A.ishape, dtype=np.complex).flatten()
            self.x = xp.expand_dims(self.x, axis=1)
            self.z, self.u = [], []

            for ii in range(M):
                self.z.append(L[ii] @ self.x)
                self.u.append(xp.expand_dims(xp.zeros(xp.shape(L[ii])[0],
                                                      dtype=xp.complex),
                                             axis=1))
            if c_max is not None:
                self.zMax = self.x
                self.uMax = xp.zeros(xp.shape(self.x), dtype=xp.complex)
            if c_norm is not None:
                self.zNorm = self.x
                self.uNorm = xp.zeros(xp.shape(self.x), dtype=xp.complex)

    def prox_rhog(self, v, c):
        with self.device:
            xp = self.device.xp
            if xp.real(xp.linalg.norm(v) ** 2) > c:
                z = v * xp.sqrt(c) / xp.sqrt(xp.real(xp.linalg.norm(v)))
            else:
                z = v
            return z

    def prox_rhog_max(self, v, c):
        with self.device:
            xp = self.device.xp
            z = v
            indices = xp.where((abs(z) ** 2 > c))
            z[indices] = xp.sqrt(c) * z[indices] / xp.absolute(z[indices])
            return z

    def prox_muf(self, v, mu, A, x, d, lam, nCGiters):
        with self.device:
            xp = self.device.xp
            d = xp.vstack((d, xp.sqrt(1 / mu) * v, xp.sqrt(lam) * x))
            Am = self.Amult(x, A, mu, lam)
            int_method = ConjugateGradient(Am.H * Am, Am.H * d, x,
                                           max_iter=nCGiters)

            while not int_method.done():
                int_method.update()

            return int_method.x

    def Amult(self, x, A, mu, lam):
        M = sp.linop.Multiply(x.shape, np.ones(x.shape) * np.sqrt(1 / mu))
        L = sp.linop.Multiply(x.shape, np.ones(x.shape) * np.sqrt(lam))
        Y = sp.linop.Vstack((A, M, L))
        Ry = sp.linop.Reshape((Y.oshape[0], 1), Y.oshape)
        Y = Ry * Y
        return Y

    def _update(self):
        with self.device:
            xp = self.device.xp
            # evaluate objective
            v = self.x
            for ii in range(len(self.L)):
                v -= self.mu / self.rho[ii] * xp.transpose(self.L[ii]) @ \
                                (self.L[ii] @ self.x - self.z[ii] + self.u[ii])
            if self.c_max is not None:
                x_min_z_pl_u = (self.x - self.zMax + self.uMax)
                v -= self.mu / self.rho_max * x_min_z_pl_u
            if self.c_norm is not None:
                x_min_z_pl_u = (self.x - self.zNorm + self.uNorm)
                v -= self.mu / self.rho_norm * x_min_z_pl_u

            self.x = self.prox_muf(v, self.mu, self.A, self.x, self.d,
                                   self.lam, self.max_cg_iter)

            # run through constraints
            z_old = self.z
            for ii in range(len(self.L)):
                self.z[ii] = self.prox_rhog(self.L[ii] @ (self.x + self.u[ii]),
                                            self.c[ii])
                self.u[ii] += self.L[ii] @ self.x - self.z[ii]

            if self.c_max is not None:
                zMax_old = self.zMax
                self.zMax = self.prox_rhog_max(self.x + self.uMax, self.c_max)
                self.uMax = self.uMax + self.x - self.zMax
            if self.c_norm is not None:
                zNorm_old = self.zNorm
                self.zNorm = self.prox_rhog(self.x + self.uNorm, self.c_norm)
                self.uNorm += self.x - self.zNorm

            # check the stopping criteria
            self.stop = True
            rMax, sMax = 0, 0
            for ii in range(len(self.L)):
                # primal residual
                r = self.L[ii] @ self.x - self.z[ii]
                # dual residual
                dz = self.z[ii] - z_old[ii]
                s = 1 / self.rho[ii] * xp.transpose(self.L[ii]) * dz
                if xp.linalg.norm(r) > self.eps_pri or\
                        xp.linalg.norm(s) > self.eps_dual:
                    self.stop = False
                if xp.linalg.norm(r) > rMax:
                    rMax = xp.linalg.norm(r)
                if xp.linalg.norm(s) > sMax:
                    sMax = xp.linalg.norm(s)
            if self.c_norm is not None:
                r = self.x - self.zNorm
                s = 1 / self.rho_norm * (self.zNorm - zNorm_old)
                if xp.linalg.norm(r) > self.eps_pri or\
                        xp.linalg.norm(s) > self.eps_dual:
                    self.stop = False
                if xp.linalg.norm(r) > rMax:
                    rMax = xp.linalg.norm(r)
                if xp.linalg.norm(s) > sMax:
                    sMax = xp.linalg.norm(s)
            if self.c_max is not None:
                r = self.x - self.zMax
                s = 1 / self.rho_max * (self.zMax - zMax_old)
                if xp.linalg.norm(r) > self.eps_pri or\
                        xp.linalg.norm(s) > self.eps_dual:
                    self.stop = False
                if xp.linalg.norm(r) > rMax:
                    rMax = xp.linalg.norm(r)
                if xp.linalg.norm(s) > sMax:
                    sMax = xp.linalg.norm(s)

    def _done(self):
        return self.iter >= self.max_iter or self.stop


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


class GerchbergSaxton(Alg):
    """Gerchberg-Saxton method, also called the variable exchange method.
    Iterative method for recovery of a signal from the amplitude of linear
    measurements |Ax|.

    Args:
        A (Linop): system matrix Linop.
        y (array): observations.
        max_iter (int): maximum number of iterations.
        tol (float): optimization stopping tolerance.
        lamb (float): Tikhonov regularization value.

    """
    def __init__(self, A, y, x0, max_iter=500, tol=0, max_tol=0, lamb=0):

        self.A = A
        self.Aholder = A
        self.y = y
        self.x = x0
        self.max_iter = max_iter
        self.iter = 0
        self.tol = tol
        self.max_tol = max_tol
        self.lamb = lamb
        self.residual = np.infty

    def _update(self):
        device = backend.get_device(self.y)
        xp = device.xp
        with device:

            y_hat = self.y * xp.exp(1j * xp.angle(self.A * self.x))
            I = sp.linop.Identity(self.A.ishape)
            system = self.A.H * self.A + self.lamb * I
            b = self.A.H * y_hat

            alg_internal = ConjugateGradient(system, b, self.x, max_iter=5)

            while not alg_internal.done():
                alg_internal.update()
                self.x = alg_internal.x

        self.residual = xp.sum(xp.absolute(xp.absolute(self.A * self.x)
                                           - self.y))
        self.iter += 1

    def _done(self):
        over_iter = self.iter >= self.max_iter
        under_tol = self.residual <= self.tol
        return over_iter or under_tol
