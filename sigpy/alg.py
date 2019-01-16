# -*- coding: utf-8 -*-
"""This module provides an abstract class Alg for iterative algorithms,
and implements commonly used methods.
"""
import numpy as np
from sigpy import backend, util, config

if config.cupy_enabled:
    import cupy as cp


class Alg(object):
    """Abstraction for iterative algorithms.

    The standard way of using an :class:`Alg` object, say alg, is as follows:

    >>> while not alg.done():
    >>>     alg.update()

    The user is free to run other things in the while loop.
    An :class:`Alg` object is meant to run once. Once done, the object should not be run again.

    When creating a new :class:`Alg` class, the user should supply an _update() function
    to perform the iterative update, and optionally a _done() function
    to determine when to terminate the iteration. The default _done() function
    simply checks whether the number of iterations has reached the maximum.

    The interface for each :class:`Alg` class should not depend on Linop or Prox explicitly.
    For example, if the user wants to design an :class:`Alg` class to accept a Linop, say A,
    as an argument, then it should also accept any function that can be called
    to compute x -> A(x). Similarly, to accept a Prox, say proxg, as an argument,
    the Alg class should accept any function that can be called to compute
    alpha, x -> proxg(x).

    Args:
        max_iter (int): Maximum number of iterations.

    """
    def __init__(self, max_iter):
        self.max_iter = max_iter 
        self.iter = 0

    def _update(self):
        raise NotImplementedError

    def _done(self):
        return self.iter >= self.max_iter

    def update(self):
        self._update()
        self.iter += 1

    def done(self):
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
    def __init__(self, A, x, max_iter=30):
        self.A = A
        self.x = x
        self.max_eig = np.infty
        super().__init__(max_iter)

    def _update(self):
        y = self.A(self.x)
        device = backend.get_device(y)
        xp = device.xp
        with device:
            self.max_eig = util.asscalar(xp.linalg.norm(y))
            backend.copyto(self.x, y / self.max_eig)

    def _done(self):
        return self.iter >= self.max_iter or self.max_eig == 0


class ProximalPointMethod(Alg):
    """Proximal point method.

    """
    def __init__(self, proxf, alpha, x, max_iter=100, device=backend.cpu_device):
        self.proxf = proxf
        self.alpha = alpha
        self.x = x
        
        super().__init__(max_iter)

    def _update(self):
        backend.copyto(self.x, self.proxf(self.alpha, self.x))


class GradientMethod(Alg):
    r"""First order gradient method.

    For the simplest setting when proxg is not specified, the method considers the objective function:
    
    .. math:: \min_x f(x)
    
    where :math:`f` is (sub)-differentiable and performs the update:

    .. math:: x_\text{new} = x - \alpha \nabla f(x)

    When proxg is specified, the method considers the composite objective function:

    .. math:: f(x) + g(x)

    where :math:`f` is (sub)-differentiable and :math:`g` is simple, and performs the update:
    
    .. math:: x_\text{new} = \text{prox}_{\alpha g}(x - \alpha \nabla f(x))

    Nesterov's acceleration is supported by toggling the `accelerate` input option.
    
    Backtracking line search is supported by setting :math:`\beta < 1`, 
    which keeps scaling the step-size :math:`\alpha` by :math:`\beta` until the following condition holds:

    .. math:: f(x_\text{new}) \leq f(x) + \left< \Delta x, \nabla f(x) \right> + \frac{1}{2 \alpha} \| \Delta x \|_2^2

    Args:
        gradf (function): function to compute :math:`\nabla f`.
        x (array): variable to optimize over.
        alpha (float or None): step size, or initial step size if backtracking line-search is on.
        beta (scalar): backtracking linesearch factor. Enables backtracking when beta < 1.
        f (function or None): function to compute :math:`f` for backtracking line-search.
        proxg (Prox, function or None): Prox or function to compute proximal operator of :math:`g`.
        accelerate (bool): toggle Nesterov acceleration.
        P (Linop, function or None): Linop or function to precondition input, 
            assumes proxg has already incorporated P.
        max_iter (int): maximum number of iterations.

    References:
        Nesterov, Y. E. (1983). 
        A method for solving the convex programming problem with convergence rate 
        O (1/k^ 2). In Dokl. Akad. Nauk SSSR (Vol. 269, pp. 543-547).

        Beck, A., & Teboulle, M. (2009). 
        A fast iterative shrinkage-thresholding algorithm for linear inverse problems. 
        SIAM journal on imaging sciences, 2(1), 183-202.

    """
    def __init__(self, gradf, x, alpha, proxg=None,
                 f=None, beta=1, accelerate=False, max_iter=100):
        if beta < 1 and f is None:
            raise TypeError("Cannot do backtracking linesearch without specifying f.")
            
        self.gradf = gradf
        self.alpha = alpha
        self.f = f
        self.beta = beta
        self.accelerate = accelerate
        self.proxg = proxg
        self.x = x

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
            if self.accelerate:
                backend.copyto(self.x, self.z)

            # Perform update
            gradf_x = self.gradf(self.x)
            alpha = self.alpha
            x_new = self.x - alpha * gradf_x
            if self.proxg is not None:
                x_new = self.proxg(alpha, x_new)
                
            delta_x = x_new - self.x
            # Backtracking line search
            if self.beta < 1:
                fx = self.f(self.x)
                while self.f(x_new) > fx + xp.vdot(delta_x, gradf_x) + 1 / (2 * alpha) * xp.linalg.norm(delta_x)**2:
                    alpha *= self.beta

                    x_new = self.x - alpha * gradf_x
                    if self.proxg is not None:
                        x_new = self.proxg(alpha, x_new)
                        
                    delta_x = x_new - self.x

            backend.copyto(self.x, x_new)
            if self.accelerate:
                t_old = self.t
                self.t = (1 + (1 + 4 * t_old**2)**0.5) / 2
                backend.copyto(self.z, x_new + ((t_old - 1) / self.t) * delta_x)
                
            self.resid = util.asscalar(xp.linalg.norm(delta_x / alpha))

    def _done(self):
        return (self.iter >= self.max_iter) or self.resid == 0


class ConjugateGradient(Alg):
    r"""Conjugate Gradient Method. Solves for:

    .. math:: A x = b

    where A is hermitian.

    Args:
        A (Linop or function): Linop or function to compute A.
        b (array): Observation.
        x (array): Variable.
        P (function or None): Preconditioner.
        max_iter (int): Maximum number of iterations.

    """
    def __init__(self, A, b, x, P=None, max_iter=100):
        self.A = A
        self.P = P
        self.x = x
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

            self.zero_gradient = False
            self.rzold = xp.real(xp.vdot(self.r, z))
            self.resid = util.asscalar(self.rzold)**0.5

        super().__init__(max_iter)

    def _update(self):
        with self.device:
            xp = self.device.xp
            Ap = self.A(self.p)
            pAp = xp.real(xp.vdot(self.p, Ap))
            if pAp == 0:
                self.zero_gradient = True
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

            self.resid = util.asscalar(self.rzold)**0.5

    def _done(self):
        return (self.iter >= self.max_iter) or self.zero_gradient or self.resid == 0


class PrimalDualHybridGradient(Alg):
    r"""Primal dual hybrid gradient.

    Considers the problem:

    .. math:: \min_x \max_u - f^*(u) + g(x) + h(x) + <Ax, u>

    Or equivalently:

    .. math:: \min_x f(A x) + g(x) + h(x)

    where f, and g are simple, and h is Lipschitz continuous.

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

    References:
       Chambolle, A., & Pock, T. (2011).
       A first-order primal-dual algorithm for convex problems with 
       applications to imaging. Journal of mathematical imaging and vision, 40(1), 120-145.

    """
    def __init__(self, proxfc, proxg, A, AH, x, u,
                 tau, sigma, theta=1, gradh=None,
                 gamma_primal=0, gamma_dual=0,
                 max_iter=100):
        self.proxfc = proxfc
        self.proxg = proxg
        self.gradh = gradh

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

        with self.u_device:
            self.u_old = self.u.copy()
            self.x_old = self.x.copy()

        self.resid = np.infty

        super().__init__(max_iter)

    def _update(self):
        backend.copyto(self.u_old, self.u)
        backend.copyto(self.x_old, self.x)

        # Update dual.
        delta_u = self.A(self.x_ext)
        util.axpy(self.u, self.sigma, delta_u)
        backend.copyto(self.u, self.proxfc(self.sigma, self.u))

        # Update primal.
        with self.x_device:
            delta_x = self.AH(self.u)
            if self.gradh is not None:
                delta_x += self.gradh(self.x)

            util.axpy(self.x, -self.tau, delta_x)
            backend.copyto(self.x, self.proxg(self.tau, self.x))

        # Update step-size if neccessary.
        if self.gamma_primal > 0 and self.gamma_dual == 0:
            with self.x_device:
                xp = self.x_device.xp
                theta = 1 / (1 + 2 * self.gamma_primal * xp.amin(xp.abs(self.tau)))**0.5
                self.tau *= theta

            with self.u_device:
                self.sigma /= theta
        elif self.gamma_primal == 0 and self.gamma_dual > 0:
            with self.u_device:
                xp = self.u_device.xp
                theta = 1 / (1 + 2 * self.gamma_dual * xp.amin(xp.abs(self.sigma)))**0.5
                self.sigma *= theta

            with self.x_device:
                self.tau /= theta
        else:
            theta = self.theta

        # Extrapolate primal.
        with self.x_device:
            xp = self.x_device.xp
            x_diff = self.x - self.x_old
            backend.copyto(self.x_ext, self.x + theta * x_diff)
            x_diff_norm = xp.linalg.norm(x_diff / self.tau**0.5)

        with self.u_device:
            xp = self.u_device.xp
            u_diff = self.u - self.u_old
            u_diff_norm = xp.linalg.norm(u_diff / self.sigma**0.5)

        self.resid = util.asscalar(x_diff_norm**2 + u_diff_norm**2)


class AltMin(Alg):
    """Alternating Minimization.

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
