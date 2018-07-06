import numpy as np
import logging
from sigpy import util, config

if config.cupy_enabled:
    import cupy as cp

__all__ = ['PowerMethod', 'ConjugateGradient', 'GradientMethod',
           'PrimalDualHybridGradient', 'AltMin', 'Alg']


class Alg(object):
    """Abstraction for iterative algorithm.

    Args:
        max_iter (int): Maximum number of iterations.
        device (int or Device): Device.
    """

    def __init__(self, max_iter, device):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_iter = max_iter
        self.device = util.Device(device)

    def _init(self):
        return

    def _update(self):
        raise NotImplementedError

    def _cleanup(self):
        return

    def _print(self):
        self.logger.debug('Iteration={iter}/{max_iter}'.format(
            iter=self.iter, max_iter=self.max_iter))

    def _done(self):
        return self.iter >= self.max_iter

    def init(self):
        self.iter = 0
        with self.device:
            self._init()

    def update(self):
        with self.device:
            self._update()

            self.iter += 1
            self._print()

    def done(self):
        with self.device:
            return self._done()


class PowerMethod(Alg):
    """Power method to estimate maximum eigenvalue and eigenvector.

    Args:
        A (function): Function to a hermitian linear mapping.
        x (array): Variable to optimize over.
        max_iter (int): Maximum number of iterations.

    Attributes:
        max_eig (float): Maximum eigenvalue of A.
    """

    def __init__(self, A, x, max_iter=30):
        self.A = A
        self.x = x

        super().__init__(max_iter, util.get_device(x))

    def _init(self):
        xp = util.get_xp(self.x)
        self.max_eig = xp.array(np.infty)

    def _update(self):
        y = self.A(self.x)
        self.max_eig = util.norm(y)
        self.x[:] = y / self.max_eig

    def _print(self):
        self.logger.debug('Iteration={iter}/{max_iter}, Maximum Eigenvalue={max_eig}'.format(
            iter=self.iter, max_iter=self.max_iter, max_eig=self.max_eig))


class GradientMethod(Alg):
    """First order gradient method.

    Considers the composite cost function:

    .. math:: f(x) + g(x)

    where f is smooth, and g is simple,
    ie proximal operator of g is simple to compute.

    Args:
        gradf (function): function to compute gradient of f.
        x (array): variable to optimize over.
        alpha (float): step size.
        proxg (function or None): function to compute proximal mapping of g.
        accelerate (bool): toggle Nesterov acceleration.
        P (function): function to precondition, assumes proxg has already incorporated P.
        max_iter (int): maximum number of iterations.
    """

    def __init__(self, gradf, x, alpha, proxg=None,
                 accelerate=False, P=lambda x: x, max_iter=100):
        self.gradf = gradf
        self.alpha = alpha
        self.accelerate = accelerate
        self.proxg = proxg
        self.P = P
        self.x = x

        super().__init__(max_iter, util.get_device(x))

    def _init(self):
        if self.accelerate:
            self.z = self.x.copy()
            self.t = 1

        if self.accelerate or self.proxg is not None:
            self.x_old = self.x.copy()

        self.residual = np.infty

    def _update(self):
        if self.accelerate or self.proxg is not None:
            self.x_old[:] = self.x

        if self.accelerate:
            self.x[:] = self.z

        gradf_x = self.P(self.gradf(self.x))
        util.axpy(self.x, -self.alpha, gradf_x)

        if self.proxg is not None:
            self.x[:] = self.proxg(self.alpha, self.x)

        if self.accelerate:
            t_old = self.t
            self.t = (1 + (1 + 4 * t_old**2)**0.5) / 2
            self.z[:] = self.x + (t_old - 1) / self.t * (self.x - self.x_old)

        if self.accelerate or self.proxg is not None:
            self.residual = util.norm(self.x - self.x_old) / self.alpha
        else:
            self.residual = util.norm(gradf_x)

    def _print(self):
        self.logger.debug('Iteration={iter}/{max_iter}, Residual={residual}'.format(
            iter=self.iter, max_iter=self.max_iter, residual=self.residual))

    def _cleanup(self):
        if self.accelerate:
            del self.z
            del self.t

        if self.accelerate or self.proxg is not None:
            del self.x_old


class ConjugateGradient(Alg):
    r"""Conjugate Gradient Method. Solves for:

    .. math:: A x = b
    where A is hermitian.

    Args:
        A (function): A hermitian linear function.
        b (array): Observation.
        x (array): Variable.
        P (function): Preconditioner.
        max_iter (int): Maximum number of iterations.
    """

    def __init__(self, A, b, x, P=lambda x: x, max_iter=100):
        self.A = A
        self.P = P
        self.x = x
        self.b = b
        self.rzold = np.infty

        super().__init__(max_iter, util.get_device(x))

    def _init(self):
        self.r = self.b - self.A(self.x)
        z = self.P(self.r)
        if self.max_iter > 1:
            self.p = z.copy()
        else:
            self.p = z

        self.zero_gradient = False

        self.rzold = util.dot(self.r, z)
        self.residual = util.move(self.rzold**0.5, util.cpu_device)

    def _update(self):
        Ap = self.A(self.p)
        pAp = util.dot(self.p, Ap)

        if pAp == 0:
            self.zero_gradient = True
            return

        self.alpha = self.rzold / pAp
        util.axpy(self.x, self.alpha, self.p)

        if self.iter < self.max_iter - 1:
            util.axpy(self.r, -self.alpha, Ap)

            z = self.P(self.r)
            rznew = util.dot(self.r, z)
            beta = rznew / self.rzold

            util.xpay(self.p, beta, z)

            self.rzold = rznew

        self.residual = util.move(self.rzold**0.5, util.cpu_device)

    def _done(self):
        return (self.iter >= self.max_iter) or self.zero_gradient or self.residual == 0

    def _print(self):
        self.logger.debug('Iteration={iter}/{max_iter}, Residual={residual}'.format(
            iter=self.iter, max_iter=self.max_iter, residual=self.residual))

    def _cleanup(self):
        del self.r
        del self.p
        del self.rzold


class NewtonsMethod(Alg):
    r"""Newton's Method with composite self-concordant formulation.

    Considers the objective function:
    
    .. math:: f(x) + g(x),
    where f is smooth and g is simple.

    Args:
        gradf (function): Function to compute gradient of f.
        hessf (function): Function to compute Hessian of f at x,
        proxHg (function): Function to compute proximal operator of g.
        x (array): Optimization variable.
    """

    def __init__(self, gradf, hessf, proxHg, x,
                 max_iter=10, sigma=(3 - 5**0.5) / 2):

        self.gradf = gradf
        self.hessf = hessf
        self.proxHg = proxHg
        self.sigma = sigma
        self.x = x
        self.lamda = np.infty

        super().__init__(max_iter, util.get_device(x))

    def _update(self):

        hessfx = self.hessf(self.x)

        s = self.proxHg(hessfx, hessfx(self.x) - self.gradf(self.x))

        d = s - self.x

        self.lamda = util.dot(d, hessfx(d))**0.5
        if self.lamda >= self.sigma:
            alpha = 1 / (1 + self.lamda)
            self.logger.debug(u'Damped region: lamda={} > {}'.
                              format(self.lamda, self.sigma))
        else:
            alpha = 1
            self.logger.debug(u'Full-step region: lamda={} < {}'
                              .format(self.lamda, self.sigma))

        self.x += alpha * d


class PrimalDualHybridGradient(Alg):
    r"""Primal dual hybrid gradient.

    Considers the problem:

    .. math:: \min_x \max_y g(x) - f^*(u) + <Ax, u>

    Or equivalently:

    .. math:: \min_x f(A x) + g(x)

    Args:
        proxfc (function): Function to compute proximal operator of f^*.
        proxg (function): Function to compute proximal operator of g.
        A (function): Function to compute linear mapping A.
        AH (function): Function to compute adjoint linear mapping of A.
        x (array): Primal solution.
        u (array): Dual solution.
        tau (float): Primal step-size.
        sigma (float): Dual step-size.
        theta (float): Primal extrapolation parameter.
        P (function): Function to compute precondition x.
        D (function): Function to compute precondition u.
        max_iter (int): Maximum number of iterations.
    """

    def __init__(
            self, proxfc, proxg, A, AH, x, u,
            tau, sigma, theta, P=lambda x: x, D=lambda x: x, max_iter=100
    ):

        self.proxfc = proxfc
        self.proxg = proxg

        self.A = A
        self.AH = AH

        self.u = u
        self.x = x

        self.tau = tau
        self.sigma = sigma
        self.theta = theta

        self.P = P
        self.D = D

        super().__init__(max_iter, util.get_device(x))

    def _init(self):
        self.x_ext = self.x.copy()
        self.u_old = self.u.copy()
        self.x_old = self.x.copy()

    def _update(self):
        self.u_old[:] = self.u
        self.x_old[:] = self.x

        self.u[:] = self.proxfc(self.sigma, self.u + self.sigma * self.D(self.A(self.x_ext)))

        self.x[:] = self.proxg(self.tau, self.x - self.tau * self.P(self.AH(self.u)))

        self.x_ext[:] = self.x + self.theta * (self.x - self.x_old)

    def _cleanup(self):
        del self.x_ext
        del self.u_old
        del self.x_old


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

        super().__init__(max_iter, util.cpu_device)

    def _update(self):

        self.min1()
        self.min2()
