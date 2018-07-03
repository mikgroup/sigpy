import numpy as np
import logging
from sigpy import util, config

if config.cupy_enabled:
    import cupy as cp

__all__ = ['PowerMethod', 'GradientMethod', 'PrimalDualHybridGradient', 'AltMin']


class Alg(object):
    '''Iterative algorithm object.
    
    Parameters
    ----------
    max_iter: int, maximum number of iterations.
    device: int or Device object, device
    '''

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
    '''Power method to estimate maximum eigenvalue and eigenvector
    of a hermitian linear operator A

    Parameter
    ---------
        A (function) - function to a hermitian linear mapping.
        x (numpy/cupy array) - variable to optimize over.
        max_iter (int) - maximum number of iterations.

    Attributes
    ----------
        max_eig (float) - maximum eigenvalue.
    '''
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
    '''First order gradient method.
    Considers the composite cost function f(x) + g(x), where f is smooth, and g is simple.

    Args:
        gradf (function) - function to compute gradient of f.
        x (numpy/cupy array) - variable to optimize over.
        alpha (float) - step size.
        proxg (function or None) - function to compute proximal mapping of g.
        accelerate (bool) - toggle Nesterov acceleration.
        max_iter (int) - maximum number of iterations.
    '''

    def __init__(self, gradf, x, alpha, proxg=None, accelerate=False, max_iter=100):
        self.gradf = gradf
        self.alpha = alpha
        self.accelerate = accelerate
        self.proxg = proxg
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

        gradf_x = self.gradf(self.x)
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
    """Newton's Method with the formulation of
    Composite Self-Concordant Minimization

    Args:
    gradf - A function gradf(x): x -> gradient of f at x
    hessf - A function H(x): x -> Hessian of f at x,
            which is another function: y -> Hessian of f at x times y
    proxHg - A function proxHg(H(x), u):
            H(x), u -> argmin_y 1/2 y^* H(x) y - u^* y + g(y)
    x - solution
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
            self.logger.debug(u'Damped region: λ={} > {}'.
                              format(self.lamda, self.sigma))
        else:
            alpha = 1
            self.logger.debug(u'Full-step region: λ={} < {}'
                              .format(self.lamda, self.sigma))

        self.x += alpha * d


class PrimalDualHybridGradient(Alg):
    '''Primal dual hybrid gradient.
    Considers the problem:
    min_x max_y G(x) - F^*(u) + <Ax, u>

    Or equivalently:
    min_x F(A x) + G(x)

    Examples
    --------
    F(A x) = 1 / 2 || A x - y ||^2
    F^*(u) = 1 / 2 || u ||^2 - < y, u >
    proxfc(v) = (v - lamda * y) / (1 + lamda)

    Examples
    --------
    F(A x) = I{|| A x - y ||_2 < e}
    F^*(u) = e * || u ||_2 - <y, u>
    proxfc(v) = (||v - lamda * y||_2 - lamda * e)_+ * (v - lamda * y)
    '''

    def __init__(
            self, proxfc, proxg, A, AH, x, u,
            tau, sigma, theta, max_iter=100
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

        super().__init__(max_iter, util.get_device(x))

    def _init(self):
        self.x_ext = self.x.copy()
        self.u_old = self.u.copy()
        self.x_old = self.x.copy()

    def _update(self):
        self.u_old[:] = self.u
        self.x_old[:] = self.x

        self.u[:] = self.proxfc(self.sigma, self.u + self.sigma * self.A(self.x_ext))

        self.x[:] = self.proxg(self.tau, self.x - self.tau * self.AH(self.u))

        self.x_ext[:] = self.x + self.theta * (self.x - self.x_old)

    def _cleanup(self):
        del self.x_ext
        del self.u_old
        del self.x_old


class AltMin(Alg):
    """Alternating Minimization.
    
    Args:
        min1 (function): function to minimize over variable 1.
        min2 (function): funciton to minimize over variable 2.
        max_iter (int): maximum number of iterations.
    """

    def __init__(self, min1, min2, max_iter=30):
        self.min1 = min1
        self.min2 = min2

        super().__init__(max_iter, util.cpu_device)

    def _update(self):

        self.min1()
        self.min2()
