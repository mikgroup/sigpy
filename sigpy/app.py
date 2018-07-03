import logging
import numpy as np

from sigpy import linop, prox, alg, util, config

if config.cupy_enabled:
    import cupy as cp


class App(object):
    '''Iterative reconstruction app.
    Each App has its own Alg.

    Args:
        _alg - Alg object
    '''
    def __init__(self, _alg):
        self._alg = _alg

    def _init(self):
        return

    def _pre_update(self):
        return

    def _post_update(self):
        return

    def _summarize(self):
        return

    def _cleanup(self):
        return

    def _output(self):
        return

    def run(self):
        self._init()
        self._alg.init()
        
        while(not self._alg.done()):
            self._pre_update()
            self._alg.update()
            self._post_update()
            self._summarize()
            
        self._alg._cleanup()
        self._cleanup()
        return self._output()


class MaxEig(App):
    '''Computes maximum eigenvalue of a Linop.

    Parameters
    ----------
    A - Linop.
    dtype - data type.
    device - Device.

    Attributes
    ----------
    A - Linop.
    x - Maximum eigenvector.

    Output
    ------
    max_eig, int, Maximum eigenvalue of A.
    '''

    def __init__(self, A, dtype=np.complex, device=util.cpu_device, max_iter=30):
        self.A = A
        self.x = util.empty(A.ishape, dtype=dtype, device=device)

        _alg = alg.PowerMethod(A, self.x, max_iter=max_iter)

        super().__init__(_alg)

    def _init(self):
        util.move_to(self.x, util.randn_like(self.x))

    def _output(self):
        return self._alg.max_eig


class LinearLeastSquares(App):
    '''Solves a linear least squares problem, with optional weights and regularizations:
    minimize_x 1 / 2 || W^0.5 (A x - y) ||_2^2 + g(G x) + lamda / 2 || R x ||_2^2

    Parameters
    ----------
    A - Linop.
    y - array.
    x - array.
    proxg - Prox, proximal function of g.
    lamda - float, l2 regularization parameter.
    G - None or Linop.
    g - None or function. Only used for when save_objs is true.
    R - None or Linop.
    weights - float or array.
    alg_name - str, {'ConjugateGradient', 'GradientMethod', 'PrimalDualHybridGradient'}.
    alpha - None or float. Step size for GradientMethod.
    
    '''

    def __init__(
            self, A, y, x, proxg=None,
            lamda=0, G=None, g=None, R=None, weights=1,
            alg_name=None, max_iter=100, save_objs=False,
            precond=1, dual_precond=1,
            alpha=None, max_power_iter=10, accelerate=True,
            tau=1, sigma=1, theta=1,
    ):
        self.A = A
        self.y = y
        self.x = x
        self.g = g
        self.G = G
        self.R = R
        self.weights = weights
        self.proxg = proxg
        self.alpha = alpha
        self.max_power_iter = max_power_iter
        self.lamda = lamda
        
        self.tau = tau
        self.sigma = sigma
        self.theta = theta
        
        self.save_objs = save_objs
        
        self.precond = precond
        self.dual_precond = dual_precond
        self.alg_name = alg_name
        self.max_iter = max_iter
        self.accelerate = accelerate

        self._get_alg_name()
        self._get_alg()
                
    def _get_alg_name(self):
        if self.alg_name is None:
            if self.proxg is None:
                self.alg_name = 'ConjugateGradient'
            elif self.G is None:
                self.alg_name = 'GradientMethod'
            else:
                self.alg_name = 'PrimalDualHybridGradient'

    def _get_alg(self):
        if self.alg_name == 'ConjugateGradient':
            self._get_ConjugateGradient()
        elif self.alg_name == 'DualConjugateGradient':
            self._get_DualConjugateGradient()
        elif self.alg_name == 'GradientMethod':
            self._get_GradientMethod()
        elif self.alg_name == 'PrimalDualHybridGradient':
            if self.G is None:
                self._get_PrimalDualHybridGradient1()
            else:
                self._get_PrimalDualHybridGradient2()
        else:
            raise ValueError('Invalid alg_name: {alg_name}.'.format(alg_name=self.alg_name))

    def gradf(self, x):
        with util.get_device(x):
            gradf_x = self.A.H(self.weights * (self.A(x) - self.y))
        
            if self.lamda != 0:
                if self.R is None:
                    util.axpy(gradf_x, self.lamda, x)
                else:
                    util.axpy(gradf_x, self.lamda, self.R.H(self.R(x)))

            return gradf_x

    def _get_AHA(self):
        I = linop.Identity(self.x.shape)
        W = linop.Multiply(self.A.oshape, self.weights)
        
        self.AHA = self.A.H * W * self.A
        if self.lamda != 0:
            if self.R is None:
                self.AHA += self.lamda * I
            else:
                self.AHA += self.lamda * self.R.H * self.R
                
    def _get_AHy(self):
        with util.get_device(self.y):
            self.AHy = self.A.H(self.weights * self.y)
        
    def _get_ConjugateGradient(self):
        self._get_AHA()
        P = linop.Multiply(self.x.shape, self.precond)            
        _alg = alg.ConjugateGradient(self.AHA, None, self.x, P=P, max_iter=self.max_iter)

        super().__init__(_alg)
        
    def _get_AAH(self):
        I = linop.Identity(self.y.shape)
        W_half = linop.Multiply(self.A.oshape, self.weights**0.5)
        
        self.AAH = W_half * self.A * self.A.H * W_half
        if self.lamda != 0:
            if self.R is None:
                self.AAH += self.lamda * I
            else:
                self.AAH += self.lamda * self.R * self.R.H
        
    def _get_DualConjugateGradient(self):
        self._get_AAH()
        P = linop.Multiply(self.y.shape, self.dual_precond)
        self.u = util.zeros_like(self.y)
        _alg = alg.ConjugateGradient(self.AAH, self.y, self.u, P=P, max_iter=self.max_iter)

        super().__init__(_alg)
        
    def _get_GradientMethod(self):
        self._get_AHA()

        if self.alpha is None:
            P = linop.Multiply(self.x.shape, self.precond)
            self.max_eig_app = MaxEig(P * self.AHA, max_iter=self.max_power_iter,
                                      dtype=self.x.dtype, device=util.get_device(self.x))
            
        _alg = alg.GradientMethod(self.gradf, self.x, self.alpha, proxg=self.proxg,
                                  max_iter=self.max_iter, accelerate=self.accelerate)
        super().__init__(_alg)

    def _get_PrimalDualHybridGradient1(self):
        device = util.get_device(self.x)
        with device:
            weights_sqrt = self.weights**0.5
            precond_sqrt = self.precond**0.5
            
        W_sqrt = linop.Multiply(self.A.oshape, weights_sqrt)
        A = W_sqrt * self.A
        P_sqrt = linop.Multiply(self.x.shape, self.precond**0.5)
        D = linop.Multiply(self.y.shape, self.dual_precond)
        self.max_eig_app = MaxEig(P_sqrt * A.H * D * A * P_sqrt,
                                  dtype=self.x.dtype, device=device,
                                  max_iter=self.max_power_iter)
        
        proxfc = prox.Conj(prox.L2Reg(self.y.shape, 1, y=weights_sqrt * self.y))
        if self.proxg is None:
            self.proxg = prox.NoOp(self.x.shape)
            
        self.u = util.zeros_like(self.y)
        _alg = alg.PrimalDualHybridGradient(
            proxfc, self.proxg, A, A.H, self.x, self.u,
            self.tau, self.sigma, self.theta, max_iter=self.max_iter)
        super().__init__(_alg)

    def _get_PrimalDualHybridGradient2(self):
        device = util.get_device(self.x)
        with device:
            weights_sqrt = self.weights**0.5
            precond_sqrt = self.precond**0.5
            
        W_sqrt = linop.Multiply(self.A.oshape, weights_sqrt)
        AG = linop.Vstack([W_sqrt * self.A, self.G])
        P_sqrt = linop.Multiply(self.x.shape, precond_sqrt)
        D = linop.Multiply(AG.oshape, self.dual_precond)
        self.max_eig_app = MaxEig(
            P_sqrt * AG.H * D * AG * P_sqrt,
            dtype=self.x.dtype, device=util.get_device(self.x),
            max_iter=self.max_power_iter)
        
        if self.proxg is None:
            self.proxg = prox.NoOp(self.x.shape)
            
        proxf1 = prox.L2Reg(self.y.shape, 1, y=self.y)
        proxf2 = self.proxg
        proxfc = prox.Conj(prox.Stack([proxf1, proxf2]))
        proxg = prox.NoOp(self.x.shape)
        
        self.u = util.zeros(AG.oshape, dtype=self.x.dtype, device=util.get_device(self.x))
        _alg = alg.PrimalDualHybridGradient(
            proxfc, proxg, AG, AG.H, self.x, self.u,
            self.tau, self.sigma, self.theta, max_iter=self.max_iter)
        
        super().__init__(_alg)

    def _get_GradientMethod_alpha(self):
        if self.alpha is None:
            lipschitz = self.max_eig_app.run()

            with util.get_device(self.x):
                self._alg.alpha = self.precond / lipschitz

    def _init(self):
        with util.get_device(self.x):
            if self.alg_name == 'GradientMethod':
                self._get_AHy()
                self._get_GradientMethod_alpha()
                    
            elif self.alg_name == 'PrimalDualHybridGradient':
                lipschitz = self.max_eig_app.run()
                self._alg.tau *= self.precond
                self._alg.sigma *= self.dual_precond / lipschitz
            
            elif self.alg_name == 'ConjugateGradient':
                self._get_AHy()
                self._alg.b = self.AHy

    def _output(self):
        if self.alg_name == 'DualConjugateGradient':
            util.move_to(self.x, self.A.H(self.weights * self.u))
            
        return self.x
    
    def objective(self):
        if self.alg_name == 'DualConjugateGradient':
            util.move_to(self.x, self.A.H(self.weights * self.u))

        xp = self._alg.device.xp
        with self._alg.device:
            l2loss = 1 / 2 * xp.sum(xp.abs(self.weights * (self.A(self.x) - self.y))**2)

            if self.R is None:
                l2reg = self.lamda / 2 * xp.sum(xp.abs(self.x)**2)
            else:
                l2reg = self.lamda / 2 * xp.sum(xp.abs(self.R(self.x)**2))
            
            if self.g is None:
                return l2loss + l2reg
            elif self.G is None:
                return l2loss + self.g(self.x) + l2reg
            else:
                return l2loss + self.g(self.G(self.x)) + l2reg

    def _summarize(self):
        if self.save_objs:
            if self._alg.iter == 1:
                self.objs = [self.objective()]
            else:
                self.objs.append(self.objective())

    def _cleanup(self):
        if self.alg_name == 'ConjugateGradient':
            del self._alg.b
        elif self.alg_name == 'GradientMethod':
            del self.AHy


class SecondOrderConeConstraint(App):
    '''
    Solves:
    min g(G x)
    s.t.  ||A x - y||_2 <= eps

    Parameters
    ----------
    A - Linop
    y - numpy array
    proxg - Prox of g.
    eps - float
    '''

    def __init__(self, A, y, x, proxg, eps, G=None, weights=1,
                 max_iter=100, precond=1, dual_precond=1,
                 tau=1, sigma=1, theta=1):

        self.x = x

        W_sqrt = linop.Multiply(A.oshape, weights**0.5)
        A = W_sqrt * A
        P = linop.Multiply(A.ishape, precond)

        if G is None:
            D = linop.Multiply(A.oshape, dual_precond)
            self.max_eig_app = MaxEig(P * A.H * D * A,
                                      dtype=x.dtype, device=util.get_device(x))
            
            proxfc = prox.Conj(prox.L2Proj(A.oshape, eps, y=y))
            
            self.u = util.zeros_like(y)
            
            _alg = alg.PrimalDualHybridGradient(proxfc, proxg, A, A.H, self.x, self.u,
                                                tau * precond, sigma * dual_precond, theta,
                                                max_iter=max_iter)
        else:
            AG = linop.Vstack([A, G])
            D = linop.Multiply(AG.oshape, dual_precond)
            self.max_eig_app = MaxEig(P * AG.H * D * AG,
                                      dtype=x.dtype, device=util.get_device(x))
            
            proxf1 = prox.L2Proj(A.oshape, np.sqrt(y.size) * eps, y=y)
            proxf2 = proxg
            proxfc = prox.Conj(prox.Stack([proxf1, proxf2]))
            proxg = prox.NoOp(A.ishape)
            
            self.u = util.zeros(AG.oshape, dtype=x.dtype, device=util.get_device(x))
            _alg = alg.PrimalDualHybridGradient(proxfc, proxg, AG, AG.H, self.x, self.u,
                                                tau * precond, sigma * dual_precond, theta,
                                                max_iter=max_iter)
            self.iter_var = []
            
        super().__init__(_alg)

    def _init(self):
        lipschitz = self.max_eig_app.run()
        self._alg.sigma /= lipschitz

    def _output(self):
        return self.x
