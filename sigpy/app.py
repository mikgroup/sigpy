import numpy as np

from sigpy import linop, prox, util, config
from sigpy.alg import PowerMethod, GradientMethod, ConjugateGradient, PrimalDualHybridGradient

if config.cupy_enabled:
    import cupy as cp


__all__ = ["MaxEig", "LinearLeastSquares", "SecondOrderConeConstraint"]


class App(object):
    """Iterative algorithm application. Each App has its own Alg.

    Args:
        alg (Alg)

    Attributes:
        alg (Alg)
    """

    def __init__(self, alg):
        self.alg = alg

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
        self.alg.init()

        while(not self.alg.done()):
            self._pre_update()
            self.alg.update()
            self._post_update()
            self._summarize()

        self.alg._cleanup()
        self._cleanup()
        return self._output()


class MaxEig(App):
    """Computes maximum eigenvalue of a Linop.

    Args:
        A (Linop): Hermitian linear operator.
        dtype (Dtype): Data type.
        device (Device): Device.

    Attributes:
        A (Linop)
        x (int): Eigenvector with largest eigenvalue.

    Output:
        max_eig (int): Largest eigenvalue of A.
    """

    def __init__(self, A, dtype=np.complex, device=util.cpu_device, max_iter=30):
        self.x = util.empty(A.ishape, dtype=dtype, device=device)

        alg = PowerMethod(A, self.x, max_iter=max_iter)

        super().__init__(alg)

    def _init(self):
        util.move_to(self.x, util.randn_like(self.x))

    def _output(self):
        return self.alg.max_eig
    

class LinearLeastSquares(App):
    r"""Linear least squares application.

    Solves for the following problem, with optional weights and regularizations:

    .. math::
        \min_x \frac{1}{2} \| W^{0.5} (A x - y) \|_2^2 + g(G x) + \frac{\lambda}{2} \| R x \|_2^2

    Three algorithms can be used: ConjugateGradient, GradientMethod,
    and PrimalDualHybridGradient. If alg_name is None, ConjugateGradient is used
    when proxg is not specified. If proximal operator is specified,
    then GradientMethod is used when G is specified, and PrimalDualHybridGradient is
    used otherwise.

    Args:
        A (Linop): Forward model linear operator.
        y (array): Observation.
        x (array): Solution.
        proxg (Prox): Proximal operator of g.
        lamda (float): l2 regularization parameter.
        g (None or function): Regularization function. Only used for when save_objs is true.
        G (None or Linop): Regularization linear operator.
        R (None or Linop): l2 regularization linear operator.
        weights (float or array): Weights for least squares.
        alg_name (str): {'ConjugateGradient', 'GradientMethod', 'PrimalDualHybridGradient'}.
        alpha (None or float): Step size for GradientMethod.
        accelerate (bool): Toggle Nesterov acceleration for GradientMethod.
        max_power_iter (int): Maximum number of iterations for power method. 
            Used for GradientMethod and PrimalDualHybridGradient when alpha is not specified.
        tau (float): Primal step-size for PrimalDualHybridGradient.
        sigma (float): Dual step-size for PrimalDualHybridGradient.
        theta (float): Primal extrapolation parameter for PrimalDualHybridGradient.

    """
    def __init__(self, A, y, x, proxg=None,
                 lamda=0, G=None, g=None, R=None, weights=1,
                 alg_name=None, max_iter=100, save_objs=False,
                 precond=1, dual_precond=1,
                 alpha=None, max_power_iter=10, accelerate=True,
                 tau=None, sigma=None, theta=1):
        
        alg = _get_LLS_alg(alg_name, A, y, x, lamda, R, max_iter,
                           proxg, G, weights, precond, dual_precond,
                           alpha, accelerate, tau, sigma, theta)

        self.A = A
        self.y = y
        self.x = x
        self.weights = weights
        self.precond = precond
        self.dual_precond = dual_precond
        self.device = util.get_device(x)
        
        if isinstance(alg, GradientMethod) and alpha is None or \
           isinstance(alg, PrimalDualHybridGradient) and (tau is None or sigma is None):
            self.max_eig_app = _get_LLS_max_eig_app(A, x, weights, lamda,
                                                    R, precond, dual_precond, max_power_iter)

        self.save_objs = save_objs
        if save_objs:
            self.objective = _get_LLS_objective(A, y, x, weights, lamda, R, g, G)

        super().__init__(alg)

    def _init(self):
        with self.device:
            if isinstance(self.alg, ConjugateGradient):
                self.alg.b = self.A.H(self.weights * self.y)
            elif isinstance(self.alg, GradientMethod) and self.alg.alpha is None:
                self.alg.alpha = 1 / self.max_eig_app.run()
            elif isinstance(self.alg, PrimalDualHybridGradient) and \
                 (self.alg.tau is None or self.alg.sigma is None):
                self.alg.tau = 1
                self.alg.sigma = 1 / self.max_eig_app.run()

        if self.save_objs:
            self.objs = []

    def _summarize(self):
        if self.save_objs:
            self.objs.append(self.objective())

    def _output(self):
        return self.x


class L2ConstrainedMinimization(App):
    """L2 contrained minimization application.

    Solves for problem:
    min g(G x)
    s.t. ||A x - y||_2 <= eps

    Args:
        A (Linop): Forward model linear operator.
        y (array): Observation.
        proxg (Prox): Proximal operator of objective.
        eps (float): Residual.

    """

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

            alg = PrimalDualHybridGradient(proxfc, proxg, A, A.H, self.x, self.u,
                                                tau * precond, sigma * dual_precond, theta,
                                                max_iter=max_iter)
        else:
            AG = linop.Vstack([A, G])
            D = linop.Multiply(AG.oshape, dual_precond)
            self.max_eig_app = MaxEig(P * AG.H * D * AG,
                                      dtype=x.dtype, device=util.get_device(x))

            proxf1 = prox.L2Proj(A.oshape, eps, y=y)
            proxf2 = proxg
            proxfc = prox.Conj(prox.Stack([proxf1, proxf2]))
            proxg = prox.NoOp(A.ishape)

            self.u = util.zeros(AG.oshape, dtype=x.dtype,
                                device=util.get_device(x))
            alg = PrimalDualHybridGradient(proxfc, proxg, AG, AG.H, self.x, self.u,
                                                tau * precond, sigma * dual_precond, theta,
                                                max_iter=max_iter)
            self.iter_var = []

        super().__init__(alg)

    def _init(self):
        lipschitz = self.max_eig_app.run()
        self.alg.sigma /= lipschitz

    def _output(self):
        return self.x

    
def _get_LLS_alg(alg_name, A, y, x, lamda, R, max_iter,
                 proxg, G, weights, precond, dual_precond,
                 alpha, accelerate, tau, sigma, theta):
    
    if alg_name is None:
        if proxg is None:
            alg_name = 'ConjugateGradient'
        elif G is None:
            alg_name = 'GradientMethod'
        else:
            alg_name = 'PrimalDualHybridGradient'
            
    if alg_name == 'ConjugateGradient':
        alg = _get_LLS_ConjugateGradient(A, y, x, weights, R, lamda, precond, max_iter)
    elif alg_name == 'GradientMethod':
        alg = _get_LLS_GradientMethod(A, y, x, weights, R, lamda, precond,
                                      max_iter, alpha, proxg, accelerate)
    elif alg_name == 'PrimalDualHybridGradient':
        alg = _get_LLS_PrimalDualHybridGradient(A, y, x, proxg, G,
                                                weights, precond, dual_precond,
                                                max_iter,
                                                tau, sigma, theta)
    else:
        raise ValueError('Invalid alg_name: {alg_name}.'.format(alg_name=alg_name))

    return alg


def _get_LLS_ConjugateGradient(A, y, x, weights, R, lamda, precond, max_iter):
    device = util.get_device(x)
    I = linop.Identity(x.shape)
    W = linop.Multiply(A.oshape, weights)

    AHA = A.H * W * A
    if lamda != 0:
        if R is None:
            AHA += lamda * I
        else:
            AHA += lamda * R.H * R

    with device:
        AHy = A.H(weights * y)
        
    P = linop.Multiply(x.shape, precond)
    alg = ConjugateGradient(AHA, AHy, x, P=P, max_iter=max_iter)

    return alg


def _get_LLS_GradientMethod(A, y, x, weights, R, lamda, precond,
                            max_iter, alpha, proxg, accelerate):
    device = util.get_device(x)
    
    def gradf(x):
        with device:
            gradf_x = A.H(weights * (A(x) - y))

            if lamda != 0:
                if R is None:
                    util.axpy(gradf_x, lamda, x)
                else:
                    util.axpy(gradf_x, lamda, R.H(R(x)))

            return gradf_x

    P = linop.Multiply(x.shape, precond)
    alg = GradientMethod(
        gradf, x, alpha, proxg=proxg,
        max_iter=max_iter, accelerate=accelerate, P=P)

    return alg


def _get_LLS_PrimalDualHybridGradient(A, y, x, proxg, G,
                                      weights, precond, dual_precond, max_iter,
                                      tau, sigma, theta):

    device = util.get_device(x)
    P = linop.Multiply(x.shape, precond)
    with device:
        weights_sqrt = weights**0.5
        w_y = weights_sqrt * y
        
    W_sqrt = linop.Multiply(A.oshape, weights_sqrt)
    w_A = W_sqrt * A
    
    if proxg is None:
        proxg = prox.NoOp(x.shape)
    
    if G is None:
        D = linop.Multiply(y.shape, dual_precond)
        proxfc = prox.L2Reg(y.shape, dual_precond, y=-w_y)

        u = util.zeros_like(y)
        alg = PrimalDualHybridGradient(
            proxfc, proxg, w_A, w_A.H, x, u,
            tau, sigma, theta, P=P, D=D, max_iter=max_iter)
    else:
        AG = linop.Vstack([w_A, G])

        proxf1 = prox.L2Reg(y.shape, dual_precond, y=-w_y)
        proxf2 = proxg
        proxfc = prox.Stack([proxf1, prox.Conj(proxf2)])
        proxg = prox.NoOp(x.shape)
        
        D1 = linop.Multiply(y.shape, dual_precond)
        D2 = linop.Identity(G.oshape)
        D = linop.Diag([D1, D2])

        u = util.zeros(AG.oshape, dtype=x.dtype, device=device)
        alg = PrimalDualHybridGradient(
            proxfc, proxg, AG, AG.H, x, u,
            tau, sigma, theta, P=P, D=D, max_iter=max_iter)

    return alg


def _get_LLS_max_eig_app(A, x, weights, lamda, R,
                         precond, dual_precond, max_power_iter):
    device = util.get_device(x)

    I = linop.Identity(x.shape)
    W = linop.Multiply(A.oshape, weights)
    D = linop.Multiply(A.oshape, dual_precond)

    AHA = A.H * W * D * A
    if lamda != 0:
        if R is None:
            AHA += lamda * I
        else:
            AHA += lamda * R.H * R

    P = linop.Multiply(A.ishape, precond)
    AHA = P * AHA

    app = MaxEig(AHA, dtype=x.dtype, device=device, max_iter=max_power_iter)

    return app


def _get_LLS_objective(A, y, x, weights, lamda, R, g, G):

    device = util.get_device(x)
    xp = device.xp
    
    def objective():
        with device:
            l2loss = 1 / 2 * xp.sum(xp.abs(weights**0.5 * (A(x) - y))**2)

            if R is None:
                l2reg = lamda / 2 * xp.sum(xp.abs(x)**2)
            else:
                l2reg = lamda / 2 * xp.sum(xp.abs(R(x)**2))

            if g is None:
                return l2loss + l2reg
            elif G is None:
                return l2loss + g(x) + l2reg
            else:
                return l2loss + g(G(x)) + l2reg

    return objective
