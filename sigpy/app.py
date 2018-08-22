import numpy as np

from sigpy import linop, prox, util, config
from sigpy.alg import PowerMethod, GradientMethod, ConjugateGradient, PrimalDualHybridGradient

if config.cupy_enabled:
    import cupy as cp


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

        self.alg.cleanup()
        self._cleanup()
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
        \min_x \frac{1}{2} \| W^{1/2} (A x - y) \|_2^2 + g(G x) + 
        \frac{\lambda}{2} \| R x \|_2^2 + \frac{\mu}{2} \| x - z \|_2^2

    Three algorithms can be used: `ConjugateGradient`, `GradientMethod`,
    and `PrimalDualHybridGradient`. If `alg_name` is None, `ConjugateGradient` is used
    when `proxg` is not specified. If `proxg` is specified,
    then `GradientMethod` is used when `G` is specified, and `PrimalDualHybridGradient` is
    used otherwise.

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
        weights (float or array): Weights for least squares.
        mu (float): l2 bias regularization parameter.
        z (float or array): Bias for l2 regularization.
        alg_name (str): {`'ConjugateGradient'`, `'GradientMethod'`, `'PrimalDualHybridGradient'`}.
        max_iter (int): Maximum number of iterations.
        P (Linop): Preconditioner. Assumes proxg incorporates it if specified.
        D (Linop): Dual preconditioner. Only used for `PrimalDualHybridGradient`.
        proxD (Prox): Only used when D is specified. Proximal operator computing:

        .. math::
            \min_u \frac{1}{2} \|u - v\|_2^2 + \frac{\alpha}{2} \|D^{-1 / 2}(u - y)\|_2^2

        alpha (None or float): Step size for `GradientMethod`.
        accelerate (bool): Toggle Nesterov acceleration for `GradientMethod`.
        max_power_iter (int): Maximum number of iterations for power method. 
            Used for `GradientMethod` when `alpha` is not specified,
            and for `PrimalDualHybridGradient` when `tau` or `sigma` is not specified.
        tau (float): Primal step-size for `PrimalDualHybridGradient`.
        sigma (float): Dual step-size for `PrimalDualHybridGradient`.
        theta (float): Primal extrapolation parameter for `PrimalDualHybridGradient`.
        save_objective_values (bool): Toggle saving objective value.

    """
    def __init__(self, A, y, x, proxg=None,
                 lamda=0, G=None, g=None, R=None, weights=1, mu=0, z=0,
                 alg_name=None, max_iter=100,
                 P=None, D=None, proxfc_D=None,
                 alpha=None, max_power_iter=10, accelerate=True,
                 tau=None, sigma=None, theta=1,
                 save_objective_values=False):
        self.A = A
        self.y = y
        self.x = x
        self.proxg = proxg
        self.lamda = lamda
        self.G = G
        self.g = g
        self.R = R
        self.weights = weights
        self.mu = mu
        self.z = z
        self.alg_name = alg_name
        self.max_iter = max_iter
        self.P = P
        self.D = D
        self.proxfc_D = proxfc_D
        self.alpha = alpha
        self.max_power_iter = max_power_iter
        self.accelerate = accelerate
        self.tau = tau
        self.sigma = sigma
        self.theta = theta
        self.save_objective_values = save_objective_values
        
        self._get_alg()
        self._get_max_eig_app()

    def _init(self):
        if isinstance(self.alg, ConjugateGradient):
            with util.get_device(self.y):
                w_y = self.weights * self.y

            with util.get_device(self.x):
                self.alg.b = self.A.H(w_y)
                self.alg.b += self.mu * self.z

        elif isinstance(self.alg, GradientMethod):
            try:
                self.alg.alpha = 1 / util.move(self.max_eig_app.run())
            except AttributeError:
                pass
        elif isinstance(self.alg, PrimalDualHybridGradient):
            try:
                self.alg.sigma = 1 / util.move(self.max_eig_app.run())
                self.alg.tau = 1
            except AttributeError:
                pass

        if self.save_objective_values:
            self.objective_values = []

    def _summarize(self):
        if self.save_objective_values:
            self.objective_values.append(self.objective())

    def _output(self):
        return self.x

    def _cleanup(self):
        if isinstance(self.alg, ConjugateGradient):
            del self.alg.b
            
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
                raise ValueError('ConjugateGradient cannot have proxg specified.')

            self._get_ConjugateGradient()
        elif self.alg_name == 'GradientMethod':
            if self.G is not None:
                raise ValueError('GradientMethod cannot have G specified.')

            self._get_GradientMethod()
        elif self.alg_name == 'PrimalDualHybridGradient':
            if self.lamda != 0 or self.mu != 0:
                raise ValueError('PrimalDualHybridGradient cannot have non-zero mu or lamda.')

            self._get_PrimalDualHybridGradient()
        else:
            raise ValueError('Invalid alg_name: {alg_name}.'.format(alg_name=self.alg_name))

    def _get_ConjugateGradient(self):
        I = linop.Identity(self.x.shape)
        W = linop.Multiply(self.A.oshape, self.weights)

        AHA = self.A.H * W * self.A
        if self.lamda != 0:
            if self.R is None:
                AHA += self.lamda * I
            else:
                AHA += self.lamda * self.R.H * self.R

        if self.mu != 0:
            AHA += self.mu * I

        self.alg = ConjugateGradient(AHA, None, self.x, P=self.P, max_iter=self.max_iter)

    def _get_GradientMethod(self):
        def gradf(x):
            with util.get_device(self.y):
                r = self.A(x) - self.y
                r *= self.weights
                
            with util.get_device(self.x):
                gradf_x = self.A.H(r)

                if self.lamda != 0:
                    if self.R is None:
                        util.axpy(gradf_x, self.lamda, x)
                    else:
                        util.axpy(gradf_x, self.lamda, self.R.H(self.R(x)))

                if self.mu != 0:
                    util.axpy(gradf_x, self.mu, x - self.z)

                return gradf_x

        self.alg = GradientMethod(gradf, self.x, self.alpha, proxg=self.proxg,
                                  max_iter=self.max_iter, accelerate=self.accelerate, P=self.P)

    def _get_PrimalDualHybridGradient(self):
        with util.get_device(self.y):
            weights_sqrt = self.weights**0.5
            w_y = weights_sqrt * self.y

        W_sqrt = linop.Multiply(self.A.oshape, weights_sqrt)
        w_A = W_sqrt * self.A

        if self.proxg is None:
            self.proxg = prox.NoOp(self.x.shape)

        if self.G is None:
            if self.D is None:
                proxfc = prox.L2Reg(self.y.shape, 1, y=-w_y)
            else:
                if self.proxfc_D is None:
                    raise ValueError('proxfc_D must be specified when D is specified.')
                
                proxfc = self.proxfc_D
                
            u = util.zeros_like(self.y)
            self.alg = PrimalDualHybridGradient(proxfc, self.proxg, w_A, w_A.H, self.x, u,
                                                self.tau, self.sigma, self.theta,
                                                P=self.P, D=self.D, max_iter=self.max_iter)
        else:
            AG = linop.Vstack([w_A, self.G])

            if self.D is None:
                proxf1c = prox.L2Reg(self.y.shape, 1, y=-w_y)
            else:
                if self.proxfc_D is None:
                    raise ValueError('proxfc_D must be specified when D is specified.')
                
                proxf1c = self.proxfc_D
            
            proxf2 = self.proxg
            proxfc = prox.Stack([proxf1c, prox.Conj(proxf2)])
            proxg = prox.NoOp(self.x.shape)

            u = util.zeros(AG.oshape, dtype=self.y.dtype, device=util.get_device(self.y))
            self.alg = PrimalDualHybridGradient(proxfc, proxg, AG, AG.H, self.x, u,
                                                self.tau, self.sigma, self.theta,
                                                P=self.P, D=self.D, max_iter=self.max_iter)

    def _get_max_eig_app(self):
        if isinstance(self.alg, GradientMethod) and self.alpha is None or \
           isinstance(self.alg, PrimalDualHybridGradient) and \
           (self.tau is None or self.sigma is None):
            I = linop.Identity(self.x.shape)
            W = linop.Multiply(self.A.oshape, self.weights)
            if self.D is not None:
                W = self.D * W

            AHA = self.A.H * W * self.A

            if self.lamda != 0:
                if self.R is None:
                    AHA += self.lamda * I
                else:
                    AHA += self.lamda * self.R.H * self.R

            if self.mu != 0:
                AHA += self.mu * I

            if self.P is not None:
                AHA = self.P * AHA

            self.max_eig_app = MaxEig(AHA, dtype=self.x.dtype, device=util.get_device(self.x),
                                      max_iter=self.max_power_iter)

    def objective(self):
        device = util.get_device(self.y)
        xp = device.xp
        with device:
            r = self.A(self.x) - self.y
            r *= self.weights**0.5
            obj = 1 / 2 * xp.sum(xp.abs(r)**2)

            if self.lamda > 0:
                if self.R is None:
                    obj += self.lamda / 2 * xp.sum(xp.abs(self.x)**2)
                else:
                    obj += self.lamda / 2 * xp.sum(xp.abs(self.R(self.x))**2)

            if self.mu != 0:
                obj += self.mu / 2 * xp.sum(xp.abs(self.x - self.z)**2)

            if self.proxg is not None:
                if self.g is None:
                    raise ValueError('Cannot compute objective when proxg is specified,'
                                     'but g is not.')
                
                if self.G is None:
                    obj += self.g(self.x)
                else:
                    obj += self.g(self.G(self.x))

            return obj


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
                 max_iter=100, tau=None, sigma=None, theta=1):

        self.x = x

        W_sqrt = linop.Multiply(A.oshape, weights**0.5)
        A = W_sqrt * A

        if G is None:
            self.max_eig_app = MaxEig(A.H * A, dtype=x.dtype, device=util.get_device(x))

            proxfc = prox.Conj(prox.L2Proj(A.oshape, eps, y=y))
            self.u = util.zeros_like(y)
            alg = PrimalDualHybridGradient(proxfc, proxg, A, A.H, self.x, self.u,
                                                tau, sigma, theta,
                                                max_iter=max_iter)
        else:
            AG = linop.Vstack([A, G])
            self.max_eig_app = MaxEig(AG.H * AG,
                                      dtype=x.dtype, device=util.get_device(x))

            proxf1 = prox.L2Proj(A.oshape, eps, y=y)
            proxf2 = proxg
            proxfc = prox.Conj(prox.Stack([proxf1, proxf2]))
            proxg = prox.NoOp(A.ishape)

            self.u = util.zeros(AG.oshape, dtype=x.dtype, device=util.get_device(x))
            alg = PrimalDualHybridGradient(proxfc, proxg, AG, AG.H, self.x, self.u,
                                           tau, sigma, theta, max_iter=max_iter)
            self.iter_var = []

        super().__init__(alg)

    def _init(self):
        if self.alg.tau is None or self.alg.sigma is None:
            self.alg.tau = 1
            self.alg.sigma = 1 / self.max_eig_app.run()

    def _output(self):
        return self.x
