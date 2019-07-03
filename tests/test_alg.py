import unittest
import numpy as np
import numpy.testing as npt
from sigpy import alg

if __name__ == '__main__':
    unittest.main()


class TestAlg(unittest.TestCase):
    def Ax_setup(self, n):
        A = np.eye(n) + 0 * np.ones([n, n])
        x = np.arange(n, dtype=np.float)
        return A, x

    def Ax_y_setup(self, n, lamda):
        A, x = self.Ax_setup(n)
        y = A @ x
        x_numpy = np.linalg.solve(
            A.T @ A + lamda * np.eye(n), A.T @ y)

        return A, x_numpy, y

    def test_PowerMethod(self):
        n = 5
        A, x = self.Ax_setup(n)
        x_hat = np.random.random([n, 1])
        alg_method = alg.PowerMethod(lambda x: A.T @ A @ x, x_hat)
        while(not alg_method.done()):
            alg_method.update()

        s_numpy = np.linalg.svd(A, compute_uv=False)[0]
        s_sigpy = np.linalg.norm(A @ x_hat)
        npt.assert_allclose(s_numpy, s_sigpy, atol=1e-3)

    def test_GradientMethod(self):
        n = 5
        lamda = 0.1
        A, x_numpy, y = self.Ax_y_setup(n, lamda)

        # Compute step-size
        lipschitz = np.linalg.svd(
            A.T @ A + lamda * np.eye(n), compute_uv=False)[0]
        alpha = 1.0 / lipschitz

        for accelerate in [True, False]:
            for proxg in [None, lambda alpha, x: x / (1 + lamda * alpha)]:
                with self.subTest(accelerate=accelerate,
                                  proxg=proxg):
                    x_sigpy = np.zeros([n])

                    def gradf(x):
                        gradf_x = A.T @ (A @ x - y)
                        if proxg is None:
                            gradf_x += lamda * x

                        return gradf_x

                    alg_method = alg.GradientMethod(
                        gradf,
                        x_sigpy,
                        alpha,
                        accelerate=accelerate,
                        proxg=proxg,
                        max_iter=1000)

                    while(not alg_method.done()):
                        alg_method.update()

                    npt.assert_allclose(x_sigpy, x_numpy)

    def test_ConjugateGradient(self):
        n = 5
        lamda = 0.1
        A, x_numpy, y = self.Ax_y_setup(n, lamda)
        x = np.zeros([n])
        alg_method = alg.ConjugateGradient(
            lambda x: A.T @ A @ x + lamda * x,
            A.T @ y, x, max_iter=1000)
        while(not alg_method.done()):
            alg_method.update()

        npt.assert_allclose(x, x_numpy)

    def test_PrimalDualHybridGradient(self):
        n = 5
        lamda = 0.1
        A, x_numpy, y = self.Ax_y_setup(n, lamda)

        # Compute step-size
        lipschitz = np.linalg.svd(np.matmul(A.T, A), compute_uv=False)[0]
        tau = 1.0 / lipschitz
        sigma = 1.0

        x = np.zeros([n])
        u = np.zeros([n])
        alg_method = alg.PrimalDualHybridGradient(
            lambda alpha, u: (u - alpha * y) / (1 + alpha),
            lambda alpha, x: x / (1 + lamda * alpha),
            lambda x: A @ x,
            lambda x: A.T @ x,
            x, u, tau, sigma, max_iter=1000)
        while(not alg_method.done()):
            alg_method.update()

        npt.assert_allclose(x, x_numpy)

    def test_AugmentedLagrangianMethod(self):
        n = 5
        lamda = 0.1
        A, x_numpy, y = self.Ax_y_setup(n, lamda)

        # Solve 1 / 2 \| A x - y \|_2^2 + lamda * \| z \|_2^2 s.t. x = z
        mu = 1
        x_z = np.zeros([2 * n])
        v = np.zeros([n])

        def minL(mu):
            x = x_z[:n]
            z = x_z[n:]
            x[:] = np.linalg.solve(
                A.T @ A + mu * np.eye(n), A.T @ y - v + mu * z)
            z[:] = (mu * x + v) / (mu + lamda)

        def h(x_z):
            x = x_z[:n]
            z = x_z[n:]
            return x - z

        alg_method = alg.AugmentedLagrangianMethod(
            minL, None, h, x_z, None, v, mu)

        while(not alg_method.done()):
            alg_method.update()

        x = x_z[:n]
        npt.assert_allclose(x, x_numpy)

    def test_NewtonsMethod(self):
        n = 5
        lamda = 0.1
        A, x_numpy, y = self.Ax_y_setup(n, lamda)

        def gradf(x):
            gradf_x = A.T @ (A @ x - y)
            gradf_x += lamda * x

            return gradf_x

        def inv_hessf(x):
            I = np.eye(n)
            return lambda x: np.linalg.pinv(A.T @ A + lamda * I) @ x
        for beta in [1, 0.5]:
            with self.subTest(beta=beta):
                if beta < 1:
                    def f(x):
                        f_x = 1 / 2 * np.linalg.norm(A @ x - y)**2
                        f_x += lamda / 2 * np.linalg.norm(x)**2

                        return f_x
                else:
                    f = None

                x = np.zeros(n)
                alg_method = alg.NewtonsMethod(
                    gradf, inv_hessf, x,
                    beta=beta, f=f)
                while (not alg_method.done()):
                    alg_method.update()

                npt.assert_allclose(x, x_numpy)

    def test_BarrierMethod(self):

        n = 5
        lamda = 0
        A, x_numpy, y = self.Ax_y_setup(n, lamda)

        #parameters for minimization procedure
        beta = 0.5
        c = 1e-3
        beta_barrier = 1.075
        tol = 1E-3 # unused

        def g(x):
            b = 3
            inequality = x - b
            #inequality[inequality > 0] = 0
            return inequality


        # minL method is a Newton's method optimization with barrier included
        def minL(B, x):

            def gradf(x):
                gradf_x = A.T @ (A @ x - y) - (1/g(x))  # analytic
                gradf_x += lamda * x
                return gradf_x

            def inv_hessf(x):
                I = np.eye(n)
                Bhess = ((1) / (g(x) ** 2) - 0 / g(x))
                return lambda x: np.linalg.pinv(A.T @ A + lamda * I +Bhess) @ x

            if beta < 1:
                def f(x):
                    f_x = 1 / 2 * np.linalg.norm(A @ x - y) ** 2
                    f_x += lamda / 2 * np.linalg.norm(x) ** 2
                    #f_x += B # add barrier to cost

                    return f_x
            else:
                f = None

            minL_method = alg.NewtonsMethod(
                gradf, inv_hessf, x,
                beta=beta, f=f, max_iter = 1)
            while (not minL_method.done()):
                minL_method.update()


        x = np.zeros(n)

        alg_method = alg.BarrierMethod(
            minL, g, x, c, beta_barrier, tol, max_iter=1000, method='log-barrier')

        while (not alg_method.done()):
            alg_method.update()

        print(x)
        print(x_numpy)

        npt.assert_allclose(x, x_numpy)
