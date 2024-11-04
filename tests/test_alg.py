import unittest

import numpy as np
import numpy.testing as npt

from sigpy import alg, linop

if __name__ == "__main__":
    unittest.main()


class TestAlg(unittest.TestCase):
    def Ax_setup(self, n):
        A = np.eye(n) + 0.1 * np.ones([n, n])
        x = np.arange(n, dtype=np.float32)
        return A, x

    def Ax_y_setup(self, n, lamda):
        A, x = self.Ax_setup(n)
        y = A @ x
        x_numpy = np.linalg.solve(A.T @ A + lamda * np.eye(n), A.T @ y)

        return A, x_numpy, y

    def test_PowerMethod(self):
        n = 5
        A, x = self.Ax_setup(n)
        x_hat = np.random.random([n, 1])
        alg_method = alg.PowerMethod(lambda x: A.T @ A @ x, x_hat)
        while not alg_method.done():
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
            A.T @ A + lamda * np.eye(n), compute_uv=False
        )[0]
        alpha = 1.0 / lipschitz

        for accelerate in [True, False]:
            for proxg in [None, lambda alpha, x: x / (1 + lamda * alpha)]:
                with self.subTest(accelerate=accelerate, proxg=proxg):
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
                        max_iter=1000,
                    )

                    while not alg_method.done():
                        alg_method.update()

                    npt.assert_allclose(x_sigpy, x_numpy)

    def test_ConjugateGradient(self):
        n = 5
        lamda = 0.1
        A, x_numpy, y = self.Ax_y_setup(n, lamda)
        x = np.zeros([n])
        alg_method = alg.ConjugateGradient(
            lambda x: A.T @ A @ x + lamda * x, A.T @ y, x, max_iter=1000
        )
        while not alg_method.done():
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
            x,
            u,
            tau,
            sigma,
            max_iter=1000,
        )
        while not alg_method.done():
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

        def minL():
            x = x_z[:n]
            z = x_z[n:]
            x[:] = np.linalg.solve(
                A.T @ A + mu * np.eye(n), A.T @ y - v + mu * z
            )
            z[:] = (mu * x + v) / (mu + lamda)

        def h(x_z):
            x = x_z[:n]
            z = x_z[n:]
            return x - z

        alg_method = alg.AugmentedLagrangianMethod(
            minL, None, h, x_z, None, v, mu
        )
        while not alg_method.done():
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
            Id = np.eye(n)
            return lambda x: np.linalg.pinv(A.T @ A + lamda * Id) @ x

        for beta in [1, 0.5]:
            with self.subTest(beta=beta):
                if beta < 1:

                    def f(x):
                        f_x = 1 / 2 * np.linalg.norm(A @ x - y) ** 2
                        f_x += lamda / 2 * np.linalg.norm(x) ** 2

                        return f_x

                else:
                    f = None

                x = np.zeros(n)
                alg_method = alg.NewtonsMethod(
                    gradf, inv_hessf, x, beta=beta, f=f
                )
                while not alg_method.done():
                    alg_method.update()

                npt.assert_allclose(x, x_numpy)

    def test_GerchbergSaxton(self):
        n = 10
        lamda = 0.1
        A, x_numpy, y = self.Ax_y_setup(n, lamda)
        y = np.expand_dims(np.csingle(y), 1)
        x_numpy = np.expand_dims(x_numpy, 1)
        A = np.csingle(A)
        A = linop.MatMul(y.shape, A)
        x0 = np.zeros(A.ishape, dtype=complex)

        alg_method = alg.GerchbergSaxton(
            A, y, x0, max_iter=100, tol=10e-9, lamb=lamda
        )

        while not alg_method.done():
            alg_method.update()

        phs = np.conj(x_numpy * alg_method.x / abs(x_numpy * alg_method.x))
        npt.assert_allclose(alg_method.x * phs, x_numpy, rtol=1e-6)

    def test_SDMM(self):
        n = 5
        lamda = 0.1
        A, x_numpy, y = self.Ax_y_setup(n, lamda)
        y = np.expand_dims(y, 1)
        A = linop.MatMul(np.expand_dims(x_numpy, 1).shape, A)

        c_norm = None
        c_max = None
        mu = 10**8  # big mu ok since no constraints used
        rho_norm = 1
        rho_max = 1
        lam = 0.1
        cg_iters = 5
        max_iter = 10

        L = []
        c = [1]
        rho = [1]
        for ii in range(len(y) - 1):
            c.append(0.00012**2)
            rho.append(0.001)

        alg_method = alg.SDMM(
            A,
            y,
            lam,
            L=L,
            c=c,
            c_max=c_max,
            c_norm=c_norm,
            mu=mu,
            rho=rho,
            rho_max=rho_max,
            rho_norm=rho_norm,
            eps_pri=10**-5,
            eps_dual=10**-2,
            max_cg_iter=cg_iters,
            max_iter=max_iter,
        )

        while not alg_method.done():
            alg_method.update()

        npt.assert_allclose(np.squeeze(abs(alg_method.x)), x_numpy)
