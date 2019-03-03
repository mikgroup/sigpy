import unittest
import numpy as np
import numpy.testing as npt
from sigpy import app, linop, util, prox

if __name__ == '__main__':
    unittest.main()


class TestApp(unittest.TestCase):

    def test_MaxEig(self):
        n = 5
        mat = util.randn([n, n])
        A = linop.MatMul([n, 1], mat)
        s = np.linalg.svd(mat, compute_uv=False)

        npt.assert_allclose(app.MaxEig(
            A.H * A, max_iter=1000, show_pbar=False).run(), s[0]**2, atol=1e-3)

    def test_LinearLeastSquares(self):
        n = 5
        _A = np.eye(n) + 0.1 * np.ones([n, n])
        A = linop.MatMul([n, 1], _A)
        x = np.arange(n).reshape([n, 1])
        y = A(x)
        z = np.arange(n).reshape([n, 1])

        for mu in [0, 0.1]:
            for lamda in [0, 0.1]:
                for proxg in [None, prox.L2Reg([n, 1], lamda)]:
                    for alg_name in ['GradientMethod',
                                     'PrimalDualHybridGradient',
                                     'ConjugateGradient']:
                        with self.subTest(proxg=proxg,
                                          alg_name=alg_name,
                                          lamda=lamda,
                                          mu=mu):
                            if proxg is None:
                                prox_lamda = 0
                            else:
                                prox_lamda = lamda

                            x_numpy = np.linalg.solve(
                                _A.T @ _A +
                                (lamda + mu + prox_lamda) * np.eye(n),
                                _A.T @ y + mu * z)

                            if (alg_name == 'ConjugateGradient'
                                and proxg is not None):
                                with self.assertRaises(ValueError):
                                    app.LinearLeastSquares(
                                        A, y,
                                        alg_name=alg_name,
                                        lamda=lamda,
                                        proxg=proxg,
                                        mu=mu, z=z,
                                        show_pbar=False).run()
                            else:
                                x_rec = app.LinearLeastSquares(
                                    A, y,
                                    alg_name=alg_name,
                                    lamda=lamda,
                                    proxg=proxg,
                                    mu=mu, z=z,
                                    show_pbar=False).run()

                                npt.assert_allclose(x_rec, x_numpy, atol=1e-3)

    def test_L2ConstrainedMinimization(self):
        n = 5
        _A = np.eye(n) + 0.1 * util.randn([n, n])
        A = linop.MatMul([n, 1], _A)
        x = util.randn([n, 1])
        y = A(x)

        eps = 0

        def proxg(lamda, x):
            return x / (1 + lamda)

        x_rec = app.L2ConstrainedMinimization(A, y, proxg, eps,
                                              show_pbar=False).run()
        npt.assert_allclose(x_rec, x, atol=1e-3)

    def test_precond_LinearLeastSquares(self):
        n = 5
        _A = np.eye(n) + 0.1 * util.randn([n, n])
        A = linop.MatMul([n, 1], _A)
        x = util.randn([n, 1])
        y = A(x)
        x_lstsq = np.linalg.lstsq(_A, y, rcond=-1)[0]
        p = 1 / (np.sum(abs(_A)**2, axis=0).reshape([n, 1]))

        P = linop.Multiply([n, 1], p)
        x_rec = app.LinearLeastSquares(A, y, show_pbar=False).run()
        npt.assert_allclose(x_rec, x_lstsq, atol=1e-3)

        alpha = p / app.MaxEig(P * A.H * A, show_pbar=False).run()
        x_rec = app.LinearLeastSquares(
            A,
            y,
            alg_name='GradientMethod',
            alpha=alpha,
            max_power_iter=100,
            max_iter=1000, show_pbar=False).run()
        npt.assert_allclose(x_rec, x_lstsq, atol=1e-3)

        tau = p
        x_rec = app.LinearLeastSquares(
            A,
            y,
            alg_name='PrimalDualHybridGradient',
            max_iter=1000,
            tau=tau, show_pbar=False).run()
        npt.assert_allclose(x_rec, x_lstsq, atol=1e-3)

    def test_dual_precond_LinearLeastSquares(self):
        n = 5
        _A = np.eye(n) + 0.1 * util.randn([n, n])
        A = linop.MatMul([n, 1], _A)
        x = util.randn([n, 1])
        y = A(x)
        x_lstsq = np.linalg.lstsq(_A, y, rcond=-1)[0]

        d = 1 / np.sum(abs(_A)**2, axis=1, keepdims=True).reshape([n, 1])
        x_rec = app.LinearLeastSquares(
            A,
            y,
            alg_name='PrimalDualHybridGradient',
            max_iter=1000,
            sigma=d, show_pbar=False).run()
        npt.assert_allclose(x_rec, x_lstsq, atol=1e-3)
