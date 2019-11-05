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

        for z in [None, x.copy()]:
            for lamda in [0, 0.1]:
                for proxg in [None, prox.L2Reg([n, 1], lamda)]:
                    for solver in ['GradientMethod',
                                   'PrimalDualHybridGradient',
                                   'ConjugateGradient',
                                   'ADMM']:
                        with self.subTest(proxg=proxg,
                                          solver=solver,
                                          lamda=lamda,
                                          z=z):
                            AHA = _A.T @ _A + lamda * np.eye(n)
                            AHy = _A.T @ y
                            if proxg is not None:
                                AHA += lamda * np.eye(n)

                            if z is not None:
                                AHy = _A.T @ y + lamda * z

                            x_numpy = np.linalg.solve(AHA, AHy)
                            if (solver == 'ConjugateGradient'
                                and proxg is not None):
                                with self.assertRaises(ValueError):
                                    app.LinearLeastSquares(
                                        A, y,
                                        solver=solver,
                                        lamda=lamda,
                                        proxg=proxg,
                                        z=z,
                                        show_pbar=False).run()
                            else:
                                x_rec = app.LinearLeastSquares(
                                    A, y,
                                    solver=solver,
                                    lamda=lamda,
                                    proxg=proxg,
                                    z=z,
                                    show_pbar=False).run()

                                npt.assert_allclose(x_rec, x_numpy, atol=1e-3)

    def test_precond_LinearLeastSquares(self):
        n = 5
        _A = np.eye(n) + 0.01 * util.randn([n, n])
        A = linop.MatMul([n, 1], _A)
        x = util.randn([n, 1])
        y = A(x)
        x_lstsq = np.linalg.lstsq(_A, y, rcond=-1)[0]
        p = 1 / (np.sum(abs(_A)**2, axis=0).reshape([n, 1]))

        P = linop.Multiply([n, 1], p)
        x_rec = app.LinearLeastSquares(A, y, show_pbar=False).run()
        npt.assert_allclose(x_rec, x_lstsq, atol=1e-3)

        alpha = 1 / app.MaxEig(P * A.H * A, show_pbar=False).run()
        x_rec = app.LinearLeastSquares(
            A,
            y,
            solver='GradientMethod',
            alpha=alpha,
            max_power_iter=100,
            max_iter=1000, show_pbar=False).run()
        npt.assert_allclose(x_rec, x_lstsq, atol=1e-3)

        tau = p
        x_rec = app.LinearLeastSquares(
            A,
            y,
            solver='PrimalDualHybridGradient',
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
            solver='PrimalDualHybridGradient',
            max_iter=1000,
            sigma=d, show_pbar=False).run()
        npt.assert_allclose(x_rec, x_lstsq, atol=1e-3)
