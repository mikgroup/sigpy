import unittest
import numpy as np
import numpy.testing as npt
from sigpy import app, linop, util

if __name__ == '__main__':
    unittest.main()


class TestApp(unittest.TestCase):

    def test_MaxEig(self):
        n = 5
        mat = util.randn([n, n])
        A = linop.MatMul([n, 1], mat)
        s = np.linalg.svd(mat, compute_uv=False)

        npt.assert_allclose(app.MaxEig(A.H * A).run(), s[0]**2, atol=1e-2)

    def test_LinearLeastSquares(self):
        n = 5
        mat = np.eye(n) + 0.1 * util.randn([n, n])
        A = linop.MatMul([n, 1], mat)
        x = util.randn([n, 1])
        y = A(x)
        x_lstsq = np.linalg.lstsq(mat, y)[0]

        x_rec = util.zeros([n, 1])
        app.LinearLeastSquares(A, y, x_rec).run()
        npt.assert_allclose(x_rec, x_lstsq)

        x_rec = util.zeros([n, 1])
        app.LinearLeastSquares(
            A, y, x_rec, alg_name='GradientMethod', max_iter=1000).run()
        npt.assert_allclose(x_rec, x_lstsq)

        x_rec = util.zeros([n, 1])
        app.LinearLeastSquares(A, y, x_rec, max_iter=1000,
                               alg_name='PrimalDualHybridGradient').run()
        npt.assert_allclose(x_rec, x_lstsq)

    def test_L2ConstrainedMinimization(self):
        n = 5
        mat = np.eye(n) + 0.1 * util.randn([n, n])
        A = linop.MatMul([n, 1], mat)
        x = util.randn([n, 1])
        y = A(x)

        eps = 0

        def proxg(lamda, x):
            return x / (1 + lamda)

        x_rec = util.zeros([n, 1])
        app.L2ConstrainedMinimization(A, y, x_rec, proxg, eps).run()
        npt.assert_allclose(x_rec, x)

    def test_weighted_LinearLeastSquares(self):
        n = 5
        mat = np.eye(n) + 0.1 * util.randn([n, n])
        A = linop.MatMul([n, 1], mat)
        x = util.randn([n, 1])
        y = A(x)
        weights = 1 / (np.sum(abs(mat)**2, axis=0).reshape([n, 1]) + 1e-11)
        x_lstsq = np.linalg.lstsq(weights**0.5 * mat, weights**0.5 * y)[0]

        x_rec = util.zeros([n, 1])
        app.LinearLeastSquares(A, y, x_rec, weights=weights).run()
        npt.assert_allclose(x_rec, x_lstsq)

        x_rec = util.zeros([n, 1])
        app.LinearLeastSquares(A, y, x_rec, alg_name='GradientMethod',
                               max_iter=1000, weights=weights).run()
        npt.assert_allclose(x_rec, x_lstsq)

        x_rec = util.zeros([n, 1])
        app.LinearLeastSquares(A, y, x_rec, alg_name='PrimalDualHybridGradient',
                               max_iter=1000, weights=weights).run()
        npt.assert_allclose(x_rec, x_lstsq)

    def test_precond_LinearLeastSquares(self):
        n = 5
        mat = np.eye(n) + 0.1 * util.randn([n, n])
        A = linop.MatMul([n, 1], mat)
        x = util.randn([n, 1])
        y = A(x)
        x_lstsq = np.linalg.lstsq(mat, y)[0]
        precond = 1 / (np.sum(abs(mat)**2, axis=0).reshape([n, 1]) + 1e-11)

        x_rec = util.zeros([n, 1])
        app.LinearLeastSquares(A, y, x_rec).run()
        npt.assert_allclose(x_rec, x_lstsq)

        x_rec = util.zeros([n, 1])
        app.LinearLeastSquares(A, y, x_rec, alg_name='GradientMethod',
                               max_iter=1000, precond=precond).run()
        npt.assert_allclose(x_rec, x_lstsq)

        x_rec = util.zeros([n, 1])
        app.LinearLeastSquares(A, y, x_rec, alg_name='PrimalDualHybridGradient',
                               max_iter=1000, precond=precond).run()
        npt.assert_allclose(x_rec, x_lstsq)

    def test_dual_precond_LinearLeastSquares(self):
        n = 5
        mat = np.eye(n) + 0.1 * util.randn([n, n])
        A = linop.MatMul([n, 1], mat)
        x = util.randn([n, 1])
        y = A(x)
        x_lstsq = np.linalg.lstsq(mat, y)[0]

        dual_precond = 1 / np.sum(abs(mat)**2, axis=1).reshape([n, 1])
        x_rec = util.zeros([n, 1])
        app.LinearLeastSquares(A, y, x_rec, alg_name='PrimalDualHybridGradient',
                               max_iter=1000, dual_precond=dual_precond).run()
        npt.assert_allclose(x_rec, x_lstsq)
