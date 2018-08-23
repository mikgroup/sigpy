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

        npt.assert_allclose(app.MaxEig(A.H * A, max_iter=100).run(), s[0]**2, atol=1e-2)

    def test_LinearLeastSquares(self):
        n = 5
        mat = np.eye(n) + 0.1 * util.randn([n, n])
        A = linop.MatMul([n, 1], mat)
        x = util.randn([n, 1])
        y = A(x)
        x_lstsq = np.linalg.lstsq(mat, y, rcond=None)[0]

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
        
    def test_l2reg_LinearLeastSquares(self):
        n = 5
        mat = np.eye(n) + 0.1 * util.randn([n, n])
        A = linop.MatMul([n, 1], mat)
        x = util.randn([n, 1])
        y = A(x)
        lamda = 0.1
        x_lstsq = np.linalg.solve(np.matmul(mat.conjugate().T, mat) + lamda * np.eye(n),
                                  np.matmul(mat.conjugate().T, y))

        x_rec = util.zeros([n, 1])
        app.LinearLeastSquares(A, y, x_rec, lamda=lamda).run()
        npt.assert_allclose(x_rec, x_lstsq)

        x_rec = util.zeros([n, 1])
        app.LinearLeastSquares(
            A, y, x_rec, alg_name='GradientMethod', max_iter=1000, lamda=lamda).run()
        npt.assert_allclose(x_rec, x_lstsq)
        
    def test_l2reg_bias_LinearLeastSquares(self):
        n = 5
        mat = np.eye(n) + 0.1 * util.randn([n, n])
        A = linop.MatMul([n, 1], mat)
        x = util.randn([n, 1])
        y = A(x)
        z = util.randn([n, 1])
        lamda = 0.1
        mu = 0.01
        x_lstsq = np.linalg.solve(np.matmul(mat.conjugate().T, mat) + (lamda + mu) * np.eye(n),
                                  np.matmul(mat.conjugate().T, y) + mu * z)

        x_rec = util.zeros([n, 1])
        app.LinearLeastSquares(A, y, x_rec, lamda=lamda, mu=mu, z=z).run()
        npt.assert_allclose(x_rec, x_lstsq)

        x_rec = util.zeros([n, 1])
        app.LinearLeastSquares(
            A, y, x_rec, alg_name='GradientMethod', max_iter=1000, lamda=lamda, mu=mu, z=z).run()
        npt.assert_allclose(x_rec, x_lstsq)
        
    def test_proxg_LinearLeastSquares(self):
        n = 5
        mat = np.eye(n) + 0.1 * util.randn([n, n])
        A = linop.MatMul([n, 1], mat)
        x = util.randn([n, 1])
        y = A(x)
        lamda = 0.1
        x_lstsq = np.linalg.solve(np.matmul(mat.conjugate().T, mat) + lamda * np.eye(n),
                                  np.matmul(mat.conjugate().T, y))

        proxg = prox.L2Reg([n, 1], lamda)
        x_rec = util.zeros([n, 1])
        app.LinearLeastSquares(
            A, y, x_rec, alg_name='GradientMethod', max_iter=1000, proxg=proxg).run()
        npt.assert_allclose(x_rec, x_lstsq)

        x_rec = util.zeros([n, 1])
        app.LinearLeastSquares(A, y, x_rec, max_iter=1000, proxg=proxg,
                               alg_name='PrimalDualHybridGradient').run()
        npt.assert_allclose(x_rec, x_lstsq)
        
    def test_l2reg_proxg_LinearLeastSquares(self):
        n = 5
        mat = np.eye(n) + 0.1 * util.randn([n, n])
        A = linop.MatMul([n, 1], mat)
        x = util.randn([n, 1])
        y = A(x)
        lamda = 0.1
        x_lstsq = np.linalg.solve(np.matmul(mat.conjugate().T, mat) + 2 * lamda * np.eye(n),
                                  np.matmul(mat.conjugate().T, y))

        proxg = prox.L2Reg([n, 1], lamda)
        x_rec = util.zeros([n, 1])
        app.LinearLeastSquares(
            A, y, x_rec, alg_name='GradientMethod', max_iter=1000, lamda=lamda, proxg=proxg).run()
        npt.assert_allclose(x_rec, x_lstsq)
        
    def test_l2reg_bias_proxg_LinearLeastSquares(self):
        n = 5
        mat = np.eye(n) + 0.1 * util.randn([n, n])
        A = linop.MatMul([n, 1], mat)
        x = util.randn([n, 1])
        y = A(x)
        z = util.randn([n, 1])
        lamda = 0.1
        mu = 0.01
        x_lstsq = np.linalg.solve(np.matmul(mat.conjugate().T, mat) + (2 * lamda + mu) * np.eye(n),
                                  np.matmul(mat.conjugate().T, y) + mu * z)

        proxg = prox.L2Reg([n, 1], lamda)
        x_rec = util.zeros([n, 1])
        app.LinearLeastSquares(
            A, y, x_rec, alg_name='GradientMethod', max_iter=1000, lamda=lamda, mu=mu, z=z,
            proxg=proxg).run()
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
        x_lstsq = np.linalg.lstsq(weights**0.5 * mat, weights**0.5 * y, rcond=None)[0]

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
        x_lstsq = np.linalg.lstsq(mat, y, rcond=None)[0]
        p = 1 / (np.sum(abs(mat)**2, axis=0).reshape([n, 1]) + 1e-11)
        P = linop.Multiply([n, 1], p)

        x_rec = util.zeros([n, 1])
        app.LinearLeastSquares(A, y, x_rec).run()
        npt.assert_allclose(x_rec, x_lstsq)

        x_rec = util.zeros([n, 1])
        app.LinearLeastSquares(A, y, x_rec, alg_name='GradientMethod',
                               max_iter=1000, P=P).run()
        npt.assert_allclose(x_rec, x_lstsq)

        x_rec = util.zeros([n, 1])
        app.LinearLeastSquares(A, y, x_rec, alg_name='PrimalDualHybridGradient',
                               max_iter=1000, P=P).run()
        npt.assert_allclose(x_rec, x_lstsq)

    def test_dual_precond_LinearLeastSquares(self):
        n = 5
        mat = np.eye(n) + 0.1 * util.randn([n, n])
        A = linop.MatMul([n, 1], mat)
        x = util.randn([n, 1])
        y = A(x)
        x_lstsq = np.linalg.lstsq(mat, y, rcond=None)[0]

        d = 1 / np.sum(abs(mat)**2, axis=1).reshape([n, 1])
        D = linop.Multiply([n, 1], d)
        proxfc_D = lambda alpha, x: (x - alpha * d * y) / (1 + alpha * d)
        
        x_rec = util.zeros([n, 1])
        app.LinearLeastSquares(A, y, x_rec, alg_name='PrimalDualHybridGradient',
                               max_iter=1000, D=D, proxfc_D=proxfc_D).run()
        npt.assert_allclose(x_rec, x_lstsq)
