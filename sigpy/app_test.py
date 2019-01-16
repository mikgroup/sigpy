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

        npt.assert_allclose(app.MaxEig(A.H * A, max_iter=1000).run(), s[0]**2, atol=1e-3)

    def test_LinearLeastSquares(self):
        n = 5
        mat = np.eye(n) + 0.1 * util.randn([n, n])
        A = linop.MatMul([n, 1], mat)
        x = util.randn([n, 1])
        y = A(x)
        x_lstsq = np.linalg.lstsq(mat, y, rcond=-1)[0]

        x_rec = app.LinearLeastSquares(A, y).run()
        npt.assert_allclose(x_rec, x_lstsq, atol=1e-3)

        x_rec = app.LinearLeastSquares(
            A, y, alg_name='GradientMethod', max_power_iter=100, max_iter=1000).run()
        npt.assert_allclose(x_rec, x_lstsq, atol=1e-3)

        x_rec = app.LinearLeastSquares(A, y, alg_name='PrimalDualHybridGradient',
                                       max_iter=1000).run()
        npt.assert_allclose(x_rec, x_lstsq, atol=1e-3)
        
    def test_l2reg_LinearLeastSquares(self):
        n = 5
        mat = np.eye(n) + 0.1 * util.randn([n, n])
        A = linop.MatMul([n, 1], mat)
        x = util.randn([n, 1])
        y = A(x)
        lamda = 0.1
        x_lstsq = np.linalg.solve(np.matmul(mat.conjugate().T, mat) + lamda * np.eye(n),
                                  np.matmul(mat.conjugate().T, y))

        x_rec = app.LinearLeastSquares(A, y, lamda=lamda).run()
        npt.assert_allclose(x_rec, x_lstsq, atol=1e-3)

        x_rec = app.LinearLeastSquares(
            A, y, alg_name='GradientMethod', lamda=lamda, max_power_iter=100, max_iter=1000).run()
        npt.assert_allclose(x_rec, x_lstsq, atol=1e-3)

        x_rec = app.LinearLeastSquares(A, y, lamda=lamda,
                                       alg_name='PrimalDualHybridGradient',
                                       max_iter=1000).run()
        npt.assert_allclose(x_rec, x_lstsq, atol=1e-3)
        
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

        x_rec = app.LinearLeastSquares(A, y, lamda=lamda, mu=mu, z=z).run()
        npt.assert_allclose(x_rec, x_lstsq, atol=1e-3)

        x_rec = app.LinearLeastSquares(
            A, y, alg_name='GradientMethod', lamda=lamda, mu=mu, z=z, max_power_iter=100, max_iter=1000).run()
        npt.assert_allclose(x_rec, x_lstsq, atol=1e-3)

        x_rec = app.LinearLeastSquares(A, y, lamda=lamda, mu=mu, z=z,
                                       alg_name='PrimalDualHybridGradient',
                                       max_iter=1000).run()
        npt.assert_allclose(x_rec, x_lstsq, atol=1e-3)
        
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
        x_rec = app.LinearLeastSquares(
            A, y, alg_name='GradientMethod', proxg=proxg, max_power_iter=100, max_iter=1000).run()
        npt.assert_allclose(x_rec, x_lstsq, atol=1e-3)

        x_rec = app.LinearLeastSquares(A, y, proxg=proxg,
                                       alg_name='PrimalDualHybridGradient',
                                       max_iter=1000).run()
        npt.assert_allclose(x_rec, x_lstsq, atol=1e-3)
        
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
        
        x_rec = app.LinearLeastSquares(
            A, y, alg_name='GradientMethod', lamda=lamda, proxg=proxg, max_power_iter=100, max_iter=1000).run()
        npt.assert_allclose(x_rec, x_lstsq, atol=1e-3)

        x_rec = app.LinearLeastSquares(
            A, y, alg_name='PrimalDualHybridGradient', max_iter=1000,
            lamda=lamda, proxg=proxg).run()
        npt.assert_allclose(x_rec, x_lstsq, atol=1e-3)
        
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
        x_rec = app.LinearLeastSquares(
            A, y, alg_name='GradientMethod', lamda=lamda, mu=mu, z=z,
            proxg=proxg, max_power_iter=100, max_iter=1000).run()
        npt.assert_allclose(x_rec, x_lstsq, atol=1e-3)
        
        x_rec = app.LinearLeastSquares(
            A, y, alg_name='PrimalDualHybridGradient', max_iter=1000,
            lamda=lamda, mu=mu, z=z,
            proxg=proxg).run()
        npt.assert_allclose(x_rec, x_lstsq, atol=1e-3)

    def test_L2ConstrainedMinimization(self):
        n = 5
        mat = np.eye(n) + 0.1 * util.randn([n, n])
        A = linop.MatMul([n, 1], mat)
        x = util.randn([n, 1])
        y = A(x)

        eps = 0

        def proxg(lamda, x):
            return x / (1 + lamda)

        x_rec = app.L2ConstrainedMinimization(A, y, proxg, eps).run()
        npt.assert_allclose(x_rec, x, atol=1e-3)

    def test_precond_LinearLeastSquares(self):
        n = 5
        mat = np.eye(n) + 0.1 * util.randn([n, n])
        A = linop.MatMul([n, 1], mat)
        x = util.randn([n, 1])
        y = A(x)
        x_lstsq = np.linalg.lstsq(mat, y, rcond=-1)[0]
        p = 1 / (np.sum(abs(mat)**2, axis=0).reshape([n, 1]))

        P = linop.Multiply([n, 1], p)
        x_rec = app.LinearLeastSquares(A, y).run()
        npt.assert_allclose(x_rec, x_lstsq, atol=1e-3)

        alpha = p / app.MaxEig(P * A.H * A).run()
        x_rec = app.LinearLeastSquares(A, y, alg_name='GradientMethod',
                                       alpha=alpha, max_power_iter=100, max_iter=1000).run()
        npt.assert_allclose(x_rec, x_lstsq, atol=1e-3)

        tau = p
        x_rec = app.LinearLeastSquares(A, y, alg_name='PrimalDualHybridGradient',
                                       max_iter=1000, tau=tau).run()
        npt.assert_allclose(x_rec, x_lstsq, atol=1e-3)

    def test_dual_precond_LinearLeastSquares(self):
        n = 5
        mat = np.eye(n) + 0.1 * util.randn([n, n])
        A = linop.MatMul([n, 1], mat)
        x = util.randn([n, 1])
        y = A(x)
        x_lstsq = np.linalg.lstsq(mat, y, rcond=-1)[0]

        d = 1 / np.sum(abs(mat)**2, axis=1, keepdims=True).reshape([n, 1])        
        x_rec = app.LinearLeastSquares(A, y, alg_name='PrimalDualHybridGradient',
                                       max_iter=1000, sigma=d).run()
        npt.assert_allclose(x_rec, x_lstsq, atol=1e-3)
