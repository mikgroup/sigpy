import unittest
import numpy as np
import numpy.testing as npt
from sigpy import alg

if __name__ == '__main__':
    unittest.main()


class TestAlg(unittest.TestCase):

    def test_PowerMethod(self):
        n = 5
        A = np.random.random([n, n])
        x = np.random.random([n])

        alg_method = alg.PowerMethod(lambda x: np.matmul(A.T, np.matmul(A, x)), x)
        while(not alg_method.done()):
            alg_method.update()

        s = np.linalg.svd(A, compute_uv=False)

        npt.assert_allclose(np.linalg.norm(np.matmul(A, x)), s[0], atol=1e-3)

    def test_GradientMethod(self):
        n = 5
        A = np.random.random([n, n])
        x_orig = np.random.random([n])
        y = np.matmul(A, x_orig)
        lamda = 1.0
        x_truth = np.linalg.solve(
            np.matmul(A.T, A) + lamda * np.eye(n), np.matmul(A.T, y))

        # Compute step-size
        lipschitz = np.linalg.svd(
            np.matmul(A.T, A) + lamda * np.eye(n), compute_uv=False)[0]
        alpha = 1.0 / lipschitz

        # Gradient method
        x = np.zeros([n])
        alg_method = alg.GradientMethod(lambda x: np.matmul(A.T, (np.matmul(A, x) - y)) +
                                        lamda * x, x,
                                        alpha, accelerate=False, max_iter=1000)
        while(not alg_method.done()):
            alg_method.update()

        npt.assert_allclose(x, x_truth)

        # Accelerated gradient method
        x = np.zeros([n])
        alg_method = alg.GradientMethod(lambda x: np.matmul(A.T, np.matmul(A, x) - y) +
                                        lamda * x, x,
                                        alpha, accelerate=True, max_iter=1000)
        while(not alg_method.done()):
            alg_method.update()

        npt.assert_allclose(x, x_truth)

        # Proximal gradient method
        x = np.zeros([n])
        alg_method = alg.GradientMethod(lambda x: np.matmul(A.T, np.matmul(A, x) - y), x,
                                        alpha, accelerate=False,
                                        proxg=lambda alpha, x: x / (1 + lamda * alpha),
                                        max_iter=1000)
        while(not alg_method.done()):
            alg_method.update()

        npt.assert_allclose(x, x_truth)

        # Accelerated proximal gradient method
        x = np.zeros([n])
        alg_method = alg.GradientMethod(lambda x: np.matmul(A.T, np.matmul(A, x) - y), x,
                                        alpha,
                                        proxg=lambda alpha, x: x /
                                        (1 + lamda * alpha),
                                        accelerate=True, max_iter=1000)
        while(not alg_method.done()):
            alg_method.update()

        npt.assert_allclose(x, x_truth)

    def test_GradientMethod_backtracking(self):
        n = 5
        A = np.random.random([n, n])
        x_orig = np.random.random([n])
        y = np.matmul(A, x_orig)
        lamda = 1.0
        x_truth = np.linalg.solve(
            np.matmul(A.T, A) + lamda * np.eye(n), np.matmul(A.T, y))
        alpha = 1
        beta = 0.5
        f = lambda x: 1 / 2 * np.linalg.norm(np.matmul(A, x) - y)**2 + lamda / 2 * np.linalg.norm(x)**2
        # Gradient method
        x = np.zeros([n])
        alg_method = alg.GradientMethod(lambda x: np.matmul(A.T, (np.matmul(A, x) - y)) +
                                        lamda * x, x,
                                        alpha, beta=beta, f=f, accelerate=False,
                                        max_iter=1000)
        while(not alg_method.done()):
            alg_method.update()

        npt.assert_allclose(x, x_truth)

        # Accelerated gradient method
        x = np.zeros([n])
        alg_method = alg.GradientMethod(lambda x: np.matmul(A.T, np.matmul(A, x) - y) +
                                        lamda * x, x,
                                        alpha, beta=beta, f=f, accelerate=True,
                                        max_iter=1000)
        while(not alg_method.done()):
            alg_method.update()

        npt.assert_allclose(x, x_truth)

        # Proximal gradient method
        f = lambda x: 1 / 2 * np.linalg.norm(np.matmul(A, x) - y)**2
        x = np.zeros([n])
        alg_method = alg.GradientMethod(lambda x: np.matmul(A.T, np.matmul(A, x) - y), x,
                                        alpha, accelerate=False, beta=beta, f=f,
                                        proxg=lambda alpha, x: x / (1 + lamda * alpha),
                                        max_iter=1000)
        while(not alg_method.done()):
            alg_method.update()

        npt.assert_allclose(x, x_truth)

        # Accelerated proximal gradient method
        x = np.zeros([n])
        alg_method = alg.GradientMethod(lambda x: np.matmul(A.T, np.matmul(A, x) - y), x,
                                        alpha, beta=beta, f=f,
                                        proxg=lambda alpha, x: x /
                                        (1 + lamda * alpha),
                                        accelerate=True, max_iter=1000)
        while(not alg_method.done()):
            alg_method.update()

        npt.assert_allclose(x, x_truth)

    def test_ConjugateGradient(self):
        n = 5
        A = np.random.random([n, n])
        x_orig = np.random.random([n])
        y = np.matmul(A, x_orig)
        x_truth = np.linalg.solve(np.matmul(A.T, A), np.matmul(A.T, y))

        x = np.zeros([n])
        alg_method = alg.ConjugateGradient(lambda x: np.matmul(A.T, np.matmul(A, x)),
                                           np.matmul(A.T, y),
                                           x, max_iter=1000)
        while(not alg_method.done()):
            alg_method.update()

        npt.assert_allclose(x, x_truth)

    def test_PrimalDualHybridGradient(self):
        n = 5
        A = np.random.random([n, n])
        x_orig = np.random.random([n])
        y = np.matmul(A, x_orig)
        lamda = 1.0
        x_truth = np.linalg.solve(
            np.matmul(A.T, A) + lamda * np.eye(n), np.matmul(A.T, y))

        # Compute step-size
        lipschitz = np.linalg.svd(np.matmul(A.T, A), compute_uv=False)[0]
        tau = 1.0 / lipschitz
        sigma = 1.0

        x = np.zeros([n])
        u = np.zeros([n])
        alg_method = alg.PrimalDualHybridGradient(
            lambda alpha, u: (u - alpha * y) / (1 + alpha),
            lambda alpha, x: x / (1 + lamda * alpha),
            lambda x: np.matmul(A, x),
            lambda x: np.matmul(A.T, x),
            x, u, tau, sigma, max_iter=1000)
        while(not alg_method.done()):
            alg_method.update()

        npt.assert_allclose(x, x_truth)
