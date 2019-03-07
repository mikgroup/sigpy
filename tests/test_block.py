import unittest
import numpy as np
from sigpy import block, config

if config.cupy_enabled:
    import cupy as cp

if __name__ == '__main__':
    unittest.main()


class TestInterp(unittest.TestCase):

    def test_array_to_blocks(self):
        xps = [np]
        if config.cupy_enabled:
            xps.append(cp)

        for xp in xps:
            for dtype in [np.float32, np.complex64]:
                for ndim in [1, 2, 3, 4]:
                    with self.subTest(xp=xp, dtype=dtype, ndim=ndim):
                        input = xp.array(
                            [0, 1, 2, 3, 4, 5], dtype=dtype).reshape(
                            [6] + [1] * (ndim - 1))

                        blk_shape = [1] + [1] * (ndim - 1)
                        blk_strides = [1] + [1] * (ndim - 1)
                        output = xp.array(
                            [[0], [1], [2], [3], [4], [5]],
                            dtype=dtype).reshape(
                            [6] + [1] * (ndim - 1) + [1] + [1] * (ndim - 1))
                        xp.testing.assert_allclose(
                            output, block.array_to_blocks(
                                input, blk_shape, blk_strides))

                        blk_shape = [2] + [1] * (ndim - 1)
                        blk_strides = [1] + [1] * (ndim - 1)
                        output = xp.array(
                            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
                            dtype=dtype).reshape(
                            [5] + [1] * (ndim - 1) + [2] + [1] * (ndim - 1))
                        xp.testing.assert_allclose(
                            output, block.array_to_blocks(
                                input, blk_shape, blk_strides))

                        blk_shape = [2] + [1] * (ndim - 1)
                        blk_strides = [2] + [1] * (ndim - 1)
                        output = xp.array(
                            [[0, 1], [2, 3], [4, 5]], dtype=dtype).reshape(
                            [3] + [1] * (ndim - 1) + [2] + [1] * (ndim - 1))
                        xp.testing.assert_allclose(
                            output, block.array_to_blocks(
                                input, blk_shape, blk_strides))

                        blk_shape = [3] + [1] * (ndim - 1)
                        blk_strides = [2] + [1] * (ndim - 1)
                        output = xp.array(
                            [[0, 1, 2], [2, 3, 4]], dtype=dtype).reshape(
                            [2] + [1] * (ndim - 1) + [3] + [1] * (ndim - 1))
                        xp.testing.assert_allclose(
                            output, block.array_to_blocks(
                                input, blk_shape, blk_strides))

    def test_blocks_to_array(self):
        xps = [np]
        if config.cupy_enabled:
            xps.append(cp)

        for xp in xps:
            for dtype in [np.float32, np.complex64]:
                for ndim in [1, 2, 3, 4]:
                    with self.subTest(xp=xp, dtype=dtype, ndim=ndim):
                        shape = [6] + [1] * (ndim - 1)

                        blk_shape = [1] + [1] * (ndim - 1)
                        blk_strides = [1] + [1] * (ndim - 1)
                        input = xp.array(
                            [[0], [1], [2], [3], [4], [5]],
                            dtype=dtype).reshape(
                            [6] + [1] * (ndim - 1) + [1] + [1] * (ndim - 1))
                        output = xp.array([0, 1, 2, 3, 4, 5],
                                          dtype=dtype).reshape(
                            [6] + [1] * (ndim - 1))
                        xp.testing.assert_allclose(output,
                                                   block.blocks_to_array(
                                                       input, shape,
                                                       blk_shape, blk_strides))

                        blk_shape = [2] + [1] * (ndim - 1)
                        blk_strides = [1] + [1] * (ndim - 1)
                        input = xp.array(
                            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
                            dtype=dtype).reshape(
                            [5] + [1] * (ndim - 1) + [2] + [1] * (ndim - 1))
                        output = xp.array([0, 2, 4, 6, 8, 5],
                                          dtype=dtype).reshape(
                            [6] + [1] * (ndim - 1))
                        xp.testing.assert_allclose(output,
                                                   block.blocks_to_array(
                                                       input, shape,
                                                       blk_shape, blk_strides))

                        blk_shape = [2] + [1] * (ndim - 1)
                        blk_strides = [2] + [1] * (ndim - 1)
                        input = xp.array(
                            [[0, 1], [2, 3], [4, 5]], dtype=dtype).reshape(
                            [3] + [1] * (ndim - 1) + [2] + [1] * (ndim - 1))
                        output = xp.array([0, 1, 2, 3, 4, 5],
                                          dtype=dtype).reshape(
                            [6] + [1] * (ndim - 1))
                        xp.testing.assert_allclose(output,
                                                   block.blocks_to_array(
                                                       input, shape,
                                                       blk_shape, blk_strides))

                        blk_shape = [3] + [1] * (ndim - 1)
                        blk_strides = [2] + [1] * (ndim - 1)
                        input = xp.array(
                            [[0, 1, 2], [2, 3, 4]], dtype=dtype).reshape(
                            [2] + [1] * (ndim - 1) + [3] + [1] * (ndim - 1))
                        output = xp.array([0, 1, 4, 3, 4, 0],
                                          dtype=dtype).reshape(
                            [6] + [1] * (ndim - 1))
                        xp.testing.assert_allclose(output,
                                                   block.blocks_to_array(
                                                       input, shape,
                                                       blk_shape, blk_strides))
