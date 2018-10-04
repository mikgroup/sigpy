import unittest
import numpy as np
from sigpy import util, block, config

if config.cupy_enabled:
    import cupy as cp

if __name__ == '__main__':
    unittest.main()


class TestInterp(unittest.TestCase):

    def test_array_to_blocks1(self):
        xps = [np]
        if config.cupy_enabled:
            xps.append(cp)
            
        for xp in xps:
            input = xp.array([0, 1, 2, 3, 4, 5], dtype=np.float)

            blk_shape = [1]
            blk_strides = [1]
            output = xp.array([[0], [1], [2], [3], [4], [5]], dtype=xp.float)
            xp.testing.assert_allclose(output, block.array_to_blocks(input, blk_shape, blk_strides))

            blk_shape = [2]
            blk_strides = [1]
            output = xp.array([[0, 1],
                               [1, 2],
                               [2, 3],
                               [3, 4],
                               [4, 5]], dtype=xp.float)
            xp.testing.assert_allclose(output, block.array_to_blocks(input, blk_shape, blk_strides))

            blk_shape = [2]
            blk_strides = [2]
            output = xp.array([[0, 1],
                               [2, 3],
                               [4, 5]], dtype=xp.float)
            xp.testing.assert_allclose(output, block.array_to_blocks(input, blk_shape, blk_strides))

            blk_shape = [3]
            blk_strides = [2]
            output = xp.array([[0, 1, 2],
                               [2, 3, 4]], dtype=xp.float)
            xp.testing.assert_allclose(output, block.array_to_blocks(input, blk_shape, blk_strides))

    def test_blocks_to_array1(self):
        shape = [6]
        
        xps = [np]
        if config.cupy_enabled:
            xps.append(cp)
            
        for xp in xps:
            blk_shape = [1]
            blk_strides = [1]
            input = xp.array([[0], [1], [2], [3], [4], [5]], dtype=xp.float)
            output = xp.array([0, 1, 2, 3, 4, 5], dtype=xp.float)
            xp.testing.assert_allclose(output, block.blocks_to_array(input, shape, blk_shape, blk_strides))

            blk_shape = [2]
            blk_strides = [1]
            input = xp.array([[0, 1],
                              [1, 2],
                              [2, 3],
                              [3, 4],
                              [4, 5]], dtype=xp.float)
            output = xp.array([0, 2, 4, 6, 8, 5], dtype=xp.float)
            xp.testing.assert_allclose(output, block.blocks_to_array(input, shape, blk_shape, blk_strides))

            blk_shape = [2]
            blk_strides = [2]
            input = xp.array([[0, 1],
                              [2, 3],
                              [4, 5]], dtype=xp.float)
            output = xp.array([0, 1, 2, 3, 4, 5], dtype=xp.float)
            xp.testing.assert_allclose(output, block.blocks_to_array(input, shape, blk_shape, blk_strides))

            blk_shape = [3]
            blk_strides = [2]
            input = xp.array([[0, 1, 2],
                              [2, 3, 4]], dtype=xp.float)
            output = xp.array([0, 1, 4, 3, 4, 0], dtype=xp.float)
            xp.testing.assert_allclose(output, block.blocks_to_array(input, shape, blk_shape, blk_strides))
