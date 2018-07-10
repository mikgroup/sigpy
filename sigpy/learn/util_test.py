import unittest
import numpy as np
import numpy.testing as npt
from sigpy.learn import util

if __name__ == '__main__':
    unittest.main()


class TestUtil(unittest.TestCase):

    def test_labels_to_scores(self):

        labels = np.array([0, 1, 2])

        scores = util.labels_to_scores(labels)

        npt.assert_allclose(scores, [[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1]])

    def test_scores_to_labels(self):

        scores = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
        
        labels = util.scores_to_labels(scores)

        npt.assert_allclose(labels, [0, 1, 2])
