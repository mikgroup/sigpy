import unittest
from sigpy import version

if __name__ == '__main__':
    unittest.main()


class TestVersion(unittest.TestCase):

    def test_version(self):
        assert version.__version__ == '0.1.23'
