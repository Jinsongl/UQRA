# -*- coding: utf-8 -*-

from .context import museuq

import unittest


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_absolute_truth_and_meaning(self):
        assert True

    def test_acfPsd(self):
        ## refer to file test_acfPsd.py
        pass

    def test_gen_gauss_time_series(self):
        ## refer to file  test_gen_gauss_time_series
        pass

    def test_sdof_var(self):
        ## refer to file: test_sdof_var
        pass

    def test_poly5(self):
        ## refer to file: test_poly5
        pass

    def test_solver(self):
        ## refer to file: test_solver
        pass



if __name__ == '__main__':
    unittest.main()
