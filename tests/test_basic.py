# -*- coding: utf-8 -*-

import context, museuq, unittest,warnings
import numpy as np, chaospy as cp, os, sys
from museuq.utilities import helpers as uqhelpers 
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()

class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_gauss_quadrature(self):
        """
        https://keisan.casio.com/exec/system/1329114617
        """
        quadrature_rule_to_test = ['LEG', 'HEM', 'LAG', 'JACOBI']
    #Legendre-Gauss quadrature 
        # n = 2
            # i	weight - wi	abscissa - xi
            # ___________________________________________________
            # 1	1.0000000000000000	-0.5773502691896257
            # 2	1.0000000000000000	0.5773502691896257
        # n = 3
            # ___________________________________________________
            # i	weight - wi	abscissa - xi
            # 1	0.8888888888888888	0.0000000000000000
            # 2	0.5555555555555556	-0.7745966692414834
            # 3	0.5555555555555556	0.7745966692414834
        # n = 4
            # ___________________________________________________
            # i	weight - wi	abscissa - xi
            # 1	0.6521451548625461	-0.3399810435848563
            # 2	0.6521451548625461	0.3399810435848563
            # 3	0.3478548451374538	-0.8611363115940526
            # 4	0.3478548451374538	0.8611363115940526
        # n = 5
            # ___________________________________________________
            # i	weight - wi	abscissa - xi
            # 1	0.5688888888888889	0.0000000000000000
            # 2	0.4786286704993665	-0.5384693101056831
            # 3	0.4786286704993665	0.5384693101056831
            # 4	0.2369268850561891	-0.9061798459386640
            # 5	0.2369268850561891	0.9061798459386640


        dist_zeta = cp.Uniform(-1,1)
        dist_zeta = cp.Gamma(4,1)
        # dist_zeta = cp.Iid(cp.Uniform(0,1),2) 

        doe_method, doe_rule, doe_orders = 'QUAD', 'hem', [2,3,4,5,6]
        quad_doe = museuq.DoE(doe_method, doe_rule, doe_orders, dist_zeta)
        samples_zeta= quad_doe.get_samples()
        quad_doe.disp()
        print('Compared results here: https://keisan.casio.com/exec/system/1329114617')

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
