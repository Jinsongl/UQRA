# -*- coding: utf-8 -*-

import museuq, unittest, warnings, os, sys 
from tqdm import tqdm
import numpy as np, chaospy as cp, scipy as sp 

sys.stdout  = museuq.utilities.classes.Logger()
class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""
    def test_PolyBase(self):
        poly = museuq.PolyBase(1,10)
        poly_basis = poly.get_basis()
        print(poly_basis)


    def test_Hermite(self):
        x, w = museuq.Hermite(2,4).gauss_quadrature(5)
        print(x.shape)
        print(w.shape)
        x = sp.random.normal(size=(2,int(1e6)))
        X = museuq.Hermite(2,4).vandermonde(x)
        print(np.diag(X.T.dot(X)/1000000))






if __name__ == '__main__':
    unittest.main()
