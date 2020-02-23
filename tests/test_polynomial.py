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
        print('--------------------Testing empty instance--------------------')
        poly = museuq.Hermite()
        print(poly.basis)
        print(poly.basis_degree)
        print(poly.num_basis)

        print('--------------------Testing Gauss-Quadrature: Hermite(d=1), n=5--------------------')
        x, w = museuq.Hermite(d=1).gauss_quadrature(5)
        print('Physicist Hermite: ')
        print(x/np.sqrt(2))
        print(w/np.sqrt(2))

        print('--------------------Testing Hermite.call(): Hermite(d=1, p=4)--------------------')
        d, p = 1, 4
        poly = museuq.Hermite(d,p)
        print(poly)
        x = np.arange(-4,4,0.5)
        for _ in range(5):
            coef = np.random.normal(size=poly.num_basis)
            print('     - coefficients: {}'.format(np.around(coef, 2)))
            y0=np.polynomial.hermite_e.hermeval(x,coef)
            poly.set_coef(coef)
            y1=poly(x)
            print('     - max abs error: {}'.format(np.around(max(abs(y0-y1)), 2)))

        
        print('--------------------Testing Hermite.call(): Hermite(d=2, p=4)--------------------')
        d, p = 2, 4
        poly = museuq.Hermite(d,p)
        print(poly)
        x = np.random.normal(size=(2,1000))
        for i in range(poly.num_basis):
            coef = np.zeros((poly.num_basis,))
            coef[i] = 1 
            print('     - coefficients: {}'.format(np.around(coef, 2)))
            print(poly.basis_degree[i])
            y0 = 1
            for i, ideg in enumerate(poly.basis_degree[i]):
                coef1 = np.zeros((p+1,))
                coef1[ideg] = 1 ##np.random.uniform(1)
                y0= y0 * np.polynomial.hermite_e.hermeval(x[i],coef1)
            poly.set_coef(coef)
            y1=poly(x)
            # print(y0)
            # print(y1)
            print('     - max abs error: {}'.format(max(abs(y0-y1))))

        


        # X = orth_poly.vandermonde(x, normed=False)
        # norms = np.sum(np.square(X).T * w, -1)
        # print(norms)
        # print(orth_poly.basis_norms*2*np.pi)
        # if not np.array_equal(norms,orth_poly.basis_norms *2*np.pi ):
            # print(max(abs(norms - orth_poly.basis_norms *2*np.pi)))

        # X = orth_poly.vandermonde(x)
        # norms = np.sum(np.square(X).T * w, -1)
        # print(norms)
        # print(orth_poly.basis_norms*2*np.pi)
        # if not np.array_equal(norms,2*np.pi ):
            # print(max(abs(norms - 2*np.pi)))


        # x = sp.random.normal(size=(2,int(1e6)))
        # X = museuq.Hermite(2,4).vandermonde(x)
        # print(np.diag(X.T.dot(X)/1000000))


    def test_Legendre(self):
        x, w = museuq.Legendre(2,4).gauss_quadrature(5)
        print(x.shape)
        print(w.shape)
        x = sp.random.uniform(-1,1,size=(2,int(1e6)))
        X = museuq.Legendre(2,4).vandermonde(x)
        print(np.diag(X.T.dot(X)/1000000))





if __name__ == '__main__':
    unittest.main()
