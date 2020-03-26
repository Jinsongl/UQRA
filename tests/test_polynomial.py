# -*- coding: utf-8 -*-

import museuq, unittest, warnings, os, sys, math
from tqdm import tqdm
import numpy as np, chaospy as cp, scipy as sp 
import scipy.stats as stats

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
        print('Probabilists Hermite: ')
        print(x)
        print(w)

        print('Physicist Hermite: ')
        x, w = museuq.Hermite(d=1, hem_type='physicists').gauss_quadrature(5)
        print(x)
        print(w)

        print('--------------------Testing Hermite.call(): Hermite(d=1, p=4)--------------------')
        d, p = 1, 4
        poly = museuq.Hermite(d,p)
        print(poly)
        x = np.arange(-4,4,0.5)
        for _ in range(5):
            coef = np.random.normal(size=poly.num_basis)
            print('     - coefficients: {}'.format(np.around(coef, 2)))
            y0=np.polynomial.hermite_e.hermeval(x,coef)
            poly.set_coef(coef*np.sqrt(poly.basis_norms))
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
            print('     > coefficients: {}'.format(np.around(coef, 2)))
            print('     - basis_degree: {}'.format(poly.basis_degree[i]))
            y0 = 1
            for i, ideg in enumerate(poly.basis_degree[i]):
                coef1 = np.zeros((p+1,))
                coef1[ideg] = 1 ##np.random.uniform(1)
                y0= y0 * np.polynomial.hermite_e.hermeval(x[i],coef1)
            poly.set_coef(coef*np.sqrt(poly.basis_norms))
            y1=poly(x)
            # print(y0)
            # print(y1)
            print('     - max abs error: {}'.format(max(abs(y0-y1))))

        print('--------------------Testing Vandermonde --------------------')
        ndim= 1
        p   = 3
        print('Physicist')
        orth_poly = museuq.Hermite(ndim, p, hem_type='physicists')
        x = np.array([-1,0,1])
        X = orth_poly.vandermonde(x, normed=False)
        print(X)

        print('Probabilists')
        orth_poly = museuq.Hermite(ndim, p, hem_type='probabilists')
        x = np.array([-1,0,1])
        X = orth_poly.vandermonde(x, normed=False)
        print(X)

        print('--------------------Testing Orthogonality --------------------')
        ndim= 2
        p   = 4
        n   = int(1e6)
        print('ndim = {:d}, poly degree = {:d}'.format(ndim, p))

        print('Probabilists')
        orth_poly = museuq.Hermite(ndim, p, hem_type='probabilists')
        # X = orth_poly.vandermonde(x, normed=False)
        x = sp.random.normal(0,1,size=(ndim, n))
        X = orth_poly.vandermonde(x)
        print(np.diag(X.T.dot(X)/n))
        # X1= np.squeeze(np.polynomial.hermite_e.hermevander2d(x[0],x[1],[p,]*ndim))
        # norms = [math.factorial(i) for i in range(p+1)]
        # print(np.diag(X1.T.dot(X1)/n)/norms)


        print('Physicist')
        orth_poly = museuq.Hermite(ndim, p, hem_type='physicists')
        # X = orth_poly.vandermonde(x, normed=False)
        x = sp.random.normal(0,np.sqrt(0.5),size=(ndim, n))
        X = orth_poly.vandermonde(x)
        print(np.diag(X.T.dot(X)/n))
        # X1= np.squeeze(np.polynomial.hermite.hermvander2d(x[0],x[1],[p,]*ndim))
        # norms = [math.factorial(i)*2**i for i in range(p+1)]
        # print(np.diag(X1.T.dot(X1)/n)/norms)



        print('--------------------Testing Orthogonality --------------------')
        ndim= 3
        p   = 4
        n   = int(1e6)
        print('ndim = {:d}, poly degree = {:d}'.format(ndim, p))
        x = sp.random.normal(0,1,size=(ndim, n))
        X = museuq.Hermite(ndim, p).vandermonde(x)
        print(np.diag(X.T.dot(X)/n))

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
        print('--------------------Testing empty instance--------------------')
        poly = museuq.Legendre()
        print(poly.basis)
        print(poly.basis_degree)
        print(poly.num_basis)

        print('--------------------Testing Gauss-Quadrature: Legendre(d=1), n=5--------------------')
        x, w = museuq.Legendre(d=1).gauss_quadrature(5)
        print(x)
        print(w)

        print('--------------------Testing Legendre.call(): Legendre(d=1, p=4)--------------------')
        d, p = 1, 4
        poly = museuq.Legendre(d,p)
        print(poly)
        x = np.arange(-4,4,0.5)
        for _ in range(5):
            coef = np.random.normal(size=poly.num_basis)
            print('     - coefficients: {}'.format(np.around(coef, 2)))
            y0=np.polynomial.legendre.legval(x,coef)
            poly.set_coef(coef*np.sqrt(poly.basis_norms))
            y1=poly(x)
            # y0=np.polynomial.legendre.legval(x,coef)
            # poly.set_coef(coef)
            # y1=poly(x)
            print('     - max abs error: {}'.format(np.around(max(abs(y0-y1)), 2)))

        
        print('--------------------Testing Legendre.call(): Legendre(d=2, p=4)--------------------')
        d, p = 2, 4
        poly = museuq.Legendre(d,p)
        print(poly)
        x = np.random.normal(size=(2,1000))
        for i in range(poly.num_basis):
            coef = np.zeros((poly.num_basis,))
            coef[i] = 1 
            print('     > coefficients: {}'.format(np.around(coef, 2)))
            print('     - basis_degree: {}'.format(poly.basis_degree[i]))
            y0 = 1
            for i, ideg in enumerate(poly.basis_degree[i]):
                coef1 = np.zeros((p+1,))
                coef1[ideg] = 1 ##np.random.uniform(1)
                y0= y0 * np.polynomial.legendre.legval(x[i],coef1)
            poly.set_coef(coef*np.sqrt(poly.basis_norms))
            y1=poly(x)
            # print(y0)
            # print(y1)
            print('     - max abs error: {}'.format(max(abs(y0-y1))))

        print('--------------------Testing Orthogonality --------------------')
        x = sp.random.uniform(-1,1,size=(2,int(1e6)))
        print(np.mean(x,axis=1))
        X = museuq.Legendre(2,4).vandermonde(x)
        print(np.diag(X.T.dot(X)/1000000))

    def test_cls_Legendre(self):
        print('--------------------Testing CLS Legendre--------------------')
        ndim= 2
        p   = 4
        print('ndim = {:d}, poly degree = {:d}'.format(ndim, p))
        x = np.load('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/Pluripotential/Uniform/DoE_McsE6R0.npy')
        x = x[:ndim,:]
        orth_poly = museuq.Legendre(ndim, p)
        P = orth_poly.num_basis
        X = orth_poly.vandermonde(x)
        Kp= np.sum(X * X, axis=1)
        w = np.sqrt(P/Kp)
        Psi = (X.T * w).T
        print(np.diag(Psi.T.dot(Psi)/x.shape[1]))


    def test_cls_Hermite(self):
        print('--------------------Testing CLS Hermite --------------------')
        ndim= 2
        # p   = 5
        for p in range(1,15):

            print('ndim = {:d}, poly degree = {:d}'.format(ndim, p))
            x = np.load('/Users/jinsongliu/BoxSync/MUSELab/museuq/tests/DoE_McsE6d{:d}R0.npy'.format(ndim))
            orth_poly = museuq.Hermite(ndim, p, hem_type='physicists')
            P = orth_poly.num_basis
            x = p**0.5*x[:ndim,:int(1.5 *np.log(P) * P)]
            X = orth_poly.vandermonde(x)
            Kp= np.sum(X * X, axis=1)
            w = np.sqrt(P/Kp)
            # print('weight shape" {}'.format(w.shape))
            Psi = (X.T * w).T
            # print(np.diag(Psi.T.dot(Psi)/x.shape[1]))
            _, s, _ = np.linalg.svd(Psi)
            ## condition number, kappa = max(svd)/min(svd)
            kappa = max(abs(s)) / min(abs(s)) 
            print('kappa = {:.2f}'.format(kappa))


if __name__ == '__main__':
    unittest.main()
