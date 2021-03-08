# -*- coding: utf-8 -*-

import uqra, unittest, warnings, os, sys, math
from tqdm import tqdm
import numpy as np, scipy as sp 
import scipy.stats as stats
import scipy

np.printoptions(precision=2)
sys.stdout  = uqra.utilities.classes.Logger()

def orthogonal_normalization(n, alpha, beta):
    """
    Calculate the orthogonality value for Jacobi polynoial, C
    int Pm(1-2x) * Pn(1-2x) * w(x) dx = C* delta_{ij} on [0,1]
    """
    c1 = 2**(alpha+beta+1)
    c2 = (2 * n + alpha + beta + 1)
    c3 = scipy.special.gamma(n+alpha+1)*scipy.special.gamma(n+beta+1)
    c4 = scipy.special.gamma(n+alpha+beta+1)*scipy.special.factorial(n)
    c  = c1 * c3 / c2 / c4
    return c

def weight_normalization(alpha, beta):
    """
    Normalization value to make the weight function being a Beta distribution as defined in scipy.stats.beta
    if x ~ Beta(alpha+1, beta+1)
    1-2x ~ w(x, alpha, beta) * C(alpha, beta)
    """
    B = lambda a, b : scipy.special.gamma(a) * scipy.special.gamma(b) / scipy.special.gamma(a+b)
    c = 1.0/ (2**(alpha+beta+1)*B(beta+1,alpha+1))
    return c

class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""
    def test_PolyBase(self):
        poly = uqra.PolyBase(1,10)
        poly_basis = poly.get_basis()
        print(poly_basis)

    def test_Hermite(self):
        print('--------------------Testing empty instance--------------------')
        poly = uqra.Hermite()
        print(poly.basis)
        print(poly.basis_degree)
        print(poly.num_basis)

        print('--------------------Testing Gauss-Quadrature: Hermite(d=1), n=5--------------------')
        x, w = uqra.Hermite(d=1).gauss_quadrature(5)
        print('Probabilists Hermite: ')
        print(x)
        print(w)

        print('Physicist Hermite: ')
        x, w = uqra.Hermite(d=1, hem_type='physicists').gauss_quadrature(5)
        print(x)
        print(w)

        print('--------------------Testing Vandermonde --------------------')
        ndim, p = 1, 3
        print('d={:d}, p={:d}'.format(ndim, p))
        print('Physicist, against np.polynomial.hermite.hermvander (not normalized)')
        orth_poly = uqra.Hermite(ndim, p, hem_type='physicists')
        x = np.array([-1,0,1])
        X = orth_poly.vandermonde(x, normalize=False)
        X1= np.polynomial.hermite.hermvander(x,p)
        if np.array_equal(X,X1):
            print('     True: Vander Matrix equal')

        print('Probabilists, against np.polynomial.hermite_e.hermvander (not normalized)')
        orth_poly = uqra.Hermite(ndim, p, hem_type='probabilists')
        x = np.array([-1,0,1])
        X = orth_poly.vandermonde(x, normalize=False)
        X1= np.polynomial.hermite_e.hermevander(x,p)
        if np.array_equal(X,X1):
            print('     True: Vander Matrix equal')

        print('--------------------Testing Hermite.call(): Probabilists Hermite(d, p)--------------------')
        x = np.arange(-5,5,0.5)
        ndim, p   = 1, 4  ## dimension
        coef      = np.ones(5)
        orth_poly = uqra.Hermite(ndim, p, coef=coef, hem_type='probabilists')
        print(orth_poly.coef)
        y = orth_poly(x)
        X = orth_poly.vandermonde(x)
        print(x)
        print(X)
        print(y)

        orth_poly = uqra.Hermite(ndim, p, hem_type='probabilists')
        coef      = np.ones(orth_poly.num_basis)
        print(orth_poly.coef)
        orth_poly.set_coef(coef)
        print(orth_poly.coef)
        y = orth_poly(x)
        X = orth_poly.vandermonde(x)
        print(x)
        print(X)
        print(y)

        orth_poly = uqra.Hermite(ndim, p, hem_type='probabilists')
        coef      = np.ones(orth_poly.num_basis) *2
        print(orth_poly.coef)
        orth_poly.set_coef(coef)
        print(orth_poly.coef)
        y = orth_poly(x)
        X = orth_poly.vandermonde(x)
        print(x)
        print(X)
        print(y)


        # print('--------------------Testing Hermite.call(): Probabilists Hermite(d, p)--------------------')
        # ndim, p   = 1, 4  ## dimension
        # poly_name = 'heme'
        # orth_poly = uqra.poly.orthogonal(ndim, p, poly_name)
        # # coef      = stats.norm.rvs(0,1, orth_poly.num_basis)
        # coef      = np.ones(orth_poly.num_basis)
        # solver    = uqra.SparsePoly(orth_poly, coef=coef)
        # x = np.arange(-5,5,0.5)
        # y = solver.run(x)
        # X = orth_poly.vandermonde(x)
        # print(x)
        # print(X)
        # print(y)
        # coef      = np.ones(orth_poly.num_basis) * 2
        # solver    = uqra.SparsePoly(orth_poly, coef=coef)
        # x = np.arange(-5,5,0.5)
        # y = solver.run(x)
        # X = orth_poly.vandermonde(x)
        # print(x)
        # print(X)
        # print(y)

        return

        print('--------------------Testing Hermite.call(): Probabilists Hermite(d, p)--------------------')
        x = np.arange(-5,5,0.5)
        d, p = 1, 4
        print('     > ndim={:d}, p = {:d}'.format(d, p))
        poly = uqra.Hermite(d,p)
        for _ in range(5):
            coef = np.random.normal(size=poly.num_basis)
            y0=np.polynomial.hermite_e.hermeval(x,coef)
            poly.set_coef(coef)
            y1=poly(x)
            # print(poly)
            if not np.array_equal(y0, y1):
                print('     - max abs error: {:.2e}'.format(np.around(max(abs(y0-y1)), 2)))
            else:
                print('     > Equal output')

        d, p = 1, 30
        print('     > ndim={:d}, p = {:d}'.format(d, p))
        poly = uqra.Hermite(d,p)
        for _ in range(5):
            coef = np.random.normal(size=poly.num_basis)
            y0=np.polynomial.hermite_e.hermeval(x,coef)
            poly.set_coef(coef)
            y1=poly(x)
            # print(poly)
            if not np.array_equal(y0, y1):
                print('     - max abs error: {:.2e}'.format(np.around(max(abs(y0-y1)), 2)))
            else:
                print('     > Equal output')
        
        p = 40
        x = np.arange(-10,10,0.5)
        y = np.arange(-10,10,0.5)
        y0=np.polynomial.hermite.hermvander2d(x,y, [p,p])
        y0= np.sum(y0, axis=1)
        print(y0.shape)
        print(np.max(y0))
        poly = uqra.Hermite(d=2,deg=p, hem_type='physicists')
        y1=poly(np.array([x,y]))
        print(y1.shape)
        print(np.max(y1))



        print('--------------------Testing Hermite.call(): Physicists Hermite(d, p)--------------------')
        d, p = 1, 4
        print('     > ndim={:d}, p = {:d}'.format(d, p))
        poly = uqra.Hermite(d,p)
        for _ in range(5):
            coef = np.random.normal(size=poly.num_basis)
            # print('     - coefficients: {}'.format(np.around(coef, 2)))
            y0=np.polynomial.hermite_e.hermeval(x,coef)
            poly.set_coef(coef)
            # print(poly)
            y1=poly(x)
            print('     - max abs error: {}'.format(np.around(max(abs(y0-y1)), 2)))

        d, p = 1, 30
        print('     > ndim={:d}, p = {:d}'.format(d, p))
        poly = uqra.Hermite(d,p)
        for _ in range(5):
            coef = np.random.normal(size=poly.num_basis)
            # print('     - coefficients: {}'.format(np.around(coef, 2)))
            y0=np.polynomial.hermite_e.hermeval(x,coef)
            poly.set_coef(coef)
            # print(poly)
            y1=poly(x)
            print('     - max abs error: {}'.format(np.around(max(abs(y0-y1)), 2)))





        print('--------------------Testing Hermite.call(): Hermite(d=2, p=4)--------------------')
        print(' >>> Testing each basis one by one:')
        d, p = 2, 4
        poly = uqra.Hermite(d,p)
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
            poly.set_coef(coef)
            y1=poly(x)
            # print(y0)
            # print(y1)
            print('     - max abs error: {}'.format(max(abs(y0-y1))))
            print('\n')

        # ndim= 2
        # p   = 4
        # print('d={:d}, p={:d}'.format(d, p))
        # print('Physicist')
        # orth_poly = uqra.Hermite(ndim, p, hem_type='physicists')
        # x = np.random.normal(size=(ndim,10)) 
        # X = orth_poly.vandermonde(x, normed=False)
        # X1= np.polynomial.hermite.hermvander2d(x,p)
        # if np.array_equal(X,X1):
            # print('     Pass: Poly.vandermonde == np.polynomial.hermite.hermvander')
        # else:
            # raise ValueError


        # print('Probabilists')
        # orth_poly = uqra.Hermite(ndim, p, hem_type='probabilists')
        # x = np.random.normal(size=(ndim,10)) 
        # X = orth_poly.vandermonde(x, normed=False)
        # X1= np.polynomial.hermite_e.hermevander2d(x,p)
        # if np.array_equal(X,X1):
            # print('     Pass: Poly.vandermonde == np.polynomial.hermite.hermvander')
        # else:
            # raise ValueError

        print('--------------------Testing large size vandermonde --------------------')
        x = sp.random.normal(size=(3,1000000))
        d, p = 3, 10
        print('     > ndim={:d}, p = {:d}'.format(d, p))
        poly = uqra.Hermite(d,p)
        y = poly(x)



        print('--------------------Testing Orthogonality --------------------')
        ndim= 2
        p   = 4
        n   = int(1e6)
        print('ndim = {:d}, poly degree = {:d}'.format(ndim, p))

        print('Probabilists')
        orth_poly = uqra.Hermite(ndim, p, hem_type='probabilists')
        # X = orth_poly.vandermonde(x, normed=False)
        x = sp.random.normal(0,1,size=(ndim, n))
        X = orth_poly.vandermonde(x)
        print(np.diag(X.T.dot(X)/n))
        # X1= np.squeeze(np.polynomial.hermite_e.hermevander2d(x[0],x[1],[p,]*ndim))
        # norms = [math.factorial(i) for i in range(p+1)]
        # print(np.diag(X1.T.dot(X1)/n)/norms)


        print('Physicist')
        orth_poly = uqra.Hermite(ndim, p, hem_type='physicists')
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
        X = uqra.Hermite(ndim, p).vandermonde(x)
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
        # X = uqra.Hermite(2,4).vandermonde(x)
        # print(np.diag(X.T.dot(X)/1000000))

    def test_Legendre(self):
        print('--------------------Testing empty instance--------------------')
        poly = uqra.Legendre()
        print(poly.basis)
        print(poly.basis_degree)
        print(poly.num_basis)

        print('--------------------Testing Gauss-Quadrature: Legendre(d=1), n=5--------------------')
        x, w = uqra.Legendre(d=1).gauss_quadrature(5)
        print(x)
        print(w)

        print('--------------------Testing Legendre.call(): Legendre(d=1, p=4)--------------------')
        d, p = 1, 4
        poly = uqra.Legendre(d,p)
        print(poly)
        x = np.arange(-1,1,0.5)
        for _ in range(5):
            coef = np.random.normal(size=poly.num_basis)
            print('     - coefficients: {}'.format(np.around(coef, 2)))
            y0=np.polynomial.legendre.legval(x,coef)
            print(max(y0))
            poly.set_coef(coef*np.sqrt(poly.basis_norms))
            y1=poly(x)
            print('     - max abs error: {}'.format(np.around(max(abs(y0-y1)), 2)))
        
        print('--------------------Testing Legendre.call(): Legendre(d=2, p=4)--------------------')
        d, p = 2, 4
        poly = uqra.Legendre(d,p)
        print(poly)
        x = np.random.uniform(-1,1,size=(d,1000))
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
            poly.set_coef(coef)
            y1=poly(x)
            print('     - max abs error: {}'.format(max(abs(y0-y1))))

        print('--------------------Testing Vandermonde --------------------')
        ndim= 1
        p   = 3
        n   = 100
        print('d={:d}, p={:d}'.format(ndim, p))
        x = np.linspace(-1,1,1000) 
        X = uqra.Legendre(ndim,p).vandermonde(x, normalize=False)
        X1= np.polynomial.legendre.legvander(x,p)
        print(X.shape)
        print(X1.shape)
        if np.array_equal(X,X1):
            print('     Pass: Poly.vandermonde == np.polynomial.legendre.legvander')
        else:
            print(      'Abs max error:'.format(np.amax(abs(X-X1))))


        print('--------------------Testing Orthogonality --------------------')
        ndim= 1
        n   = int(1e6)
        print('d={:d}, n={:d}'.format(ndim, n))
        x = sp.random.uniform(-1,1,size=(ndim,n))
        print(np.mean(x,axis=1))
        print(np.max( x,axis=1))
        print(np.min( x,axis=1))
        for p in range(1,10):
            X = uqra.Legendre(ndim,p).vandermonde(x)
            print(np.diag(X.T.dot(X)/n))

        ndim= 2
        n   = int(1e6)
        print('d={:d}, n={:d}'.format(ndim, n))
        x = sp.random.uniform(-1,1,size=(ndim,n))
        print(np.mean(x,axis=1))
        print(np.max( x,axis=1))
        print(np.min( x,axis=1))
        for p in range(1,10):
            X = uqra.Legendre(ndim,p).vandermonde(x)
            print(np.diag(X.T.dot(X)/n))

    def test_Legendre1(self):

        print('--------------------Testing instance--------------------')
        poly = uqra.Legendre1()
        print('{:20s}: {}'.format('poly.basis',poly.basis))
        print('{:20s}: {}'.format('poly.basis_degree',poly.basis_degree))
        print('{:20s}: {}'.format('poly.num_basis',poly.num_basis))

        poly = uqra.Legendre1(1, 3)
        x = np.arange(-1,1,0.5)
        print('{:20s}: {}'.format('poly.basis',poly.basis))
        print('{:20s}: {}'.format('poly.basis_degree',poly.basis_degree))
        print('{:20s}: {}'.format('poly.num_basis',poly.num_basis))
        print('{:20s}: {}'.format('weights:uqra', poly.weight(x)))
        print('{:20s}: {}'.format('weights:numpy', np.polynomial.legendre.legweight(x)))
        print('{:20s}: {}'.format('orth norm', poly.orthogonal_normalization(3)))

        print('--------------------Testing Gauss-Quadrature: Legendre1(d=1), n=5--------------------')
        x, w = uqra.Legendre1(d=1).gauss_quadrature(5)
        print(x)
        print(w)

        print('--------------------Testing Legendre1.call(): Legendre1(d=1, p=4)--------------------')
        d, p = 1, 4
        poly = uqra.Legendre1(d,p)
        x = np.arange(-1,1,0.5)
        print('uqra.Legendre1(1,4): {}'.format(poly))
        print('x: np.arange(-1,1,0.5)')
        for _ in range(5):
            coef = np.random.normal(size=poly.num_basis)
            y0   = np.polynomial.legendre.legval(x,coef)
            npleg= np.polynomial.legendre.Legendre(coef)
            uqleg= uqra.Legendre1(d=d, deg=p, coef=coef)
            poly.set_coef(coef)
            y1 = poly(x)
            print('     - coefficients: {}'.format(np.around(coef, 2)))
            print('     - Numpy coef  : {}'.format(npleg))
            print('     - UQRAleg coef: {}'.format(np.around(uqleg.coef, 2)))
            print('     - UQRA  coef  : {}'.format(np.around(poly.coef, 2)))
            print('     - Numpy y0    : {}'.format(np.around(y0, 2)))
            print('     - UQRA leg y0 : {}'.format(np.around(uqleg(x), 2)))
            print('     - UQRA  y0    : {}'.format(np.around(y1, 2)))
            print('     - max abs error: {}\n'.format(np.around(max(abs(y0-y1)), 2)))

        
        print('--------------------Testing Legendre1.call(): Legendre1(d=2, p=4)--------------------')
        d, p = 2, 4
        poly = uqra.Legendre1(d,p)
        print('uqra.Legendre1(2,4): {}'.format(poly))
        x = np.random.uniform(-1,1,size=(d,1000))
        for i in range(poly.num_basis):
            coef = np.zeros((poly.num_basis,))
            coef[i] = 1 
            print('     > coefficients: {}'.format(np.around(coef, 2)))
            print('     - basis_degree: {}'.format(poly.basis_degree[i]))
            y0 = 1
            for i, ideg in enumerate(poly.basis_degree[i]):
                coef1 = np.zeros((ideg+1,))
                coef1[ideg] = 1 ##np.random.uniform(1)
                print('     > ideg coef: {}'.format(np.around(coef1, 2)))
                y0= y0 * np.polynomial.legendre.legval(x[i],coef1)
            poly.set_coef(coef)
            y1=poly(x)
            print('     - max abs error: {}\n'.format(max(abs(y0-y1))))

        print('--------------------Testing Orthogonality --------------------')
        np.set_printoptions(precision=2)
        ndim= 1
        n   = int(1e5)
        print('d={:d}, n={:d}'.format(ndim, n))
        x = stats.uniform.rvs(-1,2,size=(ndim,n))
        print(np.mean(x,axis=1))
        print(np.max( x,axis=1))
        print(np.min( x,axis=1))
        for p in range(1,10):
            X = uqra.Legendre1(ndim,p).vandermonde(x)
            print(np.diag(X.T.dot(X)/n))

        ndim= 2
        n   = int(1e5)
        print('d={:d}, n={:d}'.format(ndim, n))
        x = sp.random.uniform(-1,1,size=(ndim,n))
        print(np.mean(x,axis=1))
        print(np.max( x,axis=1))
        print(np.min( x,axis=1))
        for p in range(1,10):
            X = uqra.Legendre1(ndim,p).vandermonde(x)
            print(np.diag(X.T.dot(X)/n))

    def test_Jacobi(self):
        print('--------------------Testing empty instance--------------------')
        poly = uqra.Jacobi()
        # print('--------------------Testing Gauss-Quadrature: Jacobi(d=1), n=5--------------------')
        # x, w = uqra.Jacobi(d=1).gauss_quadrature(5)
        # print(x)
        # print(w)

        print('--------------------Testing Jacobi basis by basis --------------------')
        d, p = 1, 4
        a, b = 1, 1
        poly = uqra.Jacobi(a,b, d=d, deg=p)
        x = np.arange(-1,1.1,0.1)
        # print('uqra.Jacobi(1,4,1,1): {}'.format(poly))
        # print('x: np.arange(0,1,0.5)')

        np.random.seed(100)
        for _ in range(1):
            print('random repeat>....')
            # coef = np.random.normal(size=poly.num_basis)
            for ideg in np.arange(p+1):
                coef = np.zeros(p+1)
                coef[ideg] = 1
                print('     - coefficients: {}'.format(np.around(coef, 2)))
                npleg= np.polynomial.legendre.Legendre(coef)
                y0   = npleg(x)
                spjac= scipy.special.jacobi(ideg,0,0)
                y1   = spjac(x)
                poly = uqra.Jacobi(a,b, d=d, deg=p, coef=coef)
                y2   = poly(x)
                print('     - max abs error y1-y0 : {}'.format(np.around(max(abs(y1-y0)), 4)))
                print('     - max abs error y2-y0 : {}'.format(np.around(max(abs(y2-y0)), 4)))
                
        print('--------------------Testing Legendre1.call(): Legendre1(d=1, p=4)--------------------')
        d, p = 1, 4
        a, b = 1, 1
        poly = uqra.Jacobi(a,b, d=d, deg=p)
        x = np.arange(-1,1.1,0.1)
        print('uqra.Jacobi(1,4,1,1): {}'.format(poly))
        print('x: np.arange(0,1,0.5)')

        for _ in range(5):
            coef = np.random.normal(size=poly.num_basis)
            npleg= np.polynomial.legendre.Legendre(coef)
            y0   = npleg(x)
            poly = uqra.Jacobi(a,b, d=d, deg=p, coef=coef)
            y1   = poly(x)
            print('     - coefficients: {}'.format(np.around(coef, 2)))
            print('     - Numpy coef  : {}'.format(npleg))
            print('     - UQRA  coef  : {}'.format(np.around(poly.coef, 2)))
            print('     - Numpy y0    : {}'.format(np.around(y0, 2)))
            print('     - UQRA  y0    : {}'.format(np.around(y1, 2)))
            print('     - max abs error: {}\n'.format(np.around(max(abs(y0-y1)), 2)))

        print('--------------------Testing Orthogonality --------------------')
        np.set_printoptions(precision=2)
        np.random.seed(100)
        n   = int(1e6)
        a, b = 1, 1
        x = stats.beta.rvs(a,b,size=(d,n),loc=-1, scale=2)
        print(np.mean(x), np.std(x))
        print(np.amin(x), np.amax(x))
        print(x)
        C = weight_normalization(0,0)

        for i in np.arange(5):
            spjac = scipy.special.jacobi(i,0,0)
            C1 = orthogonal_normalization(i,0,0)
            y = spjac(x)
            e = np.dot(y, y.T)[0][0]/n
            print(i, C, C1, e, e/C1/C)


        print('--------------------Testing Orthogonality --------------------')
        np.random.seed(100)
        np.set_printoptions(precision=2)
        a, b = 1, 1
        d, p = 1, 4
        n   = int(1e6)
        print('d={:d}, n={:d}'.format(d, n))
        x = stats.beta.rvs(a,b,size=(d,n),loc=-1, scale=2)
        poly = uqra.Jacobi(a,b,d=d,deg=p)
        X = poly.vandermonde(x)
        print(np.diag(X.T.dot(X)/n))

        d, p = 2, 4
        n   = int(1e6)
        print('d={:d}, n={:d}'.format(d, n))
        x = stats.beta.rvs(a,b,size=(d,n),loc=-1, scale=2)
        print(np.mean(x,axis=1))
        print(np.max( x,axis=1))
        print(np.min( x,axis=1))
        poly = uqra.Jacobi(a,b,d=d,deg=p)
        X = poly.vandermonde(x)
        print(np.diag(X.T.dot(X)/n))

    def test_cls_Jacobi(self):
        print('--------------------Testing CLS Legendre--------------------')
        ndim= 2
        p   = 4
        print('ndim = {:d}, poly degree = {:d}'.format(ndim, p))
        x = np.load('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/Pluripotential/Uniform/DoE_McsE6R0.npy')
        x = x[:ndim,:]
        orth_poly = uqra.Legendre(ndim, p)
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
            x = np.load('/Users/jinsongliu/BoxSync/MUSELab/uqra/tests/DoE_McsE6d{:d}R0.npy'.format(ndim))
            orth_poly = uqra.Hermite(ndim, p, hem_type='physicists')
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
