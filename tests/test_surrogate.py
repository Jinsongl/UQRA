# -*- coding: utf-8 -*-

import uqra, unittest,warnings,os, sys 
from tqdm import tqdm
import numpy as np, scipy as sp 
from uqra.solver.PowerSpectrum import PowerSpectrum
from uqra.environment import Kvitebjorn as Kvitebjorn
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import pickle
import scipy.stats as stats

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()

class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_pce(self):
        print('========================TESTING: PCE.fit_quadrature(), ~Normal d=1=======================')
        d, p = 1, 10
        solver1 = uqra.Hermite(d,p) 
        doe_x, doe_w = uqra.QuadratureDesign(solver1).samples(11)
        # x = np.linspace(-3,3,100)
        for i in range(solver1.num_basis):
            coef = np.zeros((solver1.num_basis,)) 
            coef[i] = 1
            # coef = np.random.normal(size=(solver1.num_basis))
            solver1.set_coef(coef)
            y = solver1(doe_x)
            # print(y)
            pce_model  = uqra.PCE([stats.norm,]*d, p)
            pce_model.fit_quadrature(doe_x, doe_w, y)
            # print('>> Coefficients: ')
            # print('     True model: {}'.format(solver1))
            # print('     Surrogate: {}'.format(np.around(pce_model.coef,6)))
            # print('     max abs error: {}'.format(max(abs(coef - pce_model.coef ))))

            print('>> Prediction: ')
            x = np.random.normal(size=(d,1000))
            y0=solver1(x)
            y1=pce_model.predict(x)
            print('     Max abs error: {}'.format(max(abs(y0-y1))))

        print('========================TESTING: PCE.fit_quadrature(), ~Normal d=2=======================')
        d, p = 2, 5
        solver1 = uqra.Hermite(d,p) 
        doe_x, doe_w = uqra.QuadratureDesign(solver1).samples(6)
        # x = np.linspace(-3,3,100)
        for i in range(solver1.num_basis):
            coef = np.zeros((solver1.num_basis,)) 
            coef[i] = 1
            # coef = np.random.normal(size=(solver1.num_basis))
            solver1.set_coef(coef)
            y = solver1(doe_x)
            # print(y)
            pce_model  = uqra.PCE([stats.norm,]*d, p)
            pce_model.fit_quadrature(doe_x, doe_w, y)
            # print('>> Coefficients: ')
            # print('     True model: {}'.format(solver1))
            # print('     Surrogate: {}'.format(np.around(pce_model.coef,6)))
            # print('     max abs error: {}'.format(max(abs(coef - pce_model.coef ))))

            print('>> Prediction: ')
            x = np.random.normal(size=(d,1000))
            y0=solver1(x)
            y1=pce_model.predict(x)
            print('     Max abs error: {}'.format(max(abs(y0-y1))))

        print('===================TESTING: PCE.fit_quadrature(), ~Normal d=2, random coef==================')
        d, p = 2, 5
        solver1 = uqra.Hermite(d,p) 
        doe_x, doe_w = uqra.QuadratureDesign(solver1).samples(6)
        # x = np.linspace(-3,3,100)
        for i in range(solver1.num_basis):
            coef = np.random.normal(size=(solver1.num_basis,)) 
            # coef = np.random.normal(size=(solver1.num_basis))
            solver1.set_coef(coef)
            y = solver1(doe_x)
            # print(y)
            pce_model  = uqra.PCE([stats.norm,]*d, p)
            pce_model.fit_quadrature(doe_x, doe_w, y)
            print('>> Coefficients: ')
            print('     True model: {}'.format(solver1))
            print('     Surrogate: {}'.format(np.around(pce_model.coef,6)))
            print('     max abs error: {}'.format(max(abs(coef - pce_model.coef ))))

            print('>> Prediction: ')
            x = np.random.normal(size=(d,1000))
            y0=solver1(x)
            y1=pce_model.predict(x)
            print('     Max abs error: {}'.format(max(abs(y0-y1))))

        print('========================TESTING: PCE.fit_quadrature(), ~Uniform(-1,1) d=1=======================')
        d, p = 1, 10
        solver1 = uqra.Legendre(d,p) 
        doe_x, doe_w = uqra.QuadratureDesign(solver1).samples(11)
        # x = np.linspace(-3,3,100)
        for i in range(solver1.num_basis):
            coef = np.zeros((solver1.num_basis,)) 
            coef[i] = 1
            # coef = np.random.normal(size=(solver1.num_basis))
            solver1.set_coef(coef)
            y = solver1(doe_x)
            # print(y)
            pce_model  = uqra.PCE([stats.uniform(-1,1),]*d, p)
            pce_model.fit_quadrature(doe_x, doe_w, y)
            print('>> Coefficients: ')
            print('     True model: {}'.format(solver1))
            print('     Surrogate: {}'.format(np.around(pce_model.coef,6)))
            print('     max abs error: {}'.format(max(abs(coef - pce_model.coef ))))

            print('>> Prediction: ')
            x = np.random.normal(size=(d,1000))
            y0=solver1(x)
            y1=pce_model.predict(x)
            print('     Max abs error: {}'.format(max(abs(y0-y1))))

        print('========================TESTING: PCE.fit_quadrature(), ~Uniform(-1,1) d=2=======================')
        d, p = 2, 5
        solver1 = uqra.Legendre(d,p) 
        doe_x, doe_w = uqra.QuadratureDesign(solver1).samples(6)
        # x = np.linspace(-3,3,100)
        for i in range(solver1.num_basis):
            coef = np.zeros((solver1.num_basis,)) 
            coef[i] = 1
            # coef = np.random.normal(size=(solver1.num_basis))
            solver1.set_coef(coef)
            y = solver1(doe_x)
            # print(y)
            pce_model  = uqra.PCE([stats.uniform(-1,2),]*d, p)
            pce_model.fit_quadrature(doe_x, doe_w, y)
            print('>> Coefficients: ')
            print('     True model: {}'.format(solver1))
            print('     Surrogate: {}'.format(np.around(pce_model.coef,6)))
            print('     max abs error: {}'.format(max(abs(coef - pce_model.coef ))))

            print('>> Prediction: ')
            x = np.random.normal(size=(d,1000))
            y0=solver1(x)
            y1=pce_model.predict(x)
            print('     Max abs error: {}'.format(max(abs(y0-y1))))

    def test_mPCE(self):
        foo = lambda x: x**3 + 0.5*x + np.random.randn(*x.shape)
        dist = cp.Normal()
        x = dist.sample(1000).reshape(1,-1)
        print(x.shape)
        y =  np.squeeze(np.array([foo(x), foo(x)]).T)
        print(y.shape)
        # basis = cp.orth_ttr(5, dist)
        foo_hat = uqra.PCE(5, dist) 
        foo_hat.fit(x, y, method='OLS')
        y_pred = foo_hat.predict(x)
        print(y_pred.shape)
        foo_hat = uqra.mPCE(5, dist)
        foo_hat.fit(x, y, method='OLS')
        y_pred = foo_hat.predict(x)
        print(y_pred.shape)

    def test_gpce(self):
        print('==================TESTING: Generalized PCE (Not using SurrogateModel) ===================')
        gpce_dist_to_test   = [cp.Normal(), cp.Normal(2,3), cp.Gamma(1,1), cp.Beta(1,1)]
        gpce_opt_dist       = [cp.Normal(), cp.Normal(), cp.Gamma(1,1), cp.Beta(1,1)]
        gpce_opt_rule       = ['hem', 'hem', 'lag', 'jacobi']
        npoly_orders        = range(2,5)
        dist_zeta0          = cp.Normal()
        for i, igpce_dist in enumerate(gpce_dist_to_test):
            dist_zeta1 = gpce_opt_dist[i]
            print('>>> Testing # {:d}: gpce: {}, zeta0: {} , zeta1: {}'.format(i, igpce_dist, dist_zeta0, dist_zeta1 ))
            for ipoly_order in npoly_orders:
                print('  Polynomial order: {:d}'.format(ipoly_order))
                ## gPCE with hermite chaos
                uqra.blockPrint()
                quad_doe = uqra.DoE('QUAD', 'hem', [ipoly_order+1], dist_zeta0)
                samples_zeta= quad_doe.get_samples()
                zeta_cor, zeta_weight = samples_zeta[0]
                zeta_cor = zeta_cor.reshape((len(dist_zeta0),-1))
                x_cor = igpce_dist.inv(dist_zeta0.cdf(zeta_cor))
                zeta_poly, zeta_norms = cp.orth_ttr(ipoly_order, dist_zeta0, retall=True)
                x_hat,coeffs = cp.fit_quadrature(zeta_poly, zeta_cor, zeta_weight,np.squeeze(x_cor),retall=True)
                uqra.enablePrint()

                print('\t Hermite: {}'.format( np.around(coeffs,4)))

                ## gPCE with optimal chaos
                uqra.blockPrint()
                quad_doe = uqra.DoE('QUAD', gpce_opt_rule[i], [ipoly_order+1], dist_zeta1)
                samples_zeta= quad_doe.get_samples()
                zeta_cor, zeta_weight = samples_zeta[0]
                zeta_cor = zeta_cor.reshape((len(dist_zeta1),-1))
                x_cor = igpce_dist.inv(dist_zeta1.cdf(zeta_cor))
                zeta_poly, zeta_norms = cp.orth_ttr(ipoly_order, dist_zeta1, retall=True)
                x_hat,coeffs = cp.fit_quadrature(zeta_poly, zeta_cor, zeta_weight, np.squeeze(x_cor), retall=True)
                uqra.enablePrint()
                print('\t Optimal: {}'.format( np.around(coeffs,4)))

    def test_LassoLars(self):
        from sklearn import linear_model
        from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC, LassoLars
        from sklearn import datasets

        solver3 = lambda x: x**4 + x**2 + 3 + 0.3
        solver4 = lambda x: x**9 + 1.5*x**5 + 10 + 0.5

        np.random.seed(100)
        u_samples = stats.norm.rvs(size=1000)
        y_samples = np.array([solver3(u_samples), solver4(u_samples)])
        u_samples = u_samples.reshape(1,-1)
        print('y mean: {}'.format(np.mean(y_samples)))

        orth_poly= uqra.Hermite(1, 10)
        pce_model = uqra.PCE(orth_poly)
        print(u_samples.shape)
        print(y_samples.shape)
        for iy_samples in y_samples:
            pce_model.fit('OLS', u_samples, iy_samples)
            # print(pce_model.active_)
            # print(pce_model.metamodels)
            y_pred = pce_model.predict(u_samples)
            print(pce_model.score)
            print(pce_model.cv_error)
            print(np.amax(abs(y_pred - iy_samples)))

        # pce_model.fit('OlsLars', u_samples, y_samples)
        # print(pce_model.active_)
        # print(pce_model.metamodels)
        # y_pred = pce_model.predict(u_samples)
        # print(y_pred[:4])

    def test_OLS(self):
        np.random.seed(10)
        ndim, deg, nsamples = 1, 4, 10
        solver = uqra.Hermite(d=ndim, deg=deg)
        coef = stats.norm.rvs(0,1,size=solver.num_basis)
        coef[0] = 1
        coef[1] = 0.5
        coef[-1]= 1.5
        solver.set_coef(coef)
        print('solver coef: {}'.format(solver.coef))
        xx = np.linspace(-4,4,1000)
        yy = solver(xx) #+ 0.05*stats.norm.rvs(size=(1,x.size))

        orth_poly = uqra.Hermite(d=ndim, deg=deg)
        pce_model = uqra.PCE(orth_poly)
        x_train = stats.norm.rvs(0,1,size=(ndim,nsamples))
        y_train = solver(x_train)#+ 0.5*stats.norm.rvs(size=x_train.size)
        pce_model.fit('OLS',x_train.reshape(ndim,-1),y_train)
        yy_test = pce_model.predict(xx.reshape(1,-1)) #+ pce_model.model.intercept_
        error = yy - yy_test
        print('pce coef: {}'.format(pce_model.coef))
        print(pce_model.model.intercept_)
        print('max error: {:.2}'.format(max(error)))
 
if __name__ == '__main__':
    unittest.main()
