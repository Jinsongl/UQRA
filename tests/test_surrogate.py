# -*- coding: utf-8 -*-

import museuq, unittest,warnings,os, sys 
from tqdm import tqdm
import numpy as np, chaospy as cp, scipy as sp 
from museuq.solver.PowerSpectrum import PowerSpectrum
from museuq.environment import Kvitebjorn as Kvitebjorn
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import pickle
import scipy.stats as stats

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()

class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_pce(self):
        print('========================TESTING: PCE.fit_quadrature(), ~Normal d=1=======================')
        d, p = 1, 10
        solver1 = museuq.Hermite(d,p) 
        doe_x, doe_w = museuq.QuadratureDesign(solver1).samples(11)
        # x = np.linspace(-3,3,100)
        for i in range(solver1.num_basis):
            coef = np.zeros((solver1.num_basis,)) 
            coef[i] = 1
            # coef = np.random.normal(size=(solver1.num_basis))
            solver1.set_coef(coef)
            y = solver1(doe_x)
            # print(y)
            pce_model  = museuq.PCE(stats.norm, p)
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

    def test_mPCE(self):
        foo = lambda x: x**3 + 0.5*x + np.random.randn(*x.shape)
        dist = cp.Normal()
        x = dist.sample(1000).reshape(1,-1)
        print(x.shape)
        y =  np.squeeze(np.array([foo(x), foo(x)]).T)
        print(y.shape)
        # basis = cp.orth_ttr(5, dist)
        foo_hat = museuq.PCE(5, dist) 
        foo_hat.fit(x, y, method='OLS')
        y_pred = foo_hat.predict(x)
        print(y_pred.shape)
        foo_hat = museuq.mPCE(5, dist)
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
                museuq.blockPrint()
                quad_doe = museuq.DoE('QUAD', 'hem', [ipoly_order+1], dist_zeta0)
                samples_zeta= quad_doe.get_samples()
                zeta_cor, zeta_weight = samples_zeta[0]
                zeta_cor = zeta_cor.reshape((len(dist_zeta0),-1))
                x_cor = igpce_dist.inv(dist_zeta0.cdf(zeta_cor))
                zeta_poly, zeta_norms = cp.orth_ttr(ipoly_order, dist_zeta0, retall=True)
                x_hat,coeffs = cp.fit_quadrature(zeta_poly, zeta_cor, zeta_weight,np.squeeze(x_cor),retall=True)
                museuq.enablePrint()

                print('\t Hermite: {}'.format( np.around(coeffs,4)))

                ## gPCE with optimal chaos
                museuq.blockPrint()
                quad_doe = museuq.DoE('QUAD', gpce_opt_rule[i], [ipoly_order+1], dist_zeta1)
                samples_zeta= quad_doe.get_samples()
                zeta_cor, zeta_weight = samples_zeta[0]
                zeta_cor = zeta_cor.reshape((len(dist_zeta1),-1))
                x_cor = igpce_dist.inv(dist_zeta1.cdf(zeta_cor))
                zeta_poly, zeta_norms = cp.orth_ttr(ipoly_order, dist_zeta1, retall=True)
                x_hat,coeffs = cp.fit_quadrature(zeta_poly, zeta_cor, zeta_weight, np.squeeze(x_cor), retall=True)
                museuq.enablePrint()
                print('\t Optimal: {}'.format( np.around(coeffs,4)))

    def test_surrogate_model(self):
        pass
        # solver2 = lambda x: x**2 + 1
        # solver3 = lambda x: x**3 + x**2 + x + 3
        # solver4 = lambda x: cp.Gamma(1,1).inv(cp.Normal(0,1).cdf(x))
        # solver5 = lambda x: cp.Gamma(1,1).inv(cp.Gamma(1,1).cdf(x))

        # upper_tail_probs= [0.999,0.9999,0.99999]
        # moment2cal      = [1,2,3,4]
        # metrics2cal     = [ 'explained_variance_score', 'mean_absolute_error', 'mean_squared_error',
                    # 'median_absolute_error', 'r2_score', 'r2_score_adj', 'moment', 'mquantiles']
        
        # sample_weight = None
        # multioutput   = 'uniform_average'
        # # squared       = True

        # solvers2test= [solver1,solver2,solver3, solver4, solver5]
        # solver_strs = ['x', '1 + x**2', '3 + x + x**2 + x**3', 'Gamma(1,1), Hermite', 'Gamma(1,1), Optimal']
        # poly_orders = range(2,5)
        # dist_zeta   = cp.Normal()
        # dist_x      = cp.Normal()

        # orth_poly   = museuq.Hermite(d=1)
        # for isolver , isolver_str in zip(solvers2test, solver_strs):
            # for ipoly_order in poly_orders:
                # # museuq.blockPrint()
                # orth_poly.set_degree(ipoly_order)
                # doe_x, doe_w = museuq.QuadratureDesign(orth_poly).samples(ipoly_order+1)
                # train_y = np.squeeze(isolver(doe_x))
                # # train_y = np.array([train_y,train_y]).T
                # pce_model  = museuq.PCE(stats.norm, ipoly_order)
                # # print(len(pce_model.basis[0]))
                # pce_model.fit_quadrature(doe_x, doe_w, train_y)
                # print(isolver_str)
                # print(pce_model.metamodels)
                # pce_model.predict(doe.u, train_y, metrics=metrics2cal, prob=upper_tail_probs, moment=moment2cal, sample_weight=sample_weight, multioutput=multioutput)

    def test_LassoLars(self):
        from sklearn import linear_model
        from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC, LassoLars
        from sklearn import datasets

        solver3 = lambda x: x**4 + x**2 + 3 

        np.random.seed(100)
        dist_u = cp.Normal()
        u_samples = dist_u.sample(1000)
        y_samples = solver3(u_samples)
        print('y mean: {}'.format(np.mean(y_samples)))

        pce_model = museuq.PCE(10,dist_u)
        pce_model.fit(u_samples, y_samples, method='LassoLars')
        # print(pce_model.active_)
        # print(pce_model.metamodels)
        y_pred = pce_model.predict(u_samples.reshape(1,-1))
        print(y_pred[:4])

        pce_model.fit(u_samples, y_samples, method='OlsLars')
        # print(pce_model.active_)
        # print(pce_model.metamodels)
        y_pred = pce_model.predict(u_samples.reshape(1,-1))
        print(y_pred[:4])

if __name__ == '__main__':
    unittest.main()
