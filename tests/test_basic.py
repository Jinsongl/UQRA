# -*- coding: utf-8 -*-

import context, museuq, unittest,warnings
import numpy as np, chaospy as cp, os, sys
from museuq.utilities import helpers as uqhelpers 
from museuq.utilities import constants as const
from museuq.utilities.PowerSpectrum import PowerSpectrum

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()

data_dir = '/Users/jinsongliu/BoxSync/MUSELab/museuq/examples/JupyterNotebook'
class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_gauss_quadrature(self):
        """
        https://keisan.casio.com/exec/system/1329114617
        """
        print('========================TESTING: GAUSS QUADRATURE=======================')

        quadrature_rule_to_test = ['leg', 'hem', 'lag', 'jacobi']
        quadrature_dist_to_test = [cp.Uniform(-1,1), cp.Normal(), cp.Gamma(1,1), cp.Beta(1,1)]
        #cp.Gamma() def: x**(a-1)*numpy.e**(-x) / special.gamma(a)
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


        # dist_zeta = cp.Iid(cp.Uniform(0,1),2) 

        for dist, doe_rule in zip(quadrature_dist_to_test, quadrature_rule_to_test):
            print('Gauss Quadrature with polynominal: {}'.format(const.DOE_RULE_FULL_NAMES[doe_rule.lower()]))
            doe_method, doe_rule, doe_orders = 'QUAD', doe_rule, [2,3,4,5,6]
            quad_doe = museuq.DoE(doe_method, doe_rule, doe_orders, dist)
            samples_zeta= quad_doe.get_samples()
            quad_doe.disp()
        print('Compared results here: https://keisan.casio.com/exec/system/1329114617')

    def test_gpce(self):
        print('==================TESTING: Generalized PCE (Not using SurrogateModel) ===================')
        gpce_dist_to_test   = [cp.Normal(), cp.Normal(2,3), cp.Gamma(1,1), cp.Beta(1,1)]
        gpce_opt_dist       = [cp.Normal(), cp.Normal(), cp.Gamma(1,1), cp.Beta(1,1)]
        gpce_opt_rule       = ['hem', 'hem', 'lag', 'jacobi']
        npoly_orders        = range(2,5)
        dist_zeta0          = cp.Normal()
        for i, igpce_dist in enumerate(gpce_dist_to_test):
            dist_zeta1 = gpce_opt_dist[i]
            print('>> Testing # {:d}: gpce: {}, zeta0: {} , zeta1: {}'.format(i, igpce_dist, dist_zeta0, dist_zeta1 ))
            for ipoly_order in npoly_orders:
                print('  Polynomial order: {:d}'.format(ipoly_order))
                ## gPCE with hermite chaos
                uqhelpers.blockPrint()
                quad_doe = museuq.DoE('QUAD', 'hem', [ipoly_order+1], dist_zeta0)
                samples_zeta= quad_doe.get_samples()
                zeta_cor, zeta_weight = samples_zeta[0]
                x_cor = igpce_dist.inv(dist_zeta0.cdf(zeta_cor))
                zeta_poly, zeta_norms = cp.orth_ttr(ipoly_order, dist_zeta0, retall=True)
                x_hat,coeffs = cp.fit_quadrature(zeta_poly, zeta_cor, zeta_weight, np.squeeze(x_cor), retall=True)
                uqhelpers.enablePrint()
                print('\t Hermite: {}'.format( np.around(coeffs,4)))

                ## gPCE with optimal chaos
                uqhelpers.blockPrint()
                quad_doe = museuq.DoE('QUAD', gpce_opt_rule[i], [ipoly_order+1], dist_zeta1)
                samples_zeta= quad_doe.get_samples()
                zeta_cor, zeta_weight = samples_zeta[0]
                x_cor = igpce_dist.inv(dist_zeta1.cdf(zeta_cor))
                zeta_poly, zeta_norms = cp.orth_ttr(ipoly_order, dist_zeta1, retall=True)
                x_hat,coeffs = cp.fit_quadrature(zeta_poly, zeta_cor, zeta_weight, np.squeeze(x_cor), retall=True)
                uqhelpers.enablePrint()
                print('\t Optimal: {}'.format( np.around(coeffs,4)))

    def test_PowerSpectrum(self):
        print('========================TESTING: Power Spectrum =======================')
        powerspecturms2test = ['jonswap']
        powerspecturms_args = [(8, 10)]
        df  = 0.00001
        f   = np.arange(0, 10, df)

        for psd_name, psd_args in zip(powerspecturms2test, powerspecturms_args):
            psd = PowerSpectrum(psd_name, *psd_args)
            psd_f, psd_pxx = psd.get_pxx(f) 
            psd_area = np.sum(psd_pxx * df)
            np.save(os.path.join(data_dir,psd_name+'_psd_f'), psd_f)
            np.save(os.path.join(data_dir,psd_name+'_psd_pxx'), psd_pxx)
            tau, acf = psd.get_acf()
            np.save(os.path.join(data_dir,psd_name+'_tau'), tau)
            np.save(os.path.join(data_dir,psd_name+'_acf'), acf)
            t, eta = psd.gen_process()
            np.save(os.path.join(data_dir,psd_name+'_t'), t)
            np.save(os.path.join(data_dir,psd_name+'_eta'), eta)
            print(t, eta)
            # t, eta = psd._gen_process_sum()
            print('PSD name: {:s}, args: {}, Area: {:.2f}, 4*std:{}'.format(psd_name, psd_args, psd_area, 4*np.std(eta)))

    def test_linear_oscillator(self):
        print('========================TESTING: Lienar Oscillator =======================')
        x = (Hs,Tp) = (8, 14.7)
        tmax,dt =1000, 0.1
        t = np.arange(0,tmax, dt)
        y = museuq.solver.dynamic_models.linear_oscillator(t,x)

    def test_surrogate_model(self):
        print('========================TESTING: SurrogateModel.fit(), ~Normal =======================')
        solver1 = lambda x: x
        solver2 = lambda x: x**2 + 1
        solver3 = lambda x: x**3 + x**2 + x + 3
        solver4 = lambda x: cp.Gamma(1,1).inv(cp.Normal(0,1).cdf(x))
        solver5 = lambda x: cp.Gamma(1,1).inv(cp.Gamma(1,1).cdf(x))

        metrics  = ['max_error', 'mean_absolute_error', 'mean_squared_error','moments', 'upper_tails']
        
        solvers2test= [solver1,solver2,solver3, solver4, solver5]
        solver_strs = ['x', '1 + x**2', '3 + x + x**2 + x**3', 'Gamma(1,1), Hermite', 'Gamma(1,1), Optimal']
        poly_orders = [1,2,3, range(2,5), range(2,5),]
        dist_x      = cp.Normal()

        for isolver , ipoly_order, isolver_str in zip(solvers2test, poly_orders, solver_strs):
            metamodel_class, metamodel_basis_setting = 'PCE', ipoly_order 
            metamodel_params= {'cal_coeffs': 'Galerkin', 'dist_zeta': dist_x}
            uqhelpers.blockPrint()
            quad_doe = museuq.DoE('QUAD', 'hem', 100, dist_x)
            samples_x= quad_doe.get_samples()[0]

            x_train, x_weight = samples_x[0,:].reshape(1,-1),samples_x[1,:]
            y_train = np.squeeze(isolver(x_train))
            pce_model  = museuq.SurrogateModel(metamodel_class, metamodel_basis_setting, **metamodel_params)
            pce_model.fit(x_train, y_train, weight=x_weight)
            uqhelpers.enablePrint()
            pce_model_scores = pce_model.score(x_train, y_train, metrics=metrics, moment=np.arange(1,5))
            print('Target: {}'.format(isolver_str))
            for i, ipoly_coeffs in enumerate(pce_model.poly_coeffs):
                print('{:<6s}: {}'.format('museuq'*(i==0), np.around(ipoly_coeffs,4)))


        # print('========================TESTING: SurrogateModel.fit(), Generalized  ====================')
        # gpce_dist_to_test   = [cp.Normal(), cp.Normal(2,3), cp.Gamma(1,1), cp.Beta(1,1)]
        # gpce_opt_dist       = [cp.Normal(), cp.Normal(), cp.Gamma(1,1), cp.Beta(1,1)]
        # gpce_opt_rule       = ['hem', 'hem', 'lag', 'jacobi']
        # npoly_orders        = range(2,5)
        # dist_zeta0          = cp.Normal()

        # for i, igpce_dist in enumerate(gpce_dist_to_test):
            # dist_zeta1 = gpce_opt_dist[i]
            # print('>> Testing # {:d}: gpce: {}, zeta0: {} , zeta1: {}'.format(i, igpce_dist, dist_zeta0, dist_zeta1 ))
            # for ipoly_order in npoly_orders:
                # print('  Polynomial order: {:d}'.format(ipoly_order))
                # ## gPCE with hermite chaos
                # uqhelpers.blockPrint()
                # quad_doe = museuq.DoE('QUAD', 'hem', [ipoly_order+1], dist_zeta0)
                # samples_zeta= quad_doe.get_samples()
                # zeta_cor, zeta_weight = samples_zeta[0]
                # x_cor = igpce_dist.inv(dist_zeta0.cdf(zeta_cor))
                # zeta_poly, zeta_norms = cp.orth_ttr(ipoly_order, dist_zeta0, retall=True)
                # x_hat,coeffs = cp.fit_quadrature(zeta_poly, zeta_cor, zeta_weight, np.squeeze(x_cor), retall=True)
                # uqhelpers.enablePrint()
                # print('\t Hermite: {}'.format( np.around(coeffs,4)))


                # ## gPCE with optimal chaos
                # uqhelpers.blockPrint()
                # quad_doe = museuq.DoE('QUAD', gpce_opt_rule[i], [ipoly_order+1], dist_zeta1)
                # samples_zeta= quad_doe.get_samples()
                # zeta_cor, zeta_weight = samples_zeta[0]
                # x_cor = igpce_dist.inv(dist_zeta1.cdf(zeta_cor))
                # zeta_poly, zeta_norms = cp.orth_ttr(ipoly_order, dist_zeta1, retall=True)
                # x_hat,coeffs = cp.fit_quadrature(zeta_poly, zeta_cor, zeta_weight, np.squeeze(x_cor), retall=True)
                # uqhelpers.enablePrint()
                # print('\t Optimal: {}'.format( np.around(coeffs,4)))
    def test_surrogate_model_scores(self):
        print('========================TESTING: SurrogateModel.scores() =======================')

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
