# -*- coding: utf-8 -*-

import context, museuq, unittest,warnings
import numpy as np, chaospy as cp, os, sys
import scipy.stats as stats
from museuq.utilities import helpers as uqhelpers 
from museuq.utilities import constants as const
from museuq.utilities import dataIO as museuq_dataio
from museuq.utilities.PowerSpectrum import PowerSpectrum
from museuq.environment import Kvitebjorn as Kvitebjorn

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()

data_dir = '/Users/jinsongliu/BoxSync/MUSELab/museuq/examples/JupyterNotebook'
class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_gauss_quadrature(self):
        """
        https://keisan.casio.com/exec/system/1329114617
        """

        print('========================TESTING: 1D GAUSS QUADRATURE=======================')
        dists2test = [cp.Uniform(-1,1), cp.Normal(), cp.Gamma(1,1), cp.Beta(1,1)]
        rules2test = ['leg', 'hem', 'lag', 'jacobi']
        order2test = [2,3,4,5,6,7,8]
        for idist2test, irule2test in zip(dists2test, rules2test):
            print('-'*50)
            print('>>> Gauss Quadrature with polynominal: {}'.format(const.DOE_RULE_FULL_NAMES[irule2test.lower()]))
            uqhelpers.blockPrint()
            quad_doe = museuq.DoE('QUAD', irule2test, order2test, idist2test)
            museuq_samples = quad_doe.get_samples()
            # quad_doe.disp()
            uqhelpers.enablePrint()
            if irule2test == 'hem':
                for i, iorder in enumerate(order2test):
                    print('>>> order : {}'.format(iorder))
                    coord1d_e, weight1d_e = np.polynomial.hermite_e.hermegauss(iorder)
                    print('{:<15s}: {}'.format('probabilist', np.around(coord1d_e,2)))
                    coord1d, weight1d = np.polynomial.hermite.hermgauss(iorder)
                    print('{:<15s}: {}'.format('physicist', np.around(coord1d,2)))
                    print('{:<15s}: {}'.format('museuq', np.around(np.squeeze(museuq_samples[i][:-1,:]),2)))

            elif irule2test == 'leg':
                for i, iorder in enumerate(order2test):
                    print('>>> order : {}'.format(iorder))
                    coord1d, weight1d = np.polynomial.legendre.leggauss(iorder)
                    print('{:<15s}: {}'.format('numpy ', np.around(coord1d,2)))
                    print('{:<15s}: {}'.format('museuq', np.around(np.squeeze(museuq_samples[i][:-1,:]),2)))
            elif irule2test == 'lag':
                for i, iorder in enumerate(order2test):
                    print('>>> order : {}'.format(iorder))
                    coord1d, weight1d = np.polynomial.laguerre.laggauss(iorder)
                    print('{:<15s}: {}'.format('numpy ', np.around(coord1d,2)))
                    print('{:<15s}: {}'.format('museuq', np.around(np.squeeze(museuq_samples[i][:-1,:]),2)))
            elif irule2test == 'jacobi':
                print('NOT TESTED YET')


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
            print('>>> Testing # {:d}: gpce: {}, zeta0: {} , zeta1: {}'.format(i, igpce_dist, dist_zeta0, dist_zeta1 ))
            for ipoly_order in npoly_orders:
                print('  Polynomial order: {:d}'.format(ipoly_order))
                ## gPCE with hermite chaos
                uqhelpers.blockPrint()
                quad_doe = museuq.DoE('QUAD', 'hem', [ipoly_order+1], dist_zeta0)
                samples_zeta= quad_doe.get_samples()
                zeta_cor, zeta_weight = samples_zeta[0]
                zeta_cor = zeta_cor.reshape((len(dist_zeta0),-1))
                x_cor = igpce_dist.inv(dist_zeta0.cdf(zeta_cor))
                zeta_poly, zeta_norms = cp.orth_ttr(ipoly_order, dist_zeta0, retall=True)
                x_hat,coeffs = cp.fit_quadrature(zeta_poly, zeta_cor, zeta_weight,np.squeeze(x_cor),retall=True)
                uqhelpers.enablePrint()

                print('\t Hermite: {}'.format( np.around(coeffs,4)))

                ## gPCE with optimal chaos
                uqhelpers.blockPrint()
                quad_doe = museuq.DoE('QUAD', gpce_opt_rule[i], [ipoly_order+1], dist_zeta1)
                samples_zeta= quad_doe.get_samples()
                zeta_cor, zeta_weight = samples_zeta[0]
                zeta_cor = zeta_cor.reshape((len(dist_zeta1),-1))
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
        np.save('test_linear_oscillator_y',y)

    def test_exceedance(self):
        y = np.load('DoE_McRE3R0_y_stats.npy')
        y = np.squeeze(y[:,4,:2]).T
        print('y shape: {}'.format(y.shape))
        y_excd=uqhelpers.get_exceedance_data(y, 1e-2)
        np.save('test_linear_oscillator_excd', y_excd)


    def test_Solver_run(self):
        # data_set  = np.load('EC_pfe4.npy')
        # x_samples = data_set.T
        # print(x_samples.shape)

        data_set  = np.load('DoE_McRE3R0.npy')
        x_samples = data_set[2:,:]

        model_name  = 'linear_oscillator'
        solver      = museuq.Solver(model_name, x_samples)
        kwargs      = {'doe_method': 'MCS'}
        samples_y   = solver.run(**kwargs )
        np.save('test_linear_oscillator_y', samples_y)
        # filename_tags = ['R0']
        # filename_tags = [itag+'_y' for itag in filename_tags]
        # museuq_dataio.save_data(samples_y, 'test_linear_oscillator', os.getcwd(), filename_tags)
        samples_y_stats = solver.get_stats()
        np.save('test_linear_oscillator_y_stats', samples_y_stats)
        # filename_tags = [itag+'_y_stats' for itag in filename_tags]
        # museuq_dataio.save_data(samples_y_stats, 'test_linear_oscillator', os.getcwd(), filename_tags)

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

    def test_Kvitebjorn(self):
        print('========================TESTING: Kvitebjorn =======================')


        hs1     = np.linspace(0,2.9,291)
        hs2     = np.linspace(2.90,20, 1711)
        hs      = np.hstack((hs1, hs2))
        hs_pdf  = Kvitebjorn.hs_pdf(hs) 
        np.save(os.path.join(data_dir, 'Kvitebjorn_hs'), np.vstack((hs, hs_pdf)))


        u = np.arange(0,1,1000)

        n = 1e6 
        samples_x = Kvitebjorn.samples(n)
        np.save(os.path.join(data_dir, 'Kvitebjorn_samples_n'), samples_x)



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
