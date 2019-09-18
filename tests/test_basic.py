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
