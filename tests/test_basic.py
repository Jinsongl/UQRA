# -*- coding: utf-8 -*-

import context, museuq, unittest,warnings,os, sys 
import numpy as np, chaospy as cp, scipy as sp 
import scipy.stats as stats
import museuq.utilities.metrics_collections as uq_metrics
from museuq.utilities import helpers as uqhelpers 
from museuq.utilities import dataIO as museuq_dataio
from museuq.utilities.PowerSpectrum import PowerSpectrum
from museuq.environment import Kvitebjorn as Kvitebjorn
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()

data_dir = '/Users/jinsongliu/BoxSync/MUSELab/museuq/examples/JupyterNotebook'
class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_loo(self):

        # Loading some example data
        X, y = datasets.load_boston(return_X_y=True)
        # X = X[:100,:2]
        # y = y[:100]

        # X = np.array([[0, 0], [1, 1], [2, 2]])
        # y = np.array([0, 1, 2])
        # print(X.shape)
        print(y[:5])

        # Training classifiers
        reg1 = LinearRegression()
        reg1.fit(X,y)
        y1 = reg1.predict(X)
        # print(reg1.coef_)
        print(y1[:5])

        # b = np.linalg.lstsq(X,y)[0]
        # # print(b)
        # y2 = np.dot(X, np.array(b))
        # print(y2[:5])

        mse = []
        kf  = KFold(n_splits=X.shape[0])
        residual = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # H1 = np.linalg.inv(np.dot(X_train.T, X_train))
            # H2 = np.dot(H1, X_train.T)
            # H3 = np.dot(H2, y_train)
            # y_hat = np.dot(X_test, H3)
            # residual.append(y_test[0]- y_hat[0])

            reg1.fit(X_train, y_train)
            y_pred = reg1.predict(X_test)
            residual.append(y_test[0] - y_pred[0])
            # mse.append(uq_metrics.mean_squared_error(y_test, y_pred))
        Q, R = np.linalg.qr(X)
        H = np.dot(Q, Q.T)
        h = np.diagonal(H)
        y_hat = np.dot(H, y)
        e = (y-y_hat)/(1-h)
        print(y_hat[:5])
        print('e:')
        print(np.mean(np.array(residual)**2))
        print(np.mean(np.array(e)**2))
 
        # print(uq_metrics.leave_one_out_error(X,y,is_adjusted=False))
        # print(np.mean(mse))
        

            


    def test_QuadratureDesign(self):
        print('>>> 1D quadrature design:') 

        p   = 4
        doe = museuq.QuadratureDesign(p, ndim=1, dist_names=['uniform',])
        doe.samples()
        print(' Legendre:')
        print('  {:<15s} : {}'.format('Abscissa', np.around(doe.u, 2)))
        print('  {:<15s} : {}'.format('Weights' , np.around(doe.w, 2)))

        doe = museuq.QuadratureDesign(p, ndim=1, dist_names=['normal',])
        doe.samples()
        print(' Hermite:')
        print('  {:<15s} : {}'.format('Abscissa', np.around(doe.u, 2)))
        print('  {:<15s} : {}'.format('Weights' , np.around(doe.w, 2)))

        print('>>> 1D quadrature design: Changing interval ') 
        a = -np.pi
        b =  np.pi
        loc   = a
        scale = b - loc
        print(' Legendre ({},{})'.format(np.around(a,2), np.around(b,2)))

        doe = museuq.QuadratureDesign(p, ndim=1, dist_names=['uniform',])
        doe.samples()
        print(' From chaning inverval after museuq.doe:')
        print('  {:<15s} : {}'.format('Abscissa', np.around((b-a)/2*doe.u + (a+b)/2, 2)))
        print('  {:<15s} : {}'.format('Weights' , np.around((b-a)/2*doe.w, 2)))
        
        doe = museuq.QuadratureDesign(p, ndim=1, dist_names=['uniform',], dist_theta=[(loc, scale)])
        doe.samples()
        print(' Directly from museuq.doe:')
        print('  {:<15s} : {}'.format('Abscissa', np.around(doe.u, 2)))
        print('  {:<15s} : {}'.format('Weights' , np.around(doe.w, 2)))

        print('>>> 2D quadrature design:') 

        p   = 4
        doe = museuq.QuadratureDesign(p, ndim=2, dist_names=['uniform',])
        doe.samples()
        print(' Legendre:')
        print('  {:<15s} :\n {}'.format('Abscissa', np.around(doe.u, 2)))
        print('  {:<15s} :\n {}'.format('Weights' , np.around(doe.w, 2)))

        doe = museuq.QuadratureDesign(p, ndim=2, dist_names=['normal',])
        doe.samples()
        print(' Hermite:')
        print('  {:<15s} :\n {}'.format('Abscissa', np.around(doe.u, 2)))
        print('  {:<15s} :\n {}'.format('Weights' , np.around(doe.w, 2)))

    def test_RandomDesign(self):
        doe = museuq.RandomDesign('MCS', n_samples=1e6, ndim=3, dist_names='uniform', dist_theta=[(-np.pi, 2*np.pi),]*3)
        doe.samples()

    def test_LatinHyperCube(self):
        doe = museuq.LHS(n_samples=1e3,dist_names=['uniform']*3,ndim=3,dist_theta=[(-1, 2*2)]*3)
        doe.samples()
        print(np.mean(doe.x))
        print(np.std(doe.x))


        doe = museuq.LHS(n_samples=1e3,dist_names=['uniform', 'norm'],ndim=2,dist_theta=[(-1, 2*2), (2,1)])
        doe.samples()
        print(np.mean(doe.x, axis=1))
        print(np.std(doe.x, axis=1))

    def test_OptimalDesign(self):
        """
        Optimal Design
        """
        ### Ishigami function
        # data_dir = '/Users/jinsongliu/External/MUSE_UQ_DATA/Ishigami/Data'
        ### SDOF system
        data_dir = '/Users/jinsongliu/External/MUSE_UQ_DATA/linear_oscillator/Data'
        # data_dir = 'E:\Run_MUSEUQ'
        np.random.seed(100)
        # dist_x = cp.Normal()

        ### 2D
        quad_orders = range(4,11)
        alpha = [1.0, 1.1, 1.3, 1.5, 2.0,2.5, 3.0,3.5, 5]
        dist_u= cp.Iid(cp.Normal(),2)
        for iquad_orders in quad_orders:
            basis = cp.orth_ttr(iquad_orders-1,dist_u)
            for r in range(10):
                filename  = 'DoE_McsE6R{:d}_stats.npy'.format(r)
                data_set  = np.load(os.path.join(data_dir, filename))
                samples_y = np.squeeze(data_set[:,4,:]).T
                
                filename  = 'DoE_McsE6R{:d}.npy'.format(r)
                data_set  = np.load(os.path.join(data_dir, filename))
                samples_u = data_set[0:2, :]
                samples_x = data_set[2:4, :]
                # samples_y = data_set[6  , :].reshape(1,-1)
                print('Quadrature Order: {:d}'.format(iquad_orders))
                print('Candidate samples filename: {:s}'.format(filename))
                print('   >> Candidate sample set shape: {}'.format(samples_u.shape))
                design_matrix = basis(*samples_u).T
                print('   >> Candidate Design matrix shape: {}'.format(design_matrix.shape))
                for ia in alpha:
                    print('   >> Oversampling rate : {:.2f}'.format(ia))
                    doe_size = min(int(len(basis)*ia), 10000)
                    doe = museuq.OptimalDesign('S', n_samples = doe_size )
                    doe.samples(design_matrix, u=samples_u, is_orth=True)
                    data = np.concatenate((doe.I.reshape(1,-1),doe.u,samples_x[:,doe.I], samples_y[:,doe.I]), axis=0)
                    filename = os.path.join(data_dir, 'DoE_McsE6R{:d}_p{:d}_OptS{:d}'.format(r,iquad_orders,doe_size))
                    np.save(filename, data)

                for ia in alpha:
                    print('   >> Oversampling rate : {:.2f}'.format(ia))
                    doe_size = min(int(len(basis)*ia), 10000)
                    doe = museuq.OptimalDesign('D', n_samples = doe_size )
                    doe.samples(design_matrix, u=samples_u, is_orth=True)
                    data = np.concatenate((doe.I.reshape(1,-1),doe.u,samples_x[:,doe.I], samples_y[:,doe.I]), axis=0)
                    filename = os.path.join(data_dir, 'DoE_McsE6R{:d}_p{:d}_OptD{:d}'.format(r,iquad_orders,doe_size))
                    np.save(filename, data)
                    
                    

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

    def test_weighted_exceedance(self):
        print('========================TESTING: Weighted Exceedance =======================')

        # x = np.random.normal(size=1000).reshape(1,-1)
        # res1 = stats.cumfreq(x) 

        # cdf_x = res1.lowerlimit + np.linspace(0, res1.binsize*res1.cumcount.size, res1.cumcount.size)
        # cdf_y = res1.cumcount/x.size
        # ecdf_y = 1- cdf_y
        # ecdf_x = cdf_x

        # print(np.around(ecdf_x,2))
        # print(np.around(ecdf_y,2))
        # res2 = uqhelpers.get_weighted_exceedance(x)
        # print(res2.shape)
        # print(np.around(res2[0],2))
        # print(np.around(res2[1],2))
        
        # orders = [4] ## mcs
        orders  = range(3,10) ## quad
        repeat  = range(10)

        data_dir_out= '/Users/jinsongliu/External/MUSE_UQ_DATA/linear_oscillator/Data'
        data_dir_in = '/Users/jinsongliu/External/MUSE_UQ_DATA/linear_oscillator/Data'
        for iorder in orders:
            for r in repeat:
                filename = 'DoE_IS_McRE6R{:d}_weight.npy'.format(r)
                weights  = np.load(os.path.join(data_dir_out, filename))
                ##>>> MCS results from true model
                # filename = 'DoE_IS_McRE{:d}R{:d}_stats.npy'.format(iorder,r)
                # data_out = np.load(os.path.join(data_dir_out, filename))
                # y = np.squeeze(data_out[:,4,:]).T
                filename = 'DoE_IS_QuadHem{:d}_PCE_pred_E6R{:d}.npy'.format(iorder, r)
                data_out = np.load(os.path.join(data_dir_out, filename))
                y = data_out
                print(y.shape)

                # filename = 'DoE_McRE{:d}R{:d}_stats.npy'.format(iorder, r)
                # data_out = np.load(os.path.join(data_dir, filename))
                # y = np.squeeze(data_out[:,4,:]).T
                print(r'    - exceedance for y: {:s}'.format(filename))
                for i, iy in enumerate(y):
                    print('iy.shape = {}'.format(iy.shape))
                    print('weights.shape = {}'.format(weights.shape))
                    res = stats.cumfreq(iy,numbins=iy.size,  weights=weights)
                    cdf_x = res.lowerlimit + np.linspace(0, res.binsize*res.cumcount.size, res.cumcount.size)
                    cdf_y = res.cumcount/res.cumcount[-1]
                    excd = np.array([cdf_x, cdf_y])
                    np.save(os.path.join(data_dir_out,filename[:-4]+'_y{:d}_ecdf'.format(i)), excd)

    def test_exceedance(self):
        print('========================TESTING: Lienar Oscillator =======================')

        # print('Testing: 1D')
        # a = np.random.randint(0,10,size=10)
        # print(a)
        # a_excd=uqhelpers.get_exceedance_data(a, prob=1e-3)
        # print('1D: return_index=False')
        # print(a_excd)
        # a_excd=uqhelpers.get_exceedance_data(a, prob=1e-3, return_index=True)
        # print('1D: return_index=True')
        # print(a_excd)

        # print('Testing: 2D')
        # a = np.random.randint(0,10,size=(2,10))
        # print(a)
        # a_excd=uqhelpers.get_exceedance_data(a, prob=1e-3)
        # print('2D: isExpand=False, return_index=False')
        # print(a_excd)
        # a_excd=uqhelpers.get_exceedance_data(a, prob=1e-3, return_index=True)
        # print('2D: isExpand=False, return_index=True')
        # print(a_excd)
        # a_excd=uqhelpers.get_exceedance_data(a, prob=1e-3, isExpand=True, return_index=True)
        # print('2D: isExpand=True, return_index=True')
        # print(a_excd)

        # # return_period= [1,5,10]
        # # prob_fails   = [1/(p *365.25*24*3600/1000) for p in return_period]
        # return_period= [1]
        # prob_fails   = [1e-5]
        # quad_orders  = range(3,10)
        # mcs_orders   = [6]
        # repeat       = range(10)
        # orders       = mcs_orders
        # # orders       = quad_orders 
        # return_all   = False 

        # data_dir_out= '/Users/jinsongliu/External/MUSE_UQ_DATA/linear_oscillator/Data'
        # data_dir_in = '/Users/jinsongliu/External/MUSE_UQ_DATA/linear_oscillator/Data'
        # # data_dir_out= '/Users/jinsongliu/External/MUSE_UQ_DATA/BENCH4/Data'
        # # data_dir_in = '/Users/jinsongliu/External/MUSE_UQ_DATA/BENCH4/Data'
        # # data_dir_in = '/Users/jinsongliu/Google Drive File Stream/My Drive/MUSE_UQ_DATA/linear_oscillator'
        # for ipf, ip in zip(prob_fails, return_period):
            # print('Target exceedance prob : {:.1e}'.format(ipf))
            # for iorder in orders:
                # for r in repeat:
                    # ## input
                    # filename = 'DoE_McRE6R{:d}.npy'.format(r)
                    # # filename = 'DoE_McRE7R{:d}.npy'.format(r)
                    # data_in  = np.load(os.path.join(data_dir_in, filename))  # [u1, u2,..., x1, x2...]
                    # ##>>> MCS results from surrogate model

                    # filename = 'DoE_QuadHem{:d}_GPR_pred_E6R{:d}.npy'.format(iorder, r)
                    # filename = 'DoE_QuadHem{:d}R24_mPCE_Normal_pred_E7R{:d}.npy'.format(iorder, r)
                    # filename = 'DoE_QuadHem{:d}_PCE_Normal_pred_E7R{:d}.npy'.format(iorder, r)
                    # data_out = np.load(os.path.join(data_dir_out, filename))
                    # y = data_out

                    # ##>>> MCS results from true model
                    # ## bench 4
                    # # filename = 'DoE_McRE{:d}R{:d}_y_Normal.npy'.format(iorder,r)
                    # # data_out = np.load(os.path.join(data_dir_out, filename))
                    # # y = data_out.reshape(1,-1)

                    # filename = 'DoE_McRE6R{:d}_stats.npy'.format(r)
                    # data_out = np.load(os.path.join(data_dir_out, filename))
                    # y = np.squeeze(data_out[:,4,:]).T

                    # print(y.shape)

                    # # filename = 'DoE_McRE{:d}R{:d}_stats.npy'.format(iorder, r)
                    # # data_out = np.load(os.path.join(data_dir, filename))
                    # # y = np.squeeze(data_out[:,4,:]).T

                    # print(r'    - exceedance for y: {:s}'.format(filename))
                    # for i, iy in enumerate(y):
                        # data_ = np.vstack((iy.reshape(1,-1), data_in))
                        # iexcd = uqhelpers.get_exceedance_data(data_, ipf, isExpand=True, return_all=return_all)
                        # return_all_str = '_all' if return_all else '' 
                        # np.save(os.path.join(data_dir_out,filename[:-4]+'_y{:d}_ecdf_P{:d}{}'.format(i, ip, return_all_str )), iexcd)

        # data_dir = '/Users/jinsongliu/External/MUSE_UQ_DATA/BENCH4/Data' 
        # p = 1e-5
        # print('Target exceedance prob : {:.1e}'.format(p))
        # # error_name = 'None'
        # # error_name = 'Normal'
        # error_name = 'Gumbel'
        # for r in range(10):
            # # filename = 'DoE_McRE7R{:d}_y_{:s}.npy'.format(r, error_name.capitalize())
            # # filename = 'DoE_QuadHem5_PCE_{:s}_pred_r{:d}.npy'.format(error_name.capitalize(), r)
            # filename = 'DoE_QuadHem5R24_mPCE_{:s}_pred_r{:d}.npy'.format(error_name.capitalize(), r)
            # data_set = np.load(os.path.join(data_dir, filename))
            # y        = np.squeeze(data_set)
            # print(r'    - exceedance for y: {:s}'.format(filename))
            # y_excd=uqhelpers.get_exceedance_data(y, p)
            # np.save(os.path.join(data_dir, filename[:-4]+'_ecdf_pf5.npy'), y_excd)

        data_dir = '/Users/jinsongliu/External/MUSE_UQ_DATA/Ishigami/Data' 
        pf = 1e-3
        print('Target exceedance prob : {:.1e}'.format(pf))
        # error_name = 'None'
        # error_name = 'Normal'
        # error_name = 'Gumbel'
        for p in range(10):
            # filename = 'DoE_McRE7R{:d}_y_{:s}.npy'.format(r, error_name.capitalize())
            # filename = 'DoE_QuadHem5_PCE_{:s}_pred_r{:d}.npy'.format(error_name.capitalize(), r)
            # filename = 'DoE_QuadHem5R24_mPCE_{:s}_pred_r{:d}.npy'.format(error_name.capitalize(), r)
            # filename = 'DoE_Quad_Leg{:d}_PCE_GLK_valid_y.npy'.format(p)
            filename = 'DoE_LHSE4R{:d}_y.npy'.format(p)
            data_set = np.load(os.path.join(data_dir, filename))
            y        = np.squeeze(data_set)
            print(r'    - exceedance for y: {:s}'.format(filename))
            y_excd   = uqhelpers.get_exceedance_data(y, prob=pf)
            np.save(os.path.join(data_dir, filename[:-4]+'_ecdf.npy'), y_excd)

    def test_bench4(self):
        print('========================TESTING: BENCH 4 =======================')
        data_dir    = '/Users/jinsongliu/External/MUSE_UQ_DATA/BENCH4/Data'
        model_name  = 'BENCH4'

        # ### grid points
        # x           = np.linspace(-10,20,600).reshape((1,-1))
        # solver      = museuq.Solver(model_name, x)
        # y           = solver.run()
        # res         = np.concatenate((x,y), axis=0)
        # np.save(os.path.join(data_dir,model_name.lower()), res)

        ### data from files
        for r in range(10):
            filename = 'DoE_McRE6R{:d}.npy'.format(r)
            data_set = np.load(os.path.join(data_dir, filename))
            zeta     = data_set[0,:].reshape(1,-1)
            x        = data_set[1,:].reshape(1,-1)
            solver   = museuq.Solver(model_name, x)
            y        = solver.run()
            np.save(os.path.join(data_dir,'DoE_McRE6R{:d}_y_None.npy'.format(r)), y)

    def test_Solver(self):

        # x      = (Hs,Tp) = np.array((4, 12)).reshape(2,1)
        # solver = museuq.linear_oscillator(stats2cal= ['absmax'])
        # print(solver)
        # solver.run(x)
        # print(solver.y.shape)
        # print(solver.y_stats.shape)


        x = np.arange(30).reshape(3,10)
        solver = museuq.Ishigami()
        solver.run(x)
        print(solver)
        print(solver.y.shape)

        x = np.arange(30)
        solver = museuq.xsinx()
        solver.run(x)
        print(solver)
        print(solver.y.shape)

        x = np.arange(30)
        solver = museuq.poly4th()
        solver.run(x)
        print(solver)
        print(solver.y.shape)

        x = np.arange(30).reshape(2,15)
        solver = museuq.polynomial_square_root_function()
        solver.run(x)
        print(solver)
        print(solver.y.shape)

        x = np.arange(30).reshape(2,15)
        solver = museuq.four_branch_system()
        solver.run(x)
        print(solver)
        print(solver.y.shape)

        x = np.arange(30).reshape(2,15)
        solver = museuq.polynomial_product_function()
        solver.run(x)
        print(solver)
        print(solver.y.shape)




        ### General Solver run testing 
        # print('========================TESTING: Solver =======================')

        # model_name  = 'linear_oscillator'
        # kwargs  = {
            # 'time_max'  : 100,
            # 'dt'        : 0.2,
                # }
        # tmax,dt = 1000, 0.1
        # t       = np.arange(0,tmax, dt)

        # zeta    = 0.01
        # omega_n = 2 # rad/s
        # m       = 1 
        # k       = (omega_n/2/np.pi) **2 * m 
        # c       = zeta * 2 * np.sqrt(m * k)
        # mck     = (m,c,k)
        # solver  = museuq.Solver(model_name, x)
        # y       = solver.run(**kwargs) 

        # data_dir    = '/Users/jinsongliu/External/MUSE_UQ_DATA/linear_oscillator/Data'
        # np.save(os.path.join(data_dir,'Kvitebjørn_EC_P{:d}_{:d}'.format(P, nsim)), EC_y)

        # ## run solver for EC cases
        # P, nsim     = 10, 25
        # data_dir    = '/Users/jinsongliu/BoxSync/MUSELab/museuq/museuq/environment'
        # data_set    = np.load(os.path.join(data_dir, 'Kvitebjørn_EC_P{:d}.npy'.format(P)))
        # EC_x        = data_set[2:,:]
        # model_name  = 'linear_oscillator'
        # solver      = museuq.Solver(model_name, EC_x)
        # EC_y        = np.array([solver.run(doe_method = 'EC') for _ in range(nsim)])

        # data_dir    = '/Users/jinsongliu/External/MUSE_UQ_DATA/linear_oscillator/Data'
        # np.save(os.path.join(data_dir,'Kvitebjørn_EC_P{:d}_{:d}'.format(P, nsim)), EC_y)

        # ## run solver for Hs Tp grid points
        # nsim        = 30
        # data_dir    = '/Users/jinsongliu/External/MUSE_UQ_DATA/linear_oscillator/Data'
        # filename    = 'HsTp_grid118.npy'
        # data_set    = np.load(os.path.join(data_dir, filename))
        # model_name  = 'linear_oscillator'
        # solver      = museuq.Solver(model_name, data_set)
        # grid_out    = np.array([solver.run(doe_method = 'GRID') for _ in range(nsim)])
        # np.save(os.path.join(data_dir,'HsTp_grid118_out'), grid_out)

        # data_set  = np.load('DoE_McRE3R0.npy')
        # x_samples = data_set[2:,:]

        # model_name  = 'linear_oscillator'
        # solver      = museuq.Solver(model_name, x_samples)
        # kwargs      = {'doe_method': 'MCS'}
        # samples_y   = solver.run(**kwargs )
        # np.save('test_linear_oscillator_y', samples_y)
        # # filename_tags = ['R0']
        # # filename_tags = [itag+'_y' for itag in filename_tags]
        # # museuq_dataio.save_data(samples_y, 'test_linear_oscillator', os.getcwd(), filename_tags)
        # samples_y_stats = solver.get_stats()
        # np.save('test_linear_oscillator_y_stats', samples_y_stats)
        # # filename_tags = [itag+'_y_stats' for itag in filename_tags]
        # # museuq_dataio.save_data(samples_y_stats, 'test_linear_oscillator', os.getcwd(), filename_tags)

    def test_surrogate_model(self):
        print('========================TESTING: SurrogateModel.fit(), ~Normal =======================')
        solver1 = lambda x: x
        solver2 = lambda x: x**2 + 1
        solver3 = lambda x: x**3 + x**2 + x + 3
        solver4 = lambda x: cp.Gamma(1,1).inv(cp.Normal(0,1).cdf(x))
        solver5 = lambda x: cp.Gamma(1,1).inv(cp.Gamma(1,1).cdf(x))

        upper_tail_probs= [0.999,0.9999,0.99999]
        moment2cal      = [1,2,3,4]
        metrics2cal     = [ 'explained_variance_score', 'mean_absolute_error', 'mean_squared_error',
                    'median_absolute_error', 'r2_score', 'r2_score_adj', 'moment', 'mquantiles']
        
        sample_weight = None
        multioutput   = 'uniform_average'
        # squared       = True

        solvers2test= [solver1,solver2,solver3, solver4, solver5]
        solver_strs = ['x', '1 + x**2', '3 + x + x**2 + x**3', 'Gamma(1,1), Hermite', 'Gamma(1,1), Optimal']
        poly_orders = range(2,5)
        dist_zeta   = cp.Normal()
        dist_x      = cp.Normal()

        fit_method  = 'GLK'
        for isolver , isolver_str in zip(solvers2test, solver_strs):
            for ipoly_order in poly_orders:
                # uqhelpers.blockPrint()
                doe = museuq.QuadratureDesign(ipoly_order+1, ndim = 1, dist_names=['normal'])
                doe.samples()
                doe.x = doe.u
                train_y = np.squeeze(isolver(doe.x))
                train_y = np.array([train_y,train_y]).T
                pce_model  = museuq.PCE(ipoly_order, dist_zeta)
                print(len(pce_model.basis[0]))
                pce_model.fit(doe.u, train_y, w=doe.w, fit_method=fit_method)
                pce_model.predict(doe.u, train_y, metrics=metrics2cal, prob=upper_tail_probs, moment=moment2cal, sample_weight=sample_weight, multioutput=multioutput)
                # pce_model.fit(x_train, y_train, weight=x_weight)
                # uqhelpers.enablePrint()
                # pce_model_scores = pce_model.score(x_train, y_train, metrics=metrics, moment=np.arange(1,5))
                # print('Target: {}'.format(isolver_str))
                # for i, ipoly_coeffs in enumerate(pce_model.poly_coeffs):
                    # print('{:<6s}: {}'.format('museuq'*(i==0), np.around(ipoly_coeffs,4)))

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


    def test_Kvitebjorn(self):
        print('========================TESTING: Kvitebjorn =======================')

        data_dir = '/Users/jinsongliu/BoxSync/MUSELab/museuq/museuq/environment'
        # hs1     = np.linspace(0,2.9,291)
        # hs2     = np.linspace(2.90,20, 1711)
        # hs      = np.hstack((hs1, hs2))
        # hs_pdf  = Kvitebjorn.hs_pdf(hs) 
        # np.save(os.path.join(data_dir, 'Kvitebjorn_hs'), np.vstack((hs, hs_pdf)))

        # n = 1e6 
        # samples_x = Kvitebjorn.samples(n)
        # np.save(os.path.join(data_dir, 'Kvitebjorn_samples_n'), samples_x)

        # return EC from Kvitebjorn
        P = 10
        EC_samples = Kvitebjorn.EC(P)
        np.save(os.path.join(data_dir, 'Kvitebjorn_EC_P{:d}'.format(P)), EC_samples)

        # ## test cdf method for Kvitebjørn
        # u = np.array([np.linspace(0,0.99999,11), np.linspace(0,0.99999,11)])
        # x = Kvitebjorn.samples(u)
        # u_= Kvitebjorn.cdf(x)
        # print(np.around(u,2))
        # print(np.around(x,2))
        # print(np.around(u_,2))






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
