# -*- coding: utf-8 -*-

import uqra, unittest,warnings,os, sys 
from tqdm import tqdm
import numpy as np, scipy as sp 
from uqra.solver.PowerSpectrum import PowerSpectrum
from uqra.environment import Kvitebjorn as Kvitebjorn
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import uqra.utilities.helpers as uqhelpers
import pickle

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()

data_dir = '/Users/jinsongliu/BoxSync/MUSELab/uqra/examples/JupyterNotebook'
class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

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

        data_dir_result = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/FPSO_SDOF/Data' 
        excd_prob       = 0.5/365.25/24/50
        radius_surrogate= 3
        short_term_seeds= np.arange(11)
        print('Target exceedance prob : {}'.format(excd_prob))

        y50_EC_boots_data = np.load(os.path.join(data_dir_result, 'FPSO_DoE_EC2D_T50_bootstrap.npy'))
        y50_EC_boots = np.mean(y50_EC_boots_data, axis=1)
        y50_EC_boots_std  = np.std(y50_EC_boots_data, axis=1)
        print('\n')
        print(' > Extreme reponse from EC (Bootstrap (n={:d})):'.format(y50_EC_boots_data.shape[1]))
        print('   - Data set: {}'.format(y50_EC_boots_data.shape))
        print('   - y: [mean, std]= [{:.2f}, {:.2f}]'.format(y50_EC_boots[-1],y50_EC_boots_std[-1]))
        print('   - State: u: {}, x: {}'.format(np.around(y50_EC_boots[:2],2),np.around(y50_EC_boots[2:4],2)))

        u_center = y50_EC_boots[ :2].reshape(-1, 1)
        x_center = y50_EC_boots[2:4].reshape(-1, 1)

        filename    = 'DoE_McsE7R0.npy'
        mcs_data    = np.load(os.path.join(data_dir_result, filename))
        u_test      = mcs_data[:2]
        x_test      = mcs_data[2:4]
        y_test      = np.zeros((1, u_test.shape[1]))
        data_test   = np.concatenate((u_test, x_test, y_test))
        idx_in_circle   = np.arange(u_test.shape[1])[np.linalg.norm(u_test-u_center, axis=0) < radius_surrogate]
        print(mcs_data.shape)


        for iseed in short_term_seeds:
            filename = 'FPSO_SDOF_2Hem20_MCS_Lassolars_Alpha3_ST{:d}_test.npy'.format(iseed)
            data  = np.load(os.path.join(data_dir_result, filename), allow_pickle=True)
            data_ecdf = []
            for p, n, idata in tqdm(data,desc='ST{:d}'.format(iseed), ncols=80):
                assert data_test.shape[0] == 5
                assert idata.shape[0] == 5
                ecdf_ = [p,n]
                data_test[:, idx_in_circle] = idata
                ecdf_ = uqra.utilities.helpers.ECDF(data_test.T, hinge=-1,alpha=excd_prob, compress=True)
                data_ecdf.append([p,n,ecdf_])
            data_ecdf = np.array(data_ecdf, dtype=object)
            np.save(os.path.join(data_dir_result, filename[:-8]+'ecdf.npy'), data_ecdf, allow_pickle=True)


        # for iexcd_prob in excd_prob:
            # for r in range(9):
                # try:
                    # filename = 'SDOF_3Hem_DoE_ClsE6R{:d}_y2.npy'.format(r)
                    # print(r'    - exceedance for y: {:s}'.format(filename))
                    # data_set = np.load(os.path.join(data_dir, filename))
                    # print(data_set.shape)
                    # y_ecdf   = uqhelpers.ECDF(np.array(data_set[7]).T, alpha=iexcd_prob, is_expand=False)
                    # print(len(y_ecdf.x))
                    # print(len(y_ecdf.y))
                    # print(y_ecdf.n)
                    # filename = os.path.join(data_dir,filename[:-4]+'_ecdf.p')
                    # with open(filename, 'wb') as handle:
                        # pickle.dump(y_ecdf, handle)
                # except FileNotFoundError:
                    # pass

    def test_bootstrap(self):
        data = np.arange(12).reshape(3,4)
        print(data)
        boots_data = uqhelpers.bootstrapping(data, 3)
        print(boots_data)
        boots_data = uqhelpers.bootstrapping(data, 3, 5)
        print(boots_data)
        
        # y_excd = []
        # for iexcd_prob in excd_prob:
            # y = []
            # for r in range(10):
                # filename = 'DoE_McsE6R{:d}_PCE9_LASSOLARS.npy'.format(r)
                # print(r'    - exceedance for y: {:s}'.format(filename))
                # data_set = np.load(os.path.join(data_dir, filename))
                # y.append(data_set)
            # y_ecdf   = uqhelpers.ECDF(np.array(y).T, alpha=iexcd_prob, is_expand=False)
            # filename = os.path.join(data_dir,'DoE_McsE6_PCE9_LASSOLARS_pf6_ecdf.pickle')
            # with open(filename, 'wb') as handle:
                # pickle.dump(y_ecdf, handle)


        # y_excd = []
        # for iexcd_prob in excd_prob:
            # y = []
            # for r in range(10):
                # filename = 'DoE_McsE6R{:d}_PCE9_OLSLARS.npy'.format(r)
                # print(r'    - exceedance for y: {:s}'.format(filename))
                # data_set = np.load(os.path.join(data_dir, filename))
                # y.append(data_set)
            # y_ecdf   = uqhelpers.ECDF(np.array(y).T, alpha=iexcd_prob, is_expand=False)
            # filename = os.path.join(data_dir,'DoE_McsE6_PCE9_OLSLARS_pf6_ecdf.pickle')
            # with open(filename, 'wb') as handle:
                # pickle.dump(y_ecdf, handle)

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


if __name__ == '__main__':
    unittest.main()
