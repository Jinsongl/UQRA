# -*- coding: utf-8 -*-

import uqra, unittest,warnings,os, sys 
from tqdm import tqdm
import numpy as np
import scipy as sp 
from uqra.solver.PowerSpectrum import PowerSpectrum
from uqra.environment import Kvitebjorn as Kvitebjorn
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import pickle
import uqra

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()

data_dir = '/Users/jinsongliu/BoxSync/MUSELab/uqra/examples/JupyterNotebook'
class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

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


        # data_dir = '/Users/jinsongliu/External/MUSE_UQ_DATA/Ishigami/Data' 
        # excd_prob= [1e-6]
        # print('Target exceedance prob : {}'.format(excd_prob))
        # y_excd = []
        # for iexcd_prob in excd_prob:
            # y = []
            # for r in range(10):
                # filename = 'DoE_McsE6R{:d}_PCE9_OLS.npy'.format(r)
                # print(r'    - exceedance for y: {:s}'.format(filename))
                # data_set = np.load(os.path.join(data_dir, filename))
                # y.append(data_set)
            # y_ecdf   = uqhelpers.ECDF(np.array(y).T, alpha=iexcd_prob, is_expand=False)
            # filename = os.path.join(data_dir,'DoE_McsE6_PCE9_OLS_pf6_ecdf.pickle')
            # with open(filename, 'wb') as handle:
                # pickle.dump(y_ecdf, handle)
        data = sp.io.loadmat('/Users/jinsongliu/Downloads/MCS_FPSO/Data/DoE_Mcs.mat')
        print(data.keys())
        mcs_data = np.concatenate((data['Y'],data['U'],data['X']), axis=0)
        print(mcs_data.shape)
        mcs_ecdf = uqra.utilities.helpers.get_exceedance_data(mcs_data, is_expand=True)
        print(dir(mcs_ecdf))
        print(mcs_ecdf.x.shape)
        print(mcs_ecdf.y.shape)
        print(mcs_ecdf.n)


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




if __name__ == '__main__':
    unittest.main()
