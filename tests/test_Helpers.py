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

        data_dir = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/FPSO_SDOF' 
        data_dir_result = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/FPSO_SDOF/Data/NonAdap_PCE'  
        excd_prob       = 0.5/365.25/24/50
        radius_surrogate= 3
        short_term_seeds= np.arange(11)
        short_term_seeds_applied = np.setdiff1d(np.arange(11), np.array([]))
        print('Target exceedance prob : {}'.format(excd_prob))

        print('------------------------------------------------------------')
        print('>>> Environmental Contour for Model: FPSO                   ')
        print('------------------------------------------------------------')
        filename    = 'FPSO_DoE_EC2D_T50_y.npy' 
        EC2D_data_y = np.load(os.path.join(data_dir, filename))[short_term_seeds_applied,:] 
        filename    = 'FPSO_DoE_EC2D_T50.npy' 
        EC2D_data_ux= np.load(os.path.join(data_dir, filename))[:4]

        EC2D_median = np.median(EC2D_data_y, axis=0)
        EC2D_data   = np.concatenate((EC2D_data_ux,EC2D_median.reshape(1,-1)), axis=0)
        y50_EC      = EC2D_data[:,np.argmax(EC2D_median)]

        print(' > Extreme reponse from EC:')
        print('   - {:<25s} : {}'.format('EC data set', EC2D_data_y.shape))
        print('   - {:<25s} : {}'.format('y0', np.array(y50_EC[-1])))
        print('   - {:<25s} : {}'.format('Design state (u,x)', y50_EC[:4]))

        np.random.seed(100)
        EC2D_y_boots      = uqra.bootstrapping(EC2D_data_y, 100) 
        EC2D_boots_median = np.median(EC2D_y_boots, axis=1)
        y50_EC_boots_idx  = np.argmax(EC2D_boots_median, axis=-1)
        y50_EC_boots_ux   = np.array([EC2D_data_ux[:,i] for i in y50_EC_boots_idx]).T
        y50_EC_boots_y    = np.max(EC2D_boots_median,axis=-1) 
        y50_EC_boots      = np.concatenate((y50_EC_boots_ux, y50_EC_boots_y.reshape(1,-1)), axis=0)
        y50_EC_boots_mean = np.mean(y50_EC_boots, axis=1)
        y50_EC_boots_std  = np.std(y50_EC_boots, axis=1)
        print(' > Extreme reponse from EC (Bootstrap (n={:d})):'.format(EC2D_y_boots.shape[0]))
        print('   - {:<25s} : {}'.format('Bootstrap data set', EC2D_y_boots.shape))
        print('   - {:<25s} : [{:.2f}, {:.2f}]'.format('y50[mean, std]',y50_EC_boots_mean[-1], y50_EC_boots_std[-1]))
        print('   - {:<25s} : {}'.format('Design state (u,x)', y50_EC_boots_mean[:4]))

        u_center = y50_EC_boots_mean[ :2].reshape(-1, 1)
        x_center = y50_EC_boots_mean[2:4].reshape(-1, 1)
        print(' > Important Region based on EC(boots):')
        print('   - {:<25s} : {}'.format('Radius', radius_surrogate))
        print('   - {:<25s} : {}'.format('Center U', np.squeeze(u_center)))
        print('   - {:<25s} : {}'.format('Center X', np.squeeze(x_center)))
        print('================================================================================')


        filename    = 'DoE_McsE7R0.npy'
        mcs_data    = np.load(os.path.join(data_dir, filename))
        u_test      = mcs_data[:2]
        x_test      = mcs_data[2:4]
        y_test      = np.zeros((1, u_test.shape[1]))
        data_test   = np.concatenate((u_test, x_test, y_test))
        idx_in_circle   = np.arange(u_test.shape[1])[np.linalg.norm(u_test-u_center, axis=0) < radius_surrogate]
        print(mcs_data.shape)

        for iseed in short_term_seeds_applied:
            filename = 'FPSO_SDOF_2Hem20_McsD_Lassolars_Alpha1.2_ST{:d}_test.npy'.format(iseed)
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

    def test_bootstrap(self):
        data = np.arange(12).reshape(3,4)
        print(data)
        boots_data = uqhelpers.bootstrapping(data, 3)
        print(boots_data)
        boots_data = uqhelpers.bootstrapping(data, 3, 5)
        print(boots_data)

if __name__ == '__main__':
    unittest.main()
