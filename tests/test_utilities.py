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
import scipy.stats as stats

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()

def ground2hub(x, hub_height = 90, alpha=0.1):

    u10, hs, tp = x
    uhub = u10 * ((hub_height / 10)**(alpha))
    return np.array([uhub, hs, tp])

def hub2ground(x, hub_height = 90, alpha=0.1):

    uhub, hs, tp = x
    u10 = uhub * ((hub_height / 10)**(-alpha))
    return np.array([u10, hs, tp])
data_dir = '/Users/jinsongliu/BoxSync/MUSELab/uqra/examples/JupyterNotebook'
def draw_circle(r=1,origin=[0,0],n=1000):
    theta =np.linspace(0,2*np.pi,n)
    x = r*np.cos(theta) + origin[0]
    y = r*np.sin(theta) + origin[1]
    cood = np.array([x,y])
    return cood
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

    def rosenblatt(self):
        Norway5 = uqra.environment.Norway5()
        # Kvitebjorn = uqra.environment.Kvitebjorn()
        # u_dist = [stats.norm(),]*Kvitebjorn.ndim
        # # u = stats.norm.rvs(0,1,size=(Kvitebjorn.ndim,1000))
        # u = stats.norm.rvs(0,1,size=(Norway5.ndim,1000))
        # support = np.array([[3,16],[2,4]])

        # support = [[3,16],None]
        # x = uqra.utilities.helpers.inverse_rosenblatt(Norway5, u, u_dist, support)
        # # x = uqra.utilities.helpers.inverse_rosenblatt(Kvitebjorn, u, u_dist, support)
        # print('Inverse Rosenblatt ')
        # print(u)
        # print(x)

        # print('Forward Rosenblatt ')
        # u = uqra.utilities.helpers.rosenblatt(Norway5, x, u_dist, support)
        # # u = uqra.utilities.helpers.rosenblatt(Kvitebjorn, x, u_dist, support)
        # print(u)
        # print(x)
        hub_height = 90
        alpha      = 0.1
        Uhub_range = np.array([3,25])
        hs_domains, tp_domains = None, None
        u10_domains= Uhub_range * ((hub_height / 10)**(-alpha))
        subdomains_ground = [u10_domains, hs_domains, tp_domains]
        subdomains_hub = [Uhub_range, hs_domains, tp_domains]
        EC2D_line = np.load('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/FPSO_SURGE/TestData/Norway5_EC_50yr_1000.npy')
        # print(EC2D_line.shape)
        u_dist = [stats.norm(),] * 3
        # EC2D_x_line = uqra.inverse_rosenblatt(Norway5, EC2D_line[:3], u_dist, support=subdomains_ground)
        # print(EC2D_x_line)
        # EC2D_u = draw_circle(4.71, n=36)
        # EC2D_u = np.concatenate((EC2D_u,0*EC2D_u), axis=0)[:3]
        # print(EC2D_u.shape)
        # print(EC2D_u)
        # EC2D_x = uqra.inverse_rosenblatt(Norway5, EC2D_u, u_dist, support=subdomains_ground)
        # EC2D_x = ground2hub(EC2D_x)
        data_dir = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/DeepCWind/TestData' 
        for i in range(1,10):
            filename = 'CDF_McsE7R{:d}.npy'.format(i)
            print('1. u_cdf')
            u_cdf  = np.load(os.path.join('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/CDF', filename))[:3]
            print('2. u_pred')
            u_pred = np.array([idist.ppf(iu_cdf) for idist, iu_cdf in zip(u_dist, u_cdf)]) 
            print('3. x_pred')
            x_pred = uqra.inverse_rosenblatt(Norway5, u_pred, u_dist, support=subdomains_ground)
            while np.isnan(x_pred).any():
                x_isnan= np.zeros(x_pred.shape[1])
                for ix in x_pred:
                    x_isnan = np.logical_or(x_isnan, np.isnan(ix))
                print('nan found in x_pred: {:d}'.format(np.sum(x_isnan)))
                u_pred = u_pred[:,np.logical_not(x_isnan)]
                x_pred = x_pred[:,np.logical_not(x_isnan)]
                u_cdf1  = stats.uniform(0,1).rvs(size=(len(u_dist), np.sum(x_isnan)))
                u_pred1 = np.array([idist.ppf(iu_cdf) for idist, iu_cdf in zip(u_dist, u_cdf1)]) 
                print(u_pred1)
                x_pred1 = uqra.inverse_rosenblatt(Norway5, u_pred1, u_dist, support=subdomains_ground)
                u_pred  = np.concatenate((u_pred, u_pred1), axis=1)
                x_pred  = np.concatenate((x_pred, x_pred1), axis=1)
            data = np.concatenate((u_pred, x_pred), axis=0)
            print(data.shape)
            np.save(os.path.join(data_dir, 'DeepCWind_McsE7R{:d}_pred.npy'.format(i)), data)

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


    def ellipsoid_tool(self):
        # hub_height = 90
        # alpha      = 0.1
        # Uhub_range = np.array([3,25])
        # hs_domains, tp_domains = None, None
        # u10_domains  = Uhub_range*  ((hub_height / 10)**(-alpha))
        # subdomains_ground = [u10_domains, hs_domains, tp_domains]
        # subdomains_hub = [Uhub_range, hs_domains, tp_domains]
        # Uhub_rated = 11.3
        # U10_rated  = Uhub_rated *  ((hub_height / 10)**(-alpha))
        # solver  = uqra.Solver('DeepCWind', 3)
        # Norway5 = uqra.environment.Norway5()
        # model_name    = solver.nickname# 'FPSO_SURGE'
        # return_period = 50 # years
        # sim_duration  = 1 # hours
        # pf            = sim_duration/(return_period*365.25*24)
        # beta          = -stats.norm.ppf(pf)
        # data_dir_samples = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples'
        # model_dir        = os.path.join('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA', model_name)
        # data_dir_result  = os.path.join(model_dir, 'Data')
        # figure_dir       = os.path.join(model_dir, 'Figures')

        # EC2D_u_line = np.load('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/FPSO_SURGE/TestData/Norway5_EC_50yr_1000.npy')
        # u_dist = [stats.norm(),]*solver.ndim
        # EC2D_x_line = uqra.inverse_rosenblatt(Norway5, EC2D_u_line[:3], u_dist, support=subdomains_ground)
        # EC2D_x_line = ground2hub(EC2D_x_line)

        # with open(os.path.join(model_dir, 'DeepCWindAdapIS_McsD_Alpha1_ST0_Parameters.pkl'), 'rb') as input_file:
            # simparams = pickle.load(input_file)
            
        # # print(simparams.topy_center)
        # # print(simparams.topy_distance)

        # train_data = np.load('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/DeepCWind/DeepCWindAdapIS3Hem23Hem2_McsD_Alpha1_ST0_Train.npy')
        # # print(train_data.shape)
        # pred_topy = np.load('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/DeepCWind/DeepCWindAdapIS3Hem23Hem2_McsD_Alpha1_ST0_topy.npy')
        # # print(pred_topy.shape)
        # n_topy = 44
        # ipred_topy = pred_topy[0]
        # ipred_topy_u = ipred_topy[:3]
        # center, radii, rotation = uqra.EllipsoidTool().getMinVolEllipse(ipred_topy_u.T)
        # print('center: {}'.format(center))
        # print('radii : {}'.format(radii))
        # print('rotation: {}'.format(rotation))
        # new_train = np.load('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/DeepCWind/DeepCWindAdapIS3Hem23Hem2_McsD_Alpha1_ST0_new_train.npy')
        # print(new_train.shape)

        ndim=3
        data = stats.uniform.rvs(-1,2,size=(ndim,10000))
        c = np.zeros(ndim)
        r = np.ones(ndim)
        R = np.identity(ndim)
        idx0, idx1 = uqra.samples_within_ellipse(data, c, r, R)
        data0 = data[:,idx0]
        data1 = data[:,idx1]

        idx0 = np.linalg.norm(data, axis=0) < 1
        idx1 = np.logical_not(idx0)

        data00= data[:,idx0]
        data11= data[:,idx1]

        print(np.array_equal(data0, data00))
        print(np.array_equal(data1, data11))





if __name__ == '__main__':
    unittest.main()
