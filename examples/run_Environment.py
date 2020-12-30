# -*- coding: utf-8 -*-

import uqra, unittest,warnings,os, sys
import inspect 
from tqdm import tqdm
import numpy as np, scipy as sp 
import scipy.stats as stats
from uqra.environment import Kvitebjorn as Kvitebjorn
from uqra.environment import Norway5 as Norway5


def run_Kvitebjorn():
    print('========================TESTING: Kvitebjorn =======================')

    # data_dir = '/Users/jinsongliu/BoxSync/MUSELab/uqra/uqra/environment'
    # hs1     = np.linspace(0,2.9,291)
    # hs2     = np.linspace(2.90,20, 1711)
    # hs      = np.hstack((hs1, hs2))
    # hs_pdf  = Kvitebjorn.hs_pdf(hs) 
    # np.save(os.path.join(data_dir, 'Kvitebjorn_hs'), np.vstack((hs, hs_pdf)))
    uqra_env = Kvitebjorn.Kvitebjorn()
    for r in range(10):
        data_dir    = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/MCS/Norm'
        filename = 'DoE_McsE6R{:d}.npy'.format(r)
        mcs_sampels = np.load(os.path.join(data_dir, filename))
        mcs_sampels = stats.norm().cdf(mcs_sampels)
        samples_x   = uqra_env.ppf(mcs_sampels[:2,:])
        data_dir    = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/Kvitebjorn/Norm'
        # np.save(os.path.join(data_dir, filename), samples_x)

def run_Norway5():
    print('========================TESTING: Norway5 =======================')

    # data_dir = '/Users/jinsongliu/BoxSync/MUSELab/uqra/uqra/environment'
    # hs1     = np.linspace(0,2.9,291)
    # hs2     = np.linspace(2.90,20, 1711)
    # hs      = np.hstack((hs1, hs2))
    # hs_pdf  = Norway5.hs_pdf(hs) 
    # np.save(os.path.join(data_dir, 'Kvitebjorn_hs'), np.vstack((hs, hs_pdf)))
    uqra_env = uqra.environment.Norway5()
    for r in range(10):
        data_dir    = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/ExperimentalDesign/Random'
        filename    = 'DoE_McsE6R{:d}_norm.npy'.format(r)
        mcs_sampels = np.load(os.path.join(data_dir, filename))[:uqra_env.ndim,:]
        mcs_sampels = stats.norm().cdf(mcs_sampels)
        samples_x   = uqra_env.ppf(mcs_sampels)
        data_dir    = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/UQRA_Examples/FPSO_SRUGE/TestData'
        filename    = filename.replace('DoE', uqra_env.site.capitalize())
        print('saving file: {}'.format(os.path.join(data_dir, filename)))
        np.save(os.path.join(data_dir, filename), samples_x)

def test_EC():
    P = 1000
    U, X= Kvitebjorn().environment_contour(P,T=1800,n=1000)
    EC_samples = np.concatenate((U, X), axis=0)
    np.save(os.path.join('Kvitebjorn_EC_{:d}yr_1000'.format(P)), EC_samples)


    # U, X = Norway5().environment_contour(P,T=3600,n=1000)
    # EC_samples = np.concatenate((U, X), axis=0)
    # np.save('Norway5_EC_{:d}yr_1000'.format(P), EC_samples)


    # # print(EC_samples.shape)
    # U_hub = np.arange(3,26) 
    # hub_height = 90
    # alpha = 0.1
    # U10 = U_hub * ((hub_height / 10)**(-alpha));

    # X = Norway5().target_contour(U10, P, T=3600, n=36)
    # print(X)

    ## test cdf method for Kvitebjørn
    # u = np.array([np.linspace(0,0.99999,11), np.linspace(0,0.99999,11)])
    # x = uqra_env.samples(u)
    # u_= uqra_env.cdf(x)
    # print(np.around(u,2))
    # print(np.around(x,2))
    # print(np.around(u_,2))



if __name__ == '__main__':
    run_Norway5()


