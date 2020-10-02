#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import uqra, warnings
import numpy as np, os, sys
import scipy.stats as stats
import scipy
import scipy.io
import pickle
from tqdm import tqdm
# warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()


def uniform_ball(d,n, r=1):
    """
    generate n samples uniformly distributed in d-dimensional Ball
    with radius r

    """
    intd = int(d)
    intn = int(n)
    if intd != d:
        print('Only integer taken for dimension d')
    if intn != n:
        print('Only integer taken for size n')

    z = stats.norm.rvs(0,1,size=(intd, intn))
    z = z/np.linalg.norm(z, axis=0)
    u = stats.uniform.rvs(0,1,size=intn) 
    u = r * z * u**(1/intd)
    return u

def main():
    ## ------------------------ Displaying set up ------------------- ###
    print('\n'+'#' * 80)
    print(' >>>  Start simulation: {:s}'.format(__file__))
    print('#' * 80 + '\n')
    np.random.seed(100)
    np.set_printoptions(precision=4)
    np.set_printoptions(threshold=8)
    np.set_printoptions(suppress=True)
    Kvitebjorn = uqra.environment.Kvitebjorn()
    pf = 0.5/(50*365.25*24)

    ## ------------------------ Simulation Parameters ----------------- ###
    # solver    = uqra.FPSO(random_state =theta)
    solver    = uqra.FPSO()
    simparams = uqra.Parameters(solver, doe_method=['CLS4', 'D'], fit_method='LASSOLARS')
    simparams.set_udist('norm')
    simparams.x_dist     = Kvitebjorn
    simparams.pce_degs   = np.array(range(2,11))
    simparams.n_cand     = int(1e5)
    simparams.n_test     = int(1e6)
    simparams.n_pred     = int(1e7)
    simparams.n_splits   = 50
    simparams.alphas     = 1
    n_initial = 20
    simparams.info()

    ## ----------- Define U distributions ----------- ###
    sampling_domain = -simparams.u_dist[0].ppf(1e-7)
    sampling_space  = 'u'
    dist_support    = None        ## always in x space

    # u_dist = [stats.uniform(-1,2),]* solver.ndim
    # sampling_domain = [[-1,1],] * solver.ndim 
    # sampling_space  = 'u'
    # dist_support    = np.array([[0, 18],[0,35]]) ## always in x space
    # fname_cand      = 'FPSO_SURGE_DoE_LhsE5_Uniform.npy'
    # simparams.topy_center = [np.array([0,0]).reshape(-1,1),]
    # simparams.topy_radii   = [[[-1,1],] * solver.ndim,]

    ## ----------- Candidate data set for DoE ----------- ###
    # print(' > Getting candidate data set...')
    # fname_cand      = 'FPSO_SURGE_DoE_UnfE5_Norm.npy'
    # u_cand = uniform_ball(simparams.ndim, int(1.5*simparams.n_cand), r=sampling_domain)
    # fname_cand      = 'FPSO_SURGE_DoE_Cls4E5.npy'
    # u_cand = np.load(os.path.join(simparams.data_dir_result, 'TestData', fname_cand)) 
    # x_cand = uqra.inverse_rosenblatt(simparams.x_dist, u_cand, simparams.u_dist, support=dist_support)
    # ux_isnan = np.zeros(u_cand.shape[1])
    # for ix, iu in zip(x_cand, u_cand):
        # ux_isnan = np.logical_or(ux_isnan, np.isnan(ix))
        # ux_isnan = np.logical_or(ux_isnan, np.isnan(iu))
    # x_cand = x_cand[:, np.logical_not(ux_isnan)]
    # u_cand = u_cand[:, np.logical_not(ux_isnan)]
    # x_cand = x_cand[:, :simparams.n_cand]
    # u_cand = u_cand[:, :simparams.n_cand]
    # print('   - {:<25s} : {}'.format(' Dataset (U)', u_cand.shape))
    # print('   - {:<25s} : {}, {}'.format(' U [min(U), max(U)]', np.amin(u_cand, axis=1), np.amax(u_cand, axis=1)))
    # print('   - {:<25s} : {}, {}'.format(' X [min(X), max(X)]', np.amin(x_cand, axis=1), np.amax(x_cand, axis=1)))

    # data = np.concatenate((u_cand, x_cand), axis=0)
    # filename = 'DeepCWind_DoE_UniformE5_Norm.npy' 
    # np.save(os.path.join(simparams.data_dir_result, 'TestData', filename), data)


    # ----------- predict data set ----------- ###
    for i in range(10):
        filename = 'CDF_McsE7R{:d}.npy'.format(i)
        print('1. u_cdf')
        u_cdf  = np.load(os.path.join(simparams.data_dir_sample, 'CDF', filename))[:simparams.ndim]
        print('2. u_pred')
        u_pred = np.array([idist.ppf(iu_cdf) for idist, iu_cdf in zip(simparams.u_dist, u_cdf)]) 
        print('3. x_pred')
        x_pred = uqra.inverse_rosenblatt(simparams.x_dist, u_pred, simparams.u_dist, support=dist_support)
        ## case like x is nan bc u outside sampling_domain should keep 
        while np.isnan(x_pred).any():
            x_isnan= np.zeros(x_pred.shape[1])
            for ix in x_pred:
                x_isnan = np.logical_or(x_isnan, np.isnan(ix))
            print('nan found in x_pred: {:d}'.format(np.sum(x_isnan)))
            u_pred = u_pred[:,np.logical_not(x_isnan)]
            x_pred = x_pred[:,np.logical_not(x_isnan)]
            u_cdf1  = stats.uniform(0,1).rvs(size=(simparams.ndim, np.sum(x_isnan)))
            u_pred1 = np.array([idist.ppf(iu_cdf) for idist, iu_cdf in zip(simparams.u_dist, u_cdf1)]) 
            print(u_pred1)
            x_pred1 = uqra.inverse_rosenblatt(simparams.x_dist, u_pred1, simparams.u_dist, support=dist_support)
            u_pred  = np.concatenate((u_pred, u_pred1), axis=1)
            x_pred  = np.concatenate((x_pred, x_pred1), axis=1)
        data = np.concatenate((u_pred, x_pred), axis=0)
        print(data.shape)
        np.save(os.path.join(simparams.data_dir_result,'TestData', '{:s}_Cls4E7R{:d}_pred.npy'.format(solver.nickname, i)), data)


if __name__ == '__main__':
    main()
