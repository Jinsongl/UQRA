#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import numpy as np
    # ## ------------------------ MCS Benchmark ----------------- ###
    # print('------------------------------------------------------------')
    # print('>>> Monte Carlo Sampling for Model: FPSO                    ')
    # print('------------------------------------------------------------')
    filename = 'FPSO_SDOF_DoE_McsE7R0.npy'
    mcs_data = np.load(os.path.join(simparams.data_dir_result, filename))
    # mcs_data_ux, mcs_data_y = mcs_data[:4], mcs_data[4+short_term_seeds_applied,:]
    mcs_data_ux, mcs_data_y = mcs_data[:4], mcs_data[4:,:]
    y50_MCS  = uqra.metrics.mquantiles(mcs_data_y.T, 1-pf, multioutput='raw_values')
    y50_MCS_mean  = np.mean(y50_MCS)
    y50_MCS_std   = np.std(y50_MCS)

    mcs_data_mean = np.mean(mcs_data_y, axis=0)
    y50_mean_MCS  = uqra.metrics.mquantiles(mcs_data_mean, 1-pf, multioutput='raw_values')
    y50_mean_MCS_idx = (np.abs(mcs_data_mean-y50_mean_MCS)).argmin()
    y50_mean_MCS_ux  = mcs_data_ux[:,y50_mean_MCS_idx]
    y50_mean_MCS  = np.array(list(y50_mean_MCS_ux)+[y50_mean_MCS,])

    print(' > Extreme reponse from MCS:')
    print('   - {:<25s} : {}'.format('Data set', mcs_data_y.shape))
    print('   - {:<25s} : {}'.format('y50 MCS', y50_MCS))
    print('   - {:<25s} : {}'.format('y50 MCS [mean, std]', np.array([y50_MCS_mean, y50_MCS_std])))
    print('   - {:<25s} : {}'.format('y50 Mean MCS', np.array(y50_mean_MCS[-1])))
    print('   - {:<25s} : {}'.format('Design state (u,x)', y50_mean_MCS[:4]))
    ## ------------------------ Environmental Contour ----------------- ###
    print('------------------------------------------------------------')
    print('>>> Environmental Contour for Model: FPSO                   ')
    print('------------------------------------------------------------')
    filename    = 'FPSO_DoE_EC2D_T50_y.npy' 
    EC2D_data_y = np.load(os.path.join(simparams.data_dir_result, filename))[short_term_seeds_applied,:] 
    filename    = 'FPSO_DoE_EC2D_T50.npy' 
    EC2D_data_ux= np.load(os.path.join(simparams.data_dir_result, filename))[:4]

    EC2D_median = np.median(EC2D_data_y, axis=0)
    EC2D_data   = np.concatenate((EC2D_data_ux,EC2D_median.reshape(1,-1)), axis=0)
    y50_EC      = EC2D_data[:,np.argmax(EC2D_median)]

    print(' > Extreme reponse from EC:')
    print('   - {:<25s} : {}'.format('EC data set', EC2D_data_y.shape))
    print('   - {:<25s} : {}'.format('y0', np.array(y50_EC[-1])))
    print('   - {:<25s} : {}'.format('Design state (u,x)', y50_EC[:4]))

    # np.random.seed(100)
    # EC2D_y_boots      = uqra.bootstrapping(EC2D_data_y, 100) 
    # EC2D_boots_median = np.median(EC2D_y_boots, axis=1)
    # y50_EC_boots_idx  = np.argmax(EC2D_boots_median, axis=-1)
    # y50_EC_boots_ux   = np.array([EC2D_data_ux[:,i] for i in y50_EC_boots_idx]).T
    # y50_EC_boots_y    = np.max(EC2D_boots_median,axis=-1) 
    # y50_EC_boots      = np.concatenate((y50_EC_boots_ux, y50_EC_boots_y.reshape(1,-1)), axis=0)
    # y50_EC_boots_mean = np.mean(y50_EC_boots, axis=1)
    # y50_EC_boots_std  = np.std(y50_EC_boots, axis=1)
    # print(' > Extreme reponse from EC (Bootstrap (n={:d})):'.format(EC2D_y_boots.shape[0]))
    # print('   - {:<25s} : {}'.format('Bootstrap data set', EC2D_y_boots.shape))
    # print('   - {:<25s} : [{:.2f}, {:.2f}]'.format('y50[mean, std]',y50_EC_boots_mean[-1], y50_EC_boots_std[-1]))
    # print('   - {:<25s} : {}'.format('Design state (u,x)', y50_EC_boots_mean[:4]))

    # u_center = y50_EC_boots_mean[ :2].reshape(-1, 1)
    # x_center = y50_EC_boots_mean[2:4].reshape(-1, 1)
    # print(' > Important Region based on EC(boots):')
    # print('   - {:<25s} : {}'.format('Radius', radius_surrogate))
    # print('   - {:<25s} : {}'.format('Center U', np.squeeze(u_center)))
    # print('   - {:<25s} : {}'.format('Center X', np.squeeze(x_center)))
    # print('================================================================================')
