#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import context
import chaospy as cp
import numpy as np
import scipy.signal as spsignal
import envi, doe, solver, utilities

from envi import environment
from metaModel import metaModel
from simParams import simParameter

from run_sim import run_sim
# from solver.dynamic_models import lin_oscillator
# from solver.dynamic_models import duffing_oscillator
# from solver.static_models import ishigami
# from solver.static_models import poly5
# from utilities.gen_gauss_time_series import gen_gauss_time_series
# import uqplot.plot_solver as psolver

import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import time
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

def main():
    
    ## ------------------------------------------------------------------- ###
    ##  Parameters set-up 
    ## ------------------------------------------------------------------- ###
    pf = 1e-4 ## Exceedence probability
    samples_train   = [[1e6], 'R']  # nsamples_test, sample_rule
    samples_test    = [1e6, 10, 'R'] # nsamples_test, nrepeat, sample_rule
    meta_orders     = np.arange(1,13)

    ## ------------------------------------------------------------------- ###
    ##  Define Solver parameters ###
    ## ------------------------------------------------------------------- ###
    ## Choose Wiener-Askey scheme random varaible
    # dist_zeta  = cp.Normal()  # shape=1, scale=1, shift=0
    dist_zeta  = cp.Uniform(-1,1)  # shape=1, scale=1, shift=0
    ## If transformation needed, like Rosenblatt, need to be done here
    ## Define independent random varible in physical problems
    dist_x = cp.Uniform(2,3) # normal mean = 0, normal std=0.25
    # dist_x = cp.Normal(2,3) # normal mean = 0, normal std=0.25

    ## ------------------------------------------------------------------- ###
    ##  Get samples for inputs and outputs 
    ## ------------------------------------------------------------------- ###

    doe_method, doe_rule, doe_order = 'GQ','legendre',[150]
    # sample_points_x = dist_zeta.inv(dist_x.cdf(np.arange(1,25)))
    # doe_method, doe_rule, doe_order = 'FIX',sample_points_x  , [len(sample_points_x )]*10
    # doe_method, doe_rule, doe_order = 'MC','R', [20]
    doe_params = [doe_method, doe_rule, doe_order]
    quad_simparam = simParameter(dist_zeta, doe_params = doe_params)
    doe_samples_zeta, doe_samples_phy = quad_simparam.get_doe_samples(retphy=True,dist_phy=dist_x)
    # print(len(doe_samples_phy))
    # print(np.array(doe_samples_zeta))

    training_x = []
    training_y = []
    weight_xy  = []
    for i in np.arange(len(doe_order)):
        coord_zeta, weight_zeta = doe_samples_zeta[i]
        coord_phy,  weight_phy  = doe_samples_phy[i]
        # print(coord_zeta.shape)
        # print(coord_phy.shape)
        # print(weight_phy.shape)
        training_x.append(coord_zeta)
        training_y.append(coord_phy.T)
        weight_xy.append(weight_zeta)

    # training_x = [dist_zeta.sample(isample) for isample in samples_train[0]]
    # training_y = []
    # for x in training_x:
        # training_y.append(dist_x.inv(dist_zeta.cdf(x)))
    # print(training_x)
    # print(weight_xy)
    # print(training_y)
    pce_model = metaModel('PCE', meta_orders, 'GQ',dist_zeta)
    # training_x = training_x.reshape(1,-1)
    # training_y = training_y.reshape(-1,1)
    # print(training_x.shape)
    pce_model.fit_model(training_x,training_y,weight_xy)
    # pce_model.predict()
    metrics2cal = [0,1,1,1,0,0] #'value','moments','norms','upper fractile','ECDF','pdf'
    pce_model.predict(samples_test, retmetrics=metrics2cal)
    # print(pce_model.f_orth_coeffs[0][0].shape)
    # print(pce_model.orthPoly[0])
    # metric_moms = np.array(pce_model.metrics[1])
    # # print(pce_model.f_orth_coeffs)
    # # print(np.array(metric_moms).shape)
    # # metric_moms = np.array([pce_model.metrics[i][j][1] for i in np.arange(len(samples_train[0])*len(meta_orders) ) for j in np.arange(samples_test[1])])
    # # print(metric_moms.shape)
    print(pce_model.metric_names)
    for i, imetric in enumerate(pce_model.metrics):
        print(np.array(imetric).shape)
        np.save('Uniform_Legendre'+pce_model.metric_names[i], imetric)
        # np.save('Uniform_Hermite{}'.format(i), imetric)
        # # print()


    # plt.plot(meta_orders, np.squeeze(metric_moms[0,:,0,:]))
    # plt.show()


if __name__ == '__main__':
    main()

