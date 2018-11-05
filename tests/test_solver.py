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
import envi, doe, solver, utilities

from envi import environment
from metaModel import metaModel
from simParams import simParameter

from run_sim1 import run_sim
from solver.dynamic_models import lin_oscillator
from solver.dynamic_models import duffing_oscillator
from solver.static_models import ishigami
from solver.static_models import poly5

import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import time

def main():
    ## ------------------------------------------------------------------- ###
    ##  Define Solver parameters ###
    ## ------------------------------------------------------------------- ###
    ## Choose Wiener-Askey scheme random varaible
    dist_zeta  = cp.Normal()
    ## If transformation needed, liek Rosenblatt, need to be done here
    ## Define independent random varible in physical problems
    dist_x = cp.Normal()
    ## Define solver system properties
    # sys_params = {'c': (1,1), 'x0':(0,0)} # zeta, omega_n
    sys_params = {'c': (1,1,1), 'x0':(0,0)} # zeta, omega_n, mu
    # sys_params = {'p': 2} # ishigami

    # dist_zeta   = [cp.Exponential(1), cp.Exponential(1)] 
    # qoi2analysis = [0] # for multioutput dof system, select the one need to analyze
    # ------------------------------------------------------------------- ###
    #  Define simulation parameters  ###
    # ------------------------------------------------------------------- ###
    doe_method = 'GQ'
    doe_order = [4]
    qoi2analysis = [0,]
    time_start, time_ramp, time_max, dt = 0,0,1000,0.1
    stats = [1,1,1,1,1,0]

    quad_simparam = simParameter(doe_method, doe_order, dist_zeta)
    # quad_simparam.setNumOutputs(3)
    quad_simparam.set_seed([1,100])
    ## Get DOE samples
    doe_samples_zeta, doe_samples_phy = quad_simparam.get_doe_samples(retphy=True,dist_phy=dist_x)

    ## Generate solver system input signal
    psd_params = {'name': 'T1', 'method':'ifft', 'sides':'2side'}
    sys_inputsim_method = 'ifft' ## corresponding to gen_gauss_time_series method,could be expaned later

    ## Run simulation
    # ### ------------------------------------------------------------------- ###
    # ### Run Simulations for training data  ###
    # ### ------------------------------------------------------------------- ###
    f_obsY = []
    for idoe_samples_phy in doe_samples_phy:
        # f_obsi = run_sim(lin_oscillator, idoe_samples_phy, quad_simparam,\
                # sys_params=sys_params, psd_params=psd_params)
        f_obsi = run_sim(duffing_oscillator, idoe_samples_phy, quad_simparam,\
                sys_params=sys_params, psd_params=psd_params)
        # f_obsi = run_sim(poly5, idoe_samples_phy, quad_simparam)
        f_obsY.append(f_obsi)
    print(np.array(f_obsY).shape)
    f_obsY_max = np.max(f_obsY, axis=-2)
    print(f_obsY_max)

    
    # plt.figure()
    # plt.plot(f_obsY[0][0,:,0], f_obsY[0][0,:,1:])
    # plt.show()


    #### ------------------------------------------------------------------- ###
    #### Define meta model parameters  ###
    #### ------------------------------------------------------------------- ###
    fit_method = 'SP' # SP: spectral projection, RG: regression
    quad_metamodel   = metaModel('PCE', [5], fit_method, dist_zeta)

    

    # ### ------------------------------------------------------------------- ###
    # ### Define environmental conditions ###
    # ### ------------------------------------------------------------------- ###
    # siteEnvi = environment(quad_simparam.site)


    # ### ------------------------------------------------------------------- ###
    # ### Fitting meta model  ###
    # ### ------------------------------------------------------------------- ###
    # for a in f_obsX:
        # print(a.shape)

    # for a in f_obsY:
        # print(a.shape)

    # # for a in f_obsYstats:
        # # print(a.shape)

    # # istats, iqoi = 5, 0 # absmax
    # datax = [x[:len(dist_zeta),:] for x in f_obsX] 
    # datay = f_obsY #[y[:,istats, iqoi] for y in f_obsY] 
    # # dataw = [x[len(dist_zeta),:] for x in f_obsX] 
    
    # quad_metamodel.fit_model(datax, datay)
    # print(quad_metamodel.f_hats[0][0])
    # # print(quad_metamodel.f_hats)
    # ## ------------------------------------------------------------------- ###
    # ## Cross Validation  ###
    # ## ------------------------------------------------------------------- ###
    # quad_metamodel.cross_validate(validatax, validatay)
    # print ('    Fitting Error:', quad_metamodel.fit_l2error)
    # print ('    Cross Validation Error:', quad_metamodel.cv_l2errors)

    # ### ------------------------------------------------------------------- ###
    # ### Prediction  ###
    # ### ------------------------------------------------------------------- ###
    # models_chosen = [[0,1],[1,0]] 
    # meta_pred = quad_metamodel.predict(1E2,R=10)




    # metamodel1 = quad_metamodel.f_hats[0][0]
    # datay1 = metamodel1(*datax[0])
    # print(datay[0].shape)
    # print(datay1.shape)

    # n_bins = 100 
    # fig, ax = plt.subplots(figsize=(8, 4))

# # plot the cumulative histogram
    # n, bins, patches = ax.hist(datay[0], n_bins, normed=1, histtype='step',
                               # cumulative=True, label='True value')

    # ax.hist(datay1.T, n_bins, normed=1, histtype='step',cumulative=True, label='Fitted value' )
# # tidy up the figure
    # ax.grid(True)
    # ax.legend(loc='right')
    # ax.set_title('Cumulative step histograms')
    # ax.set_xlabel('Annual rainfall (mm)')
    # ax.set_ylabel('Likelihood of occurrence')

    # plt.show()




if __name__ == '__main__':
    main()

