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
from solver.dynamic_models import lin_oscillator
from solver.dynamic_models import duffing_oscillator
from solver.static_models import ishigami
from solver.static_models import poly5
from utilities.gen_gauss_time_series import gen_gauss_time_series
import uqplot.plot_solver as psolver

import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import time

def main():
    ## Duffing oscillator problem 
    ## Ref: Dr. R. Ghanem
    ## -- source function S(w) = c/(c^2 + w^2), with c ~ LogNormal()
    ## -- 
    ## Exceedence probability
    pf = 1e-4
    ## ------------------------------------------------------------------- ###
    ##  Define Solver parameters ###
    ## ------------------------------------------------------------------- ###
    ## Choose Wiener-Askey scheme random varaible
    dist_zeta  = cp.Gamma()  # shape=1, scale=1, shift=0
    ## If transformation needed, like Rosenblatt, need to be done here
    ## Define independent random varible in physical problems
    dist_x = cp.Lognormal(0,0.5) # normal mean = 0, normal std=0.25
    # dist_x = [cp.Uniform(-1,1),cp.Uniform(-1,1)]cp.Uniform(-1,1)
    print(dist_x.inv(pf))
    print(dist_x.inv(1-pf))
    ## Define solver system properties
    sys_params = np.array([0,0,0.02,1,0.15]).reshape(5,1) #x0,v0, zeta, omega_n, mu

    # ------------------------------------------------------------------- ###
    #  Define simulation parameters  ###
    # ------------------------------------------------------------------- ###
    ## Parameters to generate solver system input signal
    source_kwargs= {'name': 'T1', 'method':'ifft', 'sides':'double'}
    source_func = gen_gauss_time_series 
    sys_source = [source_func, source_kwargs]

    ## Parameters to design of experiments
    doe_method, doe_rule, doe_order = 'GQ','lag',[9]*10
    # doe_method, doe_rule, doe_order = 'MC','R', [20]
    doe_params = [doe_method, doe_rule, doe_order]
    print(len(doe_params))

    ## Parameters to the time indexing
    time_start, time_ramp, time_max, dt = 0,0,200,0.1
    time_params =  [time_start, time_ramp, time_max, dt]
    
    ## parameters to post analysis
    qoi2analysis = [0,]
    stats = [1,1,1,1,1,1,0] # [mean, std, skewness, kurtosis, absmax, absmin, up_crossing, moving_avg, moving_std]
    post_params = [qoi2analysis, stats]
    normalize = True

    quad_simparam = simParameter(dist_zeta, doe_params = doe_params, \
            time_params = time_params, post_params = post_params,\
            sys_params = sys_params, sys_source = sys_source, normalize=normalize)

    quad_simparam.set_seed([1,100])
    ## Get DOE samples
    doe_samples_zeta, doe_samples_phy = quad_simparam.get_doe_samples(retphy=True,dist_phy=dist_x)

    print(doe_samples_zeta)
    print(doe_samples_phy)

    ## Run simulation
    # ### ------------------------------------------------------------------- ###
    # ### Run Simulations for training data  ###
    # ### ------------------------------------------------------------------- ###
    # print(doe_samples_phy[0][0][0,1])

    training_set_x = doe_samples_phy
    training_set_y = run_sim(duffing_oscillator, quad_simparam)
    training_set   = [training_set_x, training_set_y]

    # ## save training x
    # _training_set_x = training_set_x[0]
    # for i in np.arange(1,len(doe_order)):
        # _training_set_x=np.hstack((_training_set_x,training_set_x[i]))
    # np.savetxt('training_set_x.csv', np.array(_training_set_x).T, delimiter=',')

    # ## save training y
    # icount = 0
    # for itraining_set_y in training_set_y:
        # for i in np.arange(itraining_set_y.shape[0]):
            # for j in np.arange(itraining_set_y.shape[1]):
                # np.savetxt('training_set_y_{:d}.csv'.format(icount), np.squeeze(itraining_set_y[i,j,:,:]),delimiter=',')
                # icount+=1
                


    # _training_set_x=[]
    for idoeset in np.arange(len(doe_order)):
        itraining_set_x = training_set_x[idoeset]
        itraining_set_y = training_set_y[idoeset]

        np.save('training_set_x_{:d}'.format(idoeset), np.array(training_set_x[idoeset]))
        np.save('training_set_y_{:d}'.format(idoeset), np.array(training_set_y[idoeset]))

    # print(training_set_x[0].shape)
    # print(training_set_x[1].shape)
    # print(training_set_y[0].shape)
    # print(training_set_y[1].shape)


    # y = np.squeeze(training_set_y[0][0,0,:,:])
    # t = np.squeeze(training_set_y[0][0,0,:,0])
    # y_std = []
    # for i in range(1,len(y)):
        # y_std.append(np.std(y[:i]))
    # plt.figure()
    # plt.plot(t[1:],np.array(y_std))
    # plt.plot(t,y)
    # plt.show()
    # x0,v0, zeta, omega_n, mu = sys_params
    # x0,v0, zeta, omega_n, mu = x0[0],v0[0], zeta[0], omega_n[0], mu[0]

    # delta = 2 * zeta * omega_n
    # alpha = omega_n**2
    # beta  = omega_n**2 * mu
    # print('delta: {:.2f}, alpha: {:.2f}, beta: {:.2f}'.format(delta, alpha, beta)) 
    # psolver.duffing_equation(x0,v0,delta, alpha, beta,y)

    # print(np.array(training_set_y).shape)

    # f_obsY = []
    # for idoe_samples_phy in doe_samples_phy:
        # # training_set_y = run_sim(lin_oscillator, idoe_samples_phy, quad_simparam,\
                # # sys_params=sys_params, psd_params=psd_params)
        # training_set_y = run_sim(duffing_oscillator, idoe_samples_phy, quad_simparam,\
                # sys_params=sys_params, psd_params=psd_params)
        # # training_set_y = run_sim(poly5, idoe_samples_phy, quad_simparam)
        # f_obsY.append(training_set_y)
    # print(np.array(f_obsY).shape)
    # f_obsY_max = np.max(f_obsY, axis=-2)
    # print(f_obsY_max)

    
    # plt.figure()
    # plt.plot(training_set_y[0,:,0], training_set_y[0,:,1])
    # plt.show()


    #### ------------------------------------------------------------------- ###
    #### Define meta model parameters  ###
    #### ------------------------------------------------------------------- ###
    # fit_method = 'SP' # SP: spectral projection, RG: regression
    # quad_metamodel   = metaModel('PCE', [5], fit_method, dist_zeta)

    

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

