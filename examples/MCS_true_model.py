#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import context
import chaospy as cp
import numpy as np
# import scipy.signal as spsignal
# import envi, doe, solver, utilities
# import pickle

# from envi import environment
from metaModel import metaModel
from simParams import simParameter

from run_sim import run_sim
# from utilities.gen_gauss_time_series import gen_gauss_time_series
# import uqplot.plot_solver as psolver
import os
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

def make_working_directory(MODEL_NAME):
    WORKING_DIR     = os.getcwd()
    MODEL_DIR       = os.path.join(WORKING_DIR, MODEL_NAME)
    MODEL_DIR_DATA  = os.path.join(MODEL_DIR,r'Data')
    MODEL_DIR_FIGURE= os.path.join(MODEL_DIR,r'Figures')
    # Create directory for model  
    try:
        os.makedirs(MODEL_DIR)
        os.makedirs(MODEL_DIR_DATA)
        os.makedirs(MODEL_DIR_FIGURE)
    except FileExistsError:
        # one of the above directories already exists
        pass
    return MODEL_DIR, MODEL_DIR_DATA, MODEL_DIR_FIGURE 


def main():

    ## ------------------------------------------------------------------- ###
    ##  Parameters set-up 
    ## ------------------------------------------------------------------- ###
    pf              = 1e-4              # Exceedence probability
    samples_train   = [[1e6], 'R']      # nsamples_test, sample_rule
    samples_test    = [1e6, 10, 'R']    # nsamples_test, nrepeat, sample_rule
    model_def       = ['bench4'] #[solver function name, short-term distribution name, [params], size]
    MODEL_DIR, MODEL_DIR_DATA, MODEL_DIR_FIGURE =  make_working_directory(model_def[0])
    ## ------------------------------------------------------------------- ###
    ##  Define Solver parameters ###
    ## ------------------------------------------------------------------- ###
    ## >>> Choose Wiener-Askey scheme random variable
    # dist_zeta  = cp.Normal()  # shape=1, scale=1, shift=0
    dist_zeta = cp.Normal(0,1)  # shape=1, scale=1, shift=0

    ## >>> If transformation needed, like Rosenblatt, need to be done here

    ## >>> Define independent random variable in physical problems
    dist_x = cp.Normal(5,2) # normal mean = 0, normal std=0.25
    # dist_x = cp.Normal(2,3) # normal mean = 0, normal std=0.25


    # ## ------------------------------------------------------------------- ###
    # ##  Design of Experiments (DoEs) 
    # ## ------------------------------------------------------------------- ###
    # ## example:
    # # Fixed point:
    # #  sample_points_x = dist_zeta.inv(dist_x.cdf(np.arange(1,25)))
    # #  doe_method,doe_rule,doe_order='FIX',sample_points_x,[len(sample_points_x)]*10
    # # Monte Carlo:
    # #  doe_method, doe_rule, doe_order = 'MC','R', [20]
    np.random.seed(100)

    doe_method, doe_rule, doe_order = 'MC','R', [1e7]*10
    doe_params  = [doe_method, doe_rule, doe_order]
    mc_simparam = simParameter(dist_zeta, doe_params = doe_params)
    # Get DoEs samples both in zeta space and physical space
    mc_simparam.get_doe_samples(dist_x)

    ## ------------------------------------------------------------------- ###
    ##  Run simulation 
    ## ------------------------------------------------------------------- ###
    sim_output_mcs = run_sim(model_def, mc_simparam)

    print(r'------------------------------------------------------------')
    print(r'►►► Save Data ')
    print(r'------------------------------------------------------------')
    print(r' ► Saving simulation data ...')

    sim_output_mcs_path = '_'.join(['True_MCS',mc_simparam.doe_rule, 'DoE'])

    for i in np.arange(mc_simparam.ndoe):
        print('   ♦ {:<15s} : {:d}'.format('DoE set', i))
        idoe_res =[ np.array(mc_simparam.sys_input_vars[i]),
                    np.array(sim_output_mcs[0][i][:]),
                    np.array(mc_simparam.sys_input_zeta[i])]

        np.save(os.path.join(MODEL_DIR_DATA, sim_output_mcs_path+ r'{:d}'.format(i)), idoe_res)
        # plt.figure()
        # plt.plot(mc_simparam.sys_input_vars[i],sim_output_mcs[0][i][:],'k.',markersize=1)
        # plt.savefig(os.path.join(MODEL_DIR_DATA, sim_output_mcs_path+r'{:d}.png'.format(i)))



    ## ------------------------------------------------------------------- ###
    ##  Design of Experiments (DoEs) 
    ## ------------------------------------------------------------------- ###
    ## example:
    # Fixed point:
    #  sample_points_x = dist_zeta.inv(dist_x.cdf(np.arange(1,25)))
    #  doe_method,doe_rule,doe_order='FIX',sample_points_x,[len(sample_points_x)]*10
    # Monte Carlo:
    #  doe_method, doe_rule, doe_order = 'MC','R', [20]

    doe_method, doe_rule, doe_order = 'FIX','FIX', 1e3
    doe_params  = [doe_method, doe_rule, doe_order]
    fix_simparam= simParameter(dist_zeta, doe_params=doe_params)
    x_samples   = [np.linspace(-5,15,1000)[np.newaxis, :]]
    
    # Get DoEs samples both in zeta space and physical space
    fix_simparam.set_doe_samples(x_samples, dist_x)

    ## ------------------------------------------------------------------- ###
    ##  Run simulation 
    ## ------------------------------------------------------------------- ###
    sim_output_mcs = run_sim(model_def, fix_simparam)

    print(r'------------------------------------------------------------')
    print(r'►►► Save Data ')
    print(r'------------------------------------------------------------')
    print(r' ► Saving simulation data ...')

    sim_output_mcs_path = '_'.join(['True_domain',fix_simparam.doe_rule, 'DoE_fix'])

    for i in np.arange(fix_simparam.ndoe):
        print('   ♦ {:<15s} : {:d}'.format('DoE set', i))
        idoe_res =[ np.array(fix_simparam.sys_input_vars[i]),
                    np.array(sim_output_mcs[0][i][:]),
                    np.array(fix_simparam.sys_input_zeta[i])]

        # np.save(os.path.join(MODEL_DIR_DATA, sim_output_mcs_path+ r'{:d}'.format(i)), idoe_res)
        # plt.figure()
        # plt.plot(mc_simparam.sys_input_vars[i],sim_output_mcs[0][i][:],'k.',markersize=1)
        # plt.savefig(os.path.join(MODEL_DIR_DATA, sim_output_mcs_path+r'{:d}.png'.format(i)))


if __name__ == '__main__':
    main()

        
