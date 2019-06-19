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

# from envi import environment
from metaModel import metaModel
from simParams import simParameter

from run_sim import run_sim
from utilities.get_exceedance_data import get_exceedance_data
from utilities.upload2gdrive import upload2gdrive
import os,sys
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


def get_gdrive_folder_id(folder_name):
    """
    Check if the given folder_name exists in Google Drive. 
    If not, create one and return the google drive ID
    Else: return folder ID directly
    """
    command = os.path.join('/Users/jinsongliu/Google Drive File Stream/My Drive/MUSE_UQ_DATA', folder_name)
    try:
        os.makedirs(command)
    except FileExistsError:
        pass
    command = 'gdrive list --order folder |grep ' +  folder_name
    folder_id = os.popen(command).read()
    return folder_id[:33]

def make_output_dir(MODEL_NAME):
    """
    WORKING_DIR/
    +-- MODEL_DIR
    |   +-- FIGURE_DIR

    /directory saving data depends on OS/
    +-- MODEL_DIR
    |   +-- DATA_DIR

    """
    WORKING_DIR     = os.getcwd()
    MODEL_DIR       = os.path.join(WORKING_DIR, MODEL_NAME)
    FIGURE_DIR= os.path.join(MODEL_DIR,r'Figures')
    # DATA_DIR  = os.path.join(MODEL_DIR,r'Data')
    current_os  = sys.platform
    if current_os.upper()[:3] == 'WIN':
        DATA_DIR= "G:\My Drive\MUSE_UQ_DATA"
    elif current_os.upper() == 'DARWIN':
        DATA_DIR= '/Users/jinsongliu/External/MUSE_UQ_DATA'
    elif current_os.upper() == 'LINUX':
        DATA_DIR= '/home/jinsong/Box/MUSE_UQ_DATA'
    else:
        raise ValueError('Operating system {} not found'.format(current_os))    
    
    DATA_DIR  = os.path.join(DATA_DIR,MODEL_NAME,r'Data')
    # MODEL_DIR_DATA_ID = GDRIVE_DIR_ID[MODEL_NAME.upper()] 
    MODEL_DIR_DATA_ID = get_gdrive_folder_id(MODEL_NAME)


    # Create directory for model  
    print('------------------------------------------------------------')
    print('►►► Making directories for model {}'.format(MODEL_NAME))
    print('------------------------------------------------------------')
    try:
        os.makedirs(MODEL_DIR)
        os.makedirs(DATA_DIR)
        os.makedirs(FIGURE_DIR)
    except FileExistsError:
        # one of the above directories already exists
        pass
    print('WORKING_DIR: {}'.format(WORKING_DIR))
    print('+-- MODEL: {}'.format(MODEL_DIR))
    print('|   +-- {:<6s}: {}'.format('FIGURE',FIGURE_DIR))
    print('|   +-- {:<6s}: {}'.format('DATA',DATA_DIR))
    return MODEL_DIR_DATA_ID, DATA_DIR, FIGURE_DIR


def main():

    ## ------------------------------------------------------------------- ###
    ##  Parameters set-up 
    ## ------------------------------------------------------------------- ###
    prob_failures       = [1e-3, 1e-4, 1e-5, 1e-6] # Exceedence probability
    data_train_params   = [[1e6], 'R']      # nsamples_test, sample_rule
    data_test_params    = [1e7, 10, 'R']    # nsamples_test, nrepeat, sample_rule
    MODEL_NAME          = 'BENCH1'
    # MODEL_NAME          = 'Ishigami'
    DATA_DIR_ID, DATA_DIR, _ =  make_output_dir(MODEL_NAME)
    ## ------------------------------------------------------------------- ###
    ##  Define Solver parameters ###
    ## ------------------------------------------------------------------- ###
    ## >>> 1. Choose Wiener-Askey scheme random variable
    ##            # |   zeta    | Wiener-Askey chaos | support
    ## # ==============================================================
    ## # Continuous | Gaussian  | Hermite-chaos      |  (-inf, inf)
    ##              | Gamma     | Laguerre-chaos     |  [0, inf ) 
    ##              | Beta      | Jacobi-chaos       |  [a,b] 
    ##              | Uniform   | Legendre-chaos     |  [a,b] 
    ## # --------------------------------------------------------------
    ## # Discrete   | Poisson   | 
    ##              | Binomial  | 
    ##              | - Binomial| 
    ##              | hypergeometric
    ## 
    dist_zeta = cp.Normal(0,1)  # shape=1, scale=1, shift=0
    # dist_zeta = cp.Uniform(0,1)
    # dist_zeta = cp.Iid(dist_zeta,3) 

    ## >>> 2. If transformation needed, like Rosenblatt, need to be done here
    ## Perform Rosenblatt etc

    ## >>> 3. Define independent random variable in physical problems
    dist_x = cp.Normal(5,2) # normal mean = 0, normal std=0.25
    # dist_x = cp.Uniform(-np.pi, np.pi)
    # dist_x = cp.Iid(dist_x,3) 

    # ## ------------------------------------------------------------------- ###
    # ##  Design of Experiments (DoEs) 
    # ## ------------------------------------------------------------------- ###
    # ## example:

    ## >>> 1. Fixed point:
    # doe_method, doe_rule, doe_order = 'FIX','FIX', [1000]
    # doe_params  = [doe_method, doe_rule, doe_order]
    # fix_simparam= simParameter(dist_zeta, doe_params=doe_params)
    # rng         = np.random.RandomState(3)
    # x_samples   = []
    # for idoe in doe_order:
        # # x_samples.append([rng.uniform(-5,15, idoe)[np.newaxis, :]])
        # x_samples.append([np.linspace(-5,15, idoe)[np.newaxis, :]])
    # fix_simparam.set_doe_samples(x_samples, dist_x)
    # # zeta_samples= fix_simparam.sys_input_zeta[0]

    ##>>> 2. Monte Carlo:
    # np.random.seed(100)
    # doe_method, doe_rule, doe_order = 'MC','R', [int(1e7)]*10
    # doe_params  = [doe_method, doe_rule, doe_order]
    # mc_simparam = simParameter(dist_zeta, doe_params = doe_params)
    # mc_simparam.get_doe_samples(dist_x)

    ## >>> 3. Quadrature:
    doe_method, doe_rule, doe_order = 'GQ','hermite',[10,11,12,13,14,15]
    doe_params      = [doe_method, doe_rule, doe_order]
    ### ishigami_consts = [np.array([7,0.1]).reshape(2,1)]
    # quad_simparam   = simParameter(dist_zeta, doe_params = doe_params, sys_def_params=ishigami_consts)
    quad_simparam   = simParameter(dist_zeta, doe_params = doe_params)
    quad_simparam.get_doe_samples(dist_x)

    # ## ------------------------------------------------------------------- ###
    # ##  Create simulation parameter 
    # ## ------------------------------------------------------------------- ###


    # simparam    = fix_simparam
    # doe_type    = 'Uniform_DoE'
    # doe_type    = 'DoE_Linspace'        
    # simparam    = mc_simparam
    # doe_type    = 'DoE_MCS'+'{:.0E}'.format(doe_order[0])[-1] 
    simparam    = quad_simparam
    doe_type    = 'DoE_Quadrature'

    ## ------------------------------------------------------------------- ###
    ##  Noise Free case 
    ## ------------------------------------------------------------------- ###

    ## ------------------------------------------------------------------- ###
    ## >>> Define model parameters and simulation parameters
    ## run_sim(model_def, simparam)
    ##  - model_def: [solver function name, 
    ##              short-term distribution/error name, 
    ##              [error params], error size]
    ##  - simparam: simParameter class object
    ## run_sim output [isys_def_params, idoe, [solver output]]
    ##  - output = [sys_params0,sys_params1,... ]
    ##  - sys_param0 = [sim_res_DoE0, sim_res_DoE1,sim_res_DoE2,... ]
    ## ------------------------------------------------------------------- ###
    model_def   = [MODEL_NAME]        
    sim_output  = run_sim(model_def, simparam)

    ## Define output file names
    error_params= [] 
    noise_type  = '' if model_def[1] is None else model_def[1]
    fname_sim_out = '_'.join(['TrainData', noise_type, doe_type])

    ## If DoE method is MCS, calculate ECDF of samples
    y_mcs_ecdf = [[] for _ in range(len(prob_failures))]
    for idoe in np.arange(simparam.ndoe):
        print('   ♦ {:<15s} : {:d} / {:d}'.format('DoE set', idoe, simparam.ndoe))
        x   = np.squeeze(np.array(simparam.sys_input_vars[idoe]))
        y   = np.squeeze(np.array(sim_output[idoe]))
        zeta= np.squeeze(np.array(simparam.sys_input_zeta[idoe]))
        idoe_res =[x,y,zeta]
        # error_params.append([0, 0.2*abs(y)])

        if doe_method.upper() == 'MC':
            fname_sim_out_idoe = fname_sim_out +r'_R{:d}'.format(idoe) 
            #### Upload data to google drive
            upload2gdrive(os.path.join(DATA_DIR,fname_sim_out_idoe), idoe_res, DATA_DIR_ID)   
            for ipf, pf in enumerate(prob_failures):
                ### Calculate Exceedance (ONLY for MCS sampling)         
                print(' ► Calculating ECDF of MCS data and retrieve data to plot...')
                y_mcs_ecdf[ipf].append(get_exceedance_data(y, prob_failure=pf))
        else:
            fname_sim_out_idoe = fname_sim_out +r'{:d}'.format(doe_order[idoe])
            print(os.path.join(DATA_DIR, fname_sim_out_idoe))         
            np.save(os.path.join(DATA_DIR, fname_sim_out_idoe), idoe_res)

    ### Save Exceedence values for MCS
    if y_mcs_ecdf[0]:
        for ipf, pf in enumerate(prob_failures):
            fname_ipf_ecdf = '_'.join(['Ecdf_pf'+'{:.0E}'.format(pf)[-1], 'MCS'+'{:.0E}'.format(doe_order[0])[-1]])  
            np.save(os.path.join(DATA_DIR, fname_ipf_ecdf), y_mcs_ecdf[ipf]) 
    # ## ------------------------------------------------------------------- ###
    # ##  Add noise case 
    # ## ------------------------------------------------------------------- ###
    # model_def  = [MODEL_NAME, 'normal', error_params]        
    # sim_output = run_sim(model_def, simparam)
    # noise_type  = 'noise_free' if model_def[1] is None else model_def[1]
    # fname_sim_out = '_'.join(['Train', noise_type, doe_type])

    # for idoe in np.arange(simparam.ndoe):

        # print('   ♦ {:<15s} : {:d} / {:d}'.format('DoE set', idoe, simparam.ndoe))
        # x   = np.squeeze(np.array(simparam.sys_input_vars[idoe]))
        # y   = np.squeeze(np.array(sim_output[0][idoe]))
        # zeta= np.squeeze(np.array(simparam.sys_input_zeta[idoe]))
        # idoe_std = np.squeeze(error_params[idoe][1])
        # idoe_mean= idoe_std * 0
        # idoe_res =[x,y,zeta,idoe_mean, idoe_std]

        # if doe_method.upper() == 'MC':
            # fname_sim_out_idoe = fname_sim_out+ r'{:d}'.format(idoe) 
            # #### Upload data to google drive
            upload2gdrive(os.path.join(DATA_DIR,fname_sim_out_idoe), idoe_res, DATA_DIR_ID)
            # #### Calculate Exceedance (ONLY for MCS sampling)
            # print(' ► Calculating ECDF of MCS data and retrieve data to plot...')
            # y_mcs_ecdf = get_exceedance_data(y,prob_failure=pf)
            # fname_sim_out_idoe_ecdf = fname_sim_out_idoe+ '_ecdf'
            # np.save(os.path.join(DATA_DIR, fname_sim_out_idoe_ecdf+'.npy'), y_mcs_ecdf)
        # else:
            # fname_sim_out_idoe = fname_sim_out+ r'{:d}'.format(doe_order[idoe]) 
            # np.save(os.path.join(DATA_DIR, fname_sim_out_idoe), idoe_res)



if __name__ == '__main__':
    main()

       
