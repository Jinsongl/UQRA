# /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
# import context
import os, sys, chaospy as cp, numpy as np

from ..surrogate_model.SurrogateModel import SurrogateModel 
from ..simParameters import simParameters
from ..utilities import upload2gdrive, get_exceedance_data,make_output_dir, get_gdrive_folder_id 

def run_doe(sim_parameters):

    # ## ------------------------------------------------------------------- ###
    # ##  Design of Experiments (DoEs) 
    # ## ------------------------------------------------------------------- ###
    # ## example:
    
    # ## >>> 1. Fixed point:
    # fix_simparam= sim_parameters
    # doe_method, doe_rule, doe_order = 'FIX','FIX', [10, 11,12,13,14,15]
    # doe_params  = [doe_method, doe_rule, doe_order]
    # rng         = np.random.RandomState(3)
    # x_samples   = []
    # for idoe in doe_order:
        # x_samples.append([rng.uniform(-5,15, idoe)[np.newaxis, :]])
    # # x_samples   = [np.linspace(-5,14,10)[np.newaxis, :]]
    # fix_simparam.set_doe_samples(x_samples)
    # # zeta_samples= fix_simparam.sys_input_zeta[0]

    ##>>> 2. Monte Carlo:
    # mc_simparam = sim_parameters
    # np.random.seed(100)
    # doe_method, doe_rule, doe_order = 'MC','R', [int(1e1), int(1e2)]
    # doe_params  = [doe_method, doe_rule, doe_order]
    # mc_simparam.set_doe_method(doe_params)
    # mc_simparam.get_doe_samples()
    # # mc_simparam.get_input_vars(dist_x)
    # sim_parameters    = mc_simparam

    # ## >>> 3. Quadrature:
    # quad_simparam = sim_parameters
    # doe_method, doe_rule, doe_order = 'QUAD','hem',[10,11,12,13,14,15]
    # doe_params      = [doe_method, doe_rule, doe_order]
    # quad_simparam.set_doe_method(doe_params)
    # quad_simparam.get_doe_samples()
    # sim_parameters    = quad_simparam

    ## ------------------------------------------------------------------- ###
    ##  Run simulation 
    ## ------------------------------------------------------------------- ###
    ## model_name, error_type_name, error_type_params, error_type_size = model_def
    ## run_sim output [isys_def_params, idoe, [solver output]]
    ## [solver function name, error_dist.name, [params], size]
    # sim_parameters    = fix_simparam
    # doe_type    = 'Uniform_DoE'
    # doe_type    = 'Linspace_DoE'        
    # doe_type    = 'MCS'+'{:.0E}'.format(doe_order[0])[-1] +'_DoE'
    # doe_type    = 'Quadrature_DoE'

    ## ------------------------------------------------------------------- ###
    ##  Noise Free case 
    ## ------------------------------------------------------------------- ###

    sim_output  = run_sim(sim_parameters)
    # print(usim_output)
    # fname_sim_out = 

    for idoe in np.arange(sim_parameters.ndoe):
        fname_sim_out_idoe = '_'.join([sim_parameters.doe_filenames[idoe], sim_parameters.error.name,'train'])
        print(r'   * {:<15s} : {:d} / {:d}'.format('DoE set', idoe, sim_parameters.ndoe))
        x   = np.squeeze(np.array(sim_parameters.sys_input_x[idoe]))
        y   = np.squeeze(np.array(sim_output[idoe]))
        zeta= np.squeeze(np.array(sim_parameters.sys_input_zeta[idoe]))
        idoe_res =[x,y,zeta]
        # error_params.append([0, 0.2*abs(y)])

        if doe_method.upper() == 'MC':
            #### Upload data to google drive
            upload2gdrive(os.path.join(sim_parameters.data_dir,fname_sim_out_idoe), idoe_res, sim_parameters.data_dir_id)   
            ### Calculate Exceedance (ONLY for MCS sampling)         
            print(r' > Calculating ECDF of MCS data and retrieve data to plot...')
            y_mcs_ecdf = get_exceedance_data(y, prob=sim_parameters.prob_fails)
            fname_sim_out_idoe_ecdf = fname_sim_out_idoe+ '_pf' + '{:.0E}'.format(sim_parameters.prob_fails)[-1] + '_ecdf'
            np.save(os.path.join(sim_parameters.data_dir, fname_sim_out_idoe_ecdf+'.npy'), y_mcs_ecdf)

        else:
            # print(uos.path.join(sim_parameters.data_dir, fname_sim_out_idoe))         
            np.save(os.path.join(sim_parameters.data_dir, fname_sim_out_idoe), idoe_res)

    # ## ------------------------------------------------------------------- ###
    # ##  Add noise case 
    # ## ------------------------------------------------------------------- ###
    # model_def  = [MODEL_NAME, 'normal', error_params]        
    # sim_output = run_sim(model_def, sim_parameters)
    # noise_type  = 'noise_free' if model_def[1] is None else model_def[1]
    # fname_sim_out = '_'.join(['Train', noise_type, doe_type])

    # for idoe in np.arange(sim_parameters.ndoe):

        # print(u'   * {:<15s} : {:d} / {:d}'.format('DoE set', idoe, sim_parameters.ndoe))
        # x   = np.squeeze(np.array(sim_parameters.sys_input_vars[idoe]))
        # y   = np.squeeze(np.array(sim_output[0][idoe]))
        # zeta= np.squeeze(np.array(sim_parameters.sys_input_zeta[idoe]))
        # idoe_std = np.squeeze(error_params[idoe][1])
        # idoe_mean= idoe_std * 0
        # idoe_res =[x,y,zeta,idoe_mean, idoe_std]

        # if doe_method.upper() == 'MC':
            # fname_sim_out_idoe = fname_sim_out+ r'{:d}'.format(idoe) 
            # #### Upload data to google drive
            # upload2gdrive(os.path.join(sim_parameters.data_dir,fname_sim_out_idoe), idoe_res, DATA_DIR_ID)
            # #### Calculate Exceedance (ONLY for MCS sampling)
            # print(u' > Calculating ECDF of MCS data and retrieve data to plot...')
            # y_mcs_ecdf = get_exceedance_data(y,prob_failure=pf)
            # fname_sim_out_idoe_ecdf = fname_sim_out_idoe+ '_ecdf'
            # np.save(os.path.join(sim_parameters.data_dir, fname_sim_out_idoe_ecdf+'.npy'), y_mcs_ecdf)
        # else:
            # fname_sim_out_idoe = fname_sim_out+ r'{:d}'.format(doe_order[idoe]) 
            # np.save(os.path.join(sim_parameters.data_dir, fname_sim_out_idoe), idoe_res)



if __name__ == '__main__':
    run_doe(sim_parameters)

      
