#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import os
import numpy as np
from solver.solver_wrapper import solver_wrapper


def run_sim(simParameters):
    """
    Run simulation with given "solver" (real model) and predefined parameters 
        General format of a solver
            M(x,y, w) = f(theta)
            M: system solver
            x: system inputs, could be one of the following three formats
                1. array of x samples (ndim, nsamples), y = M(x)
                2. ndarray of time series
                3. inputs for the sys_excit_func_name to generate time series
            y: system output, 
            w: sys_def_params
            theta: sys_excit_params
            (optional:)
            sys_def_params: array of shape (m,n), parameters defining system solver
                - m: number of set, 
                - n: number of system parameters per set
            sys_excit_params: array of shape(m,n), parameters defining system input signal

    Arguments:
        simParameters: simParameters class object

    Return:
        List of simulation results (Different doe_order, sample size would change, so return list)
        if more than one set of system params: 
          - output      = [sys_params0,  sys_params1,  ... ]
          -- sys_param0 = [sim_res_DoE0, sim_res_DoE1, sim_res_DoE2,... ]
        1. len(res) = len(sys_def_params) 
        2. For each element in res: simulation results from 1 DoE set
        3. Return from solver:
            a). [*, ndofs]
                each column represents a time series for one dof
            b). 

    """
    run_sim_res = [] # a list saving simulation results
    print('------------------------------------------------------------')
    print('►►► Run Real Model')
    print('------------------------------------------------------------')
    ###-------------Systerm input params ----------------------------
    ### sys_def_params is of shape (m,n),
    ###     m: number of set, 
    ###     n: number of system parameters per set
    print(' ► Solver (system) properties:')
    print('   ♦ {:<17s} : {:15s}'.format('Solver name', simParameters.model_name))

    if simParameters.sys_def_params is None or simParameters.sys_def_params[0] is None:
        print('   ♦ System definition parameters: NA ' )
    else:
        print('   ♦ System definition parameters: ndim={:d}, nsets={:d}'\
                .format(simParameters.sys_def_params.shape[1], simParameters.sys_def_params.shape[0]))
                # .format(simParameters.sys_def_params.shape[0], len(simParameters.sys_def_params)))
    print('   ♦ System input parameters:')
    print('     ∙ {:<15s} : {}'.format('function',simParameters.sys_excit_params[0]))
    print('     ∙ {:<15s} : {}'.format('kwargs', simParameters.sys_excit_params[1]))

    ###------------- Time domain step ----------------------------
    if simParameters.time_max:
        print('     ∙ {:<15s} : {:s}, Tmax={:.1e}, dt={:.2f}'\
            .format('time steps', simParameters.time_max, simParameters.dt))
    else:
        print('     ∙ {:<15s} : NA'.format('time steps'))
    # print('     ∙ {:<15s} : ndoe={:d}, ndim={:d}, nsets={}'\
            # .format('input variables', simParameters.ndoe, simParameters.ndim_sys_inputs, simParameters.doe_order))

    ###------------- Error properties ----------------------------
    simParameters.error.disp()

    print(' ► Running Simulation...')
    print('   ♦ Job list: [{:^20} {:^20}]'.format('# Sys params sets', '# DoE sets'))
    print('   ♦ Target  : [{:^20d} {:^20d}]'.format(len(simParameters.sys_def_params), simParameters.ndoe))
    print('   ' + '·'*55)


    for isys_done, isys_def_params in enumerate(simParameters.sys_def_params): 
        run_sim_1doe_res = []
        for idoe, sys_input_vars_1doe in enumerate(simParameters.sys_input_x):
            if simParameters.doe_method.upper() == 'QUADRATURE':
                input_vars, input_vars_weights = sys_input_vars_1doe
            else:
                input_vars, input_vars_weights = sys_input_vars_1doe, None
            ### Run simulations
            y = solver_wrapper(simParameters.model_name,input_vars,\
                    error_type      = simParameters.error,\
                    sys_excit_params= simParameters.sys_excit_params,\
                    sys_def_params  = isys_def_params)

            print('   ♦ Achieved: [{:^20d} {:^20d}]'.format(isys_done+1,idoe+1))
            run_sim_1doe_res.append(y)
        run_sim_res.append(run_sim_1doe_res)
    ## 
    if len(run_sim_res) ==1:
        run_sim_res = run_sim_res[0]
    if len(run_sim_res) ==1:
        run_sim_res = run_sim_res[0]
    print(' ► Simulation Done, Output shape: {}'.format(np.array(run_sim_res).shape) )
    return run_sim_res




 
