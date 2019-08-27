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


def run_sim(model_name, simParams):
    """
    Run simulation with given "solver" (real model) and predefined parameters 
        General format of a solver
            y =  M(x, sys_def_params, sys_input_params)
            M: system solver, taking following arguments
            x: system inputs, could be one of the following three formats
                1. array of x samples (ndim, nsamples), y = M(x)
                2. ndarray of time series
                3. inputs for the sys_input_func_name to generate time series
            (optional:)
            sys_def_params: array of shape (m,n), parameters defining system solver
                - m: number of set, 
                - n: number of system parameters per set
            sys_input_params: array of shape(m,n), parameters defining system input signal

    Arguments:
        model_name: string, solver function name
        simParams: simParameter class object

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
    print('   ♦ {:<17s} : {:15s}'.format('Solver name', model_name))

    if simParams.sys_def_params is None or simParams.sys_def_params[0] is None:
        print('   ♦ System definition parameters: NA ' )
    else:
        print('   ♦ System definition parameters: ndim={:d}, nsets={:d}'\
                .format(simParams.sys_def_params.shape[1], simParams.sys_def_params.shape[0]))
                # .format(simParams.sys_def_params.shape[0], len(simParams.sys_def_params)))
    print('   ♦ System input parameters:')
    print('     ∙ {:<15s} : {}'.format('function',simParams.sys_input_params[0]))
    print('     ∙ {:<15s} : {}'.format('kwargs', simParams.sys_input_params[1]))

    ###------------- Time domain step ----------------------------
    if simParams.time_max:
        print('     ∙ {:<15s} : {:s}, Tmax={:.1e}, dt={:.2f}'\
            .format('time steps', simParams.time_max, simParams.dt))
    else:
        print('     ∙ {:<15s} : NA'.format('time steps'))
    # print('     ∙ {:<15s} : ndoe={:d}, ndim={:d}, nsets={}'\
            # .format('input variables', simParams.ndoe, simParams.ndim_sys_inputs, simParams.doe_order))

    ###------------- Error properties ----------------------------
    simParams.error.disp()

    print(' ► Running Simulation...')
    print('   ♦ Job list: [{:^20} {:^20}]'.format('# Sys params sets', '# DoE sets'))
    print('   ♦ Target  : [{:^20d} {:^20d}]'.format(len(simParams.sys_def_params), simParams.ndoe))
    print('   ' + '·'*55)


    for isys_done, isys_def_params in enumerate(simParams.sys_def_params): 
        run_sim_1doe_res = []
        for idoe, sys_input_vars_1doe in enumerate(simParams.sys_input_x):
            if simParams.doe_method.upper() == 'QUADRATURE':
                input_vars, input_vars_weights = sys_input_vars_1doe
            else:
                input_vars, input_vars_weights = sys_input_vars_1doe, None
            ### Run simulations
            y = solver_wrapper(model_name,input_vars,\
                    error_type      = simParams.error,\
                    sys_input_params= simParams.sys_input_params,\
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




 
