#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import os, numpy as np
from .solver.solver_wrapper import solver_wrapper


def run_solver(sim_parameters):
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
        sim_parameters: sim_parameters class object

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
    print(r'------------------------------------------------------------')
    print(r'►►► Run Real Model')
    print(r'------------------------------------------------------------')
    ###-------------Systerm input params ----------------------------
    ### sys_def_params is of shape (m,n),
    ###     m: number of set, 
    ###     n: number of system parameters per set
    print(r' ► Solver (system) properties:')
    print(r'   ♦ {:<17s} : {:15s}'.format('Solver name', sim_parameters.model_name))

    if sim_parameters.sys_def_params is None or sim_parameters.sys_def_params[0] is None:
        print(r'   ♦ System definition parameters: NA ' )
    else:
        print(r'   ♦ System definition parameters: ndim={:d}, nsets={:d}'\
                .format(sim_parameters.sys_def_params.shape[1], sim_parameters.sys_def_params.shape[0]))
                # .format(sim_parameters.sys_def_params.shape[0], len(sim_parameters.sys_def_params)))
    print(r'   ♦ System input parameters:')
    print(r'     ∙ {:<15s} : {}'.format('function',sim_parameters.sys_excit_params[0]))
    print(r'     ∙ {:<15s} : {}'.format('kwargs', sim_parameters.sys_excit_params[1]))

    ###------------- Time domain step ----------------------------
    if sim_parameters.time_max:
        print(r'     ∙ {:<15s} : {:s}, Tmax={:.1e}, dt={:.2f}'\
            .format('time steps', sim_parameters.time_max, sim_parameters.dt))
    else:
        print(r'     ∙ {:<15s} : NA'.format('time steps'))
    # print(r'     ∙ {:<15s} : ndoe={:d}, ndim={:d}, nsets={}'\
            # .format('input variables', sim_parameters.ndoe, sim_parameters.ndim_sys_inputs, sim_parameters.doe_order))

    ###------------- Error properties ----------------------------
    sim_parameters.error.disp()

    print(r' ► Running Simulation...')
    print(r'   ♦ Job list: [{:^20} {:^20}]'.format('# Sys params sets', '# DoE sets'))
    print(r'   ♦ Target  : [{:^20d} {:^20d}]'.format(len(sim_parameters.sys_def_params), sim_parameters.ndoe))
    print(r'   ' + '·'*55)


    for isys_done, isys_def_params in enumerate(sim_parameters.sys_def_params): 
        run_sim_1doe_res = []
        for idoe, sys_input_vars_1doe in enumerate(sim_parameters.sys_input_x):
            if sim_parameters.doe_method.upper() == 'QUADRATURE':
                input_vars, input_vars_weights = sys_input_vars_1doe
            else:
                input_vars, input_vars_weights = sys_input_vars_1doe, None
            ### Run simulations
            y = solver_wrapper(sim_parameters.model_name,input_vars,\
                    error_type      = sim_parameters.error,\
                    sys_excit_params= sim_parameters.sys_excit_params,\
                    sys_def_params  = isys_def_params)

            print(r'   ♦ Achieved: [{:^20d} {:^20d}]'.format(isys_done+1,idoe+1))
            run_sim_1doe_res.append(y)
        run_sim_res.append(run_sim_1doe_res)
    ## 
    if len(run_sim_res) ==1:
        run_sim_res = run_sim_res[0]
    if len(run_sim_res) ==1:
        run_sim_res = run_sim_res[0]
    print(r' ► Simulation Done, Output shape: {}'.format(np.array(run_sim_res).shape) )
    return run_sim_res


if __name__ == '__main__':
    run_solver(sim_parameters)


 
