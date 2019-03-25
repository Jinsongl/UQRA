#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import numpy as np
from solver.solver_wrapper import solver_wrapper

class ErrorType():
    def __init__(self, name=None, params=[],size=None):
        self.name = name
        self.params = params
        self.size = None

def run_sim(model_def, simParams):
    """
    Run simulation with given "solver" (real model) and predefined environment
    General format of a solver
        y =  M(x,sys_def_params)
        M: system solver, taking a set of system parameters and input x
        x: system inputs, could be one of the following three formats
            1. array of x samples (ndim, nsamples), y = M(x)
            2. ndarray of time series
            3. inputs for the sys_input_func_name to generate time series
    Arguments:
        model_def: [solver function name, short-term distribution name, [params], size]
        return [*.ndofs] array. Each column is a time series for specified channel
        simParams: simParameter class object

    Return:
        List of simulation results (Different doe_order, sample size would change, so return list)

        1. len(res) = len(doe_order)
        2. For each element in res: [#source args sets, # sys parameter sets, returns from solver]
        3. Return from solver 
            a). [*, ndofs]
                each column represents a time series for one dof
            b). 

    """
    NDOE_DONE, NSOURCE_DONE, NSYS_DONE = 0, 0, 0 
    run_sim_res = []
    [model_def.append(None) for i in range (len(model_def),4)]
    model_name, error_type_name, error_type_params, error_type_size = model_def
    error_type = ErrorType(name=error_type_name,params=error_type_params,size=error_type_size)

    # sys_def_params = args[0] if len(args) == 1 else None
    print('------------------------------------------------------------')
    print('►►► Run Real Model')
    print('------------------------------------------------------------')
    print(' ► Solver (system) properties:')
    print('   ♦ {:<17s} : {:15s}'.format('Solver name', model_name))

    if simParams.sys_def_params[0]:
        print('   ♦ System definition parameters: ndim={}, nsets={:d}'\
                .format(simParams.sys_def_params[0].shape, len(simParams.sys_def_params)))
    else:
        print('   ♦ System definition parameters: NA ' )

    print('   ♦ System input parameters:')
    print('     ∙ {:<15s} : {}'.format('function',simParams.sys_input_params[0]))
    print('     ∙ {:<15s} : {}'.format('kwargs', simParams.sys_input_params[1]))

    if simParams.time_max:
        print('     ∙ {:<15s} : {:s}, Tmax={:.1e}, dt={:.2f}'\
            .format('time steps', simParams.time_max, simParams.dt))
    else:
        print('     ∙ {:<15s} : NA'.format('time steps'))
    print('     ∙ {:<15s} : ndoe={:d}, ndim={:d}, nsets={}'\
            .format('input variables', simParams.ndoe, simParams.nsys_input_vars_dim, simParams.nsamples_per_doe))

    print(' ► Running Simulation...')
    print('   ♦ Job list: [{:^20} {:^20}]'.format('# Sys params sets', '# DoE sets'))
    print('   ♦ Target  : [{:^20d} {:^20d}]'.format(len(simParams.sys_def_params), simParams.ndoe))
    print('   ' + '·'*55)
    for isys_def_params in simParams.sys_def_params:
        NSYS_DONE += 1 
        run_sim_1doe_res = []
        for sys_input_vars_1doe in simParams.sys_input_vars:

            if simParams.doe_method.upper() == 'QUADRATURE':
                input_vars, input_vars_weights = sys_input_vars_1doe
            else:
                input_vars, input_vars_weights = sys_input_vars_1doe, None

            y = solver_wrapper(model_name,simParams,input_vars, error_type, sys_def_params=isys_def_params)

            NDOE_DONE += 1
            print('   ♦ Achieved: [{:^20d} {:^20d}]'.format(NSYS_DONE,NDOE_DONE))
            run_sim_1doe_res.append(y)
        run_sim_res.append(run_sim_1doe_res)
    print(' ► Simulation Done' )
    return run_sim_res
## here, need to rebuild three loops


    # if isinstance(sys_input_vars, np.ndarray):
        # ## only need inner loop for changing physical system
        # print('Source parameters: signal of shape {}'.format(sys_input_vars.shape))
        # res_inner = _loop_sysparams(solver_func, simParams, sys_input_vars, sys_def_params)
        # run_sim_res.append(res_inner)

    # elif isinstance(sys_input_vars, list) and callable(sys_input_vars[0]):
        # print('Source parameters:')
        # # Three loops:
        # # for each DoE:
        # #   for each source args:
        # #      for each sys parameter sets
        # # return [ndoe, nsourceargs, nsysparams, ...]

        # ## First loop for different doe order sys_sources
        # if len(sys_input_vars) == 3:
            # sys_input_func_name, sys_input_kwargs, input_vars = sys_input_vars
        # elif len(sys_input_vars) == 2:
            # sys_input_func_name, sys_input_kwargs, input_vars = sys_input_vars[0], None, sys_input_vars[1]
        # ndoe            = simParams.ndoe 
        # nsys_input_vars_dim      = simParams.nsys_input_vars_dim
        # nsamples_per_doe= simParams.nsamples_per_doe 

        # print('   function : {:s}'.format(sys_input_func_name)
        # print('   arguments: ndoe={:d}, ndim={:d}, nsets={}'\
                # .format(ndoe, nsys_input_vars_dim, nsamples_per_doe))
        # print('   kwargs   : {}'.format(sys_input_kwargs))
        # print('   ------------------------------------')
        # print('   Job list: [# DoE sets, # Souce args, # Sys params sets]')
        # print('   Target  : [{:4d}, {}, {:4d}]'.format(ndoe, nsamples_per_doe, sys_def_params.shape[1]))
        # print('   --------')
        # for i, sys_input_vars_1doe in enumerate(input_vars):
            # sys_input_params_1doe = [sys_input_func_name, sys_input_kwargs, sys_input_vars_1doe]
            # y = _loop_input_vars(solver_func, simParams, sys_input_params_1doe,sys_def_params) 
            # run_sim_res.append(y)
            # NDOE_DONE +=1
    # else:
        # raise NotImplementedError("Type of system source not defined")


           

# def _loop_sysparams(solver_func, simParams, sys_source_fixed, sys_param_all):
    # """
    # For a fixed sys_input_vars, loop over sys_def_params if available
    # """
    # global NSYS_DONE
    # res = []
    # ## no system parameters is needed
    # if sys_param_all is None:
        # y = solver_wrapper(solver_func,simParams,sys_source_fixed)
        # res.append(y)
    # ## different sets of system parameters
    # else:
        # for i, isys_param in enumerate(sys_param_all.T):
            # y = solver_wrapper(solver_func,simParams,sys_source_fixed,sys_def_params=isys_param)
            # res.append(y)
            # NSYS_DONE = i+1
            # # print('   Target  : [{:8d}, {:8d}, {:8d}]'.format(ndoe, nsamples_per_doe, sys_def_params.shape[1]))
    # # print(np.array(res).shape)
    # return np.array(res)

# def _loop_input_vars(solver_func,simParams, sys_input_vars_1doe, sys_param_all):

    # global NSOURCE_DONE
    # res = []

    # if simParams.doe_method.upper() in ['QUAD', 'GQ']:
        # input_vars, input_vars_weights = sys_input_vars_1doe
    # else:
        # input_vars, input_vars_weights = sys_input_vars_1doe, None

    # ## input_vars must of shape(ndim, nsampes)
    # # NEED TO IMPLEMENT WHEN INPUT IS ARRAY
    # if sys_input_func_name is None:
        # ## y = M(x) 
        # res_inner = _loop_sysparams(solver_func, simParams, isys_source, sys_param_all)
    # else: 


    # for i, isource_args in enumerate(input_vars.T):

        # # print(' [{:8d}'.format(i),end='')
        # isys_source = [sys_input_func_name, sys_input_kwargs, isource_args]
        # res_inner = _loop_sysparams(solver_func, simParams, isys_source, sys_param_all)
        # res.append(res_inner)
        # NSOURCE_DONE +=1
    # return np.array(res)




 
