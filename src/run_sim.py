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
    def __init__(self, name=None, params=None, size=None):
        """
        name:   string, error distribution name or None if not defined
        params: list of error distribution parameters, float or array_like of floats 
                [ [mu1, sigma1, ...]
                  [mu2, sigma2, ...]
                  ...]
        size:   list of [int or tuple of ints], optional
            -- Output shape:
            If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. 
            If size is None (default), a single value is returned if loc and scale are both scalars. 
            Otherwise, np.broadcast(loc, scale).size samples are drawn.
        """
        assert name is None or isinstance(name, str) 
        if name is None:
            self.name   = None 
            self.params = None 
            self.size   = None 
        elif len(params) == 1:
            self.name   = name
            self.params = params[0]
            self.size   = size[0] if size else None
        else:
            self.name   = [] 
            for _ in range(len(params)):
                self.name.append(name) 
            self.size   = size if size else [None] * len(params)
            self.params = params

    def tolist(self, ndoe):
        if not isinstance(self.name, list):
            return [ErrorType(self.name, [self.params], [self.size])] * ndoe
        else:
            assert len(self.name) == ndoe,'ErrorType.name list length {} != ndoe {}'.format(len(self.name), ndoe)
            error_type_list = []
            for i in range(ndoe):
                error_type_list.append(ErrorType(self.name[i], [self.params[i]], self.size[i]))
            return error_type_list

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
    NDOE_DONE, NSOURCE_DONE, NSYS_DONE = 0, 0, 0 
    run_sim_res = []
    [model_def.append(None) for i in range (len(model_def),4)]
    model_name, error_type_name, error_type_params, error_type_size = model_def
    error_type = ErrorType(name=error_type_name,params=error_type_params,size=error_type_size)
    # print('before to list')
    # print('error type name:{}'.format(error_type.name))
    # print('error type params:{}'.format(error_type.params))
    # print('error type size:{}'.format(error_type.size))
    error_type_list = error_type.tolist(simParams.ndoe)
    # print('after to list')
    # print('error type name:{}'.format(error_type_list[0].name))
    # print('error type params:{}'.format(error_type_list[0].params))
    # print('error type size:{}'.format(error_type_list[0].size))

    # sys_def_params = args[0] if len(args) == 1 else None
    print('------------------------------------------------------------')
    print('►►► Run Real Model')
    print('------------------------------------------------------------')
    print(' ► Solver (system) properties:')
    print('   ♦ {:<17s} : {:15s}'.format('Solver name', model_name))

    if simParams.sys_def_params is None or simParams.sys_def_params[0] is None:
        print('   ♦ System definition parameters: NA ' )
    else:
        print('   ♦ System definition parameters: ndim={}, nsets={:d}'\
                .format(simParams.sys_def_params.shape[1], simParams.sys_def_params.shape[0]))
                # .format(simParams.sys_def_params.shape[0], len(simParams.sys_def_params)))

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
    if error_type_name:
        print('   ♦ Short-term/error distribution parameters:')
        print('     ∙ {:<15s} : {}'.format('dist_name'  ,error_type_name))
        for i, ierror_params in enumerate(error_type_params):
            ierror_params_shape = []
            for iparam in ierror_params:
                if np.isscalar(iparam):
                    ierror_params_shape.append(1)
                else:
                    ierror_params_shape.append(np.array(iparam).shape)

            if i == 0:
                print('     ∙ {:<15s} : {}'.format('dist_params',ierror_params_shape))
            else:
                print('     ∙ {:<15s} : {}'.format('',ierror_params_shape))
        print('     ∙ {:<15s} : {}'.format('dist_size'  ,error_type_size))
    else:
        print('   ♦ Short-term/error distribution parameters: NA')

    ### sys_def_params is of shape (m,n), m: number of set, n: number of system parameters per set
    sys_def_params_len  = len(simParams.sys_def_params)

    print(' ► Running Simulation...')
    print('   ♦ Job list: [{:^20} {:^20}]'.format('# Sys params sets', '# DoE sets'))
    print('   ♦ Target  : [{:^20d} {:^20d}]'.format(sys_def_params_len, simParams.ndoe))
    print('   ' + '·'*55)

    for isys_def_params in sys_def_params: 
        NSYS_DONE += 1 
        run_sim_1doe_res = []
        for idoe, sys_input_vars_1doe in enumerate(simParams.sys_input_vars):
            if simParams.doe_method.upper() == 'QUADRATURE':
                input_vars, input_vars_weights = sys_input_vars_1doe
            else:
                input_vars, input_vars_weights = sys_input_vars_1doe, None
            y = solver_wrapper(model_name,simParams,input_vars, error_type_list[idoe],\
                    sys_def_params=isys_def_params)

            NDOE_DONE += 1
            print('   ♦ Achieved: [{:^20d} {:^20d}]'.format(NSYS_DONE,NDOE_DONE))
            run_sim_1doe_res.append(y)
        run_sim_res.append(run_sim_1doe_res)
    ## 
    if len(run_sim_res) ==1:
        run_sim_res = run_sim_res[0]
    if len(run_sim_res) ==1:
        run_sim_res = run_sim_res[0]
    print(' ► Simulation Done, Output shape: {}'.format(np.array(run_sim_res).shape) )
    return run_sim_res




 
