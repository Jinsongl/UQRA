#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import numpy as np

from .dynamic_models import lin_oscillator, duffing_oscillator
from .benchmark import bench1, bench2, bench3, bench4, ishigami
from .solver_wrapper import solver_wrapper
from ..utilities.classes import ErrorType
from ..utilities.helpers import num2print

class Solver(object):
    """
    Solver object, served as a swrapper for all solvers contained in this package
    M(x,t,theta_m;y) = s(x,t; theta_s)
    - x: array -like (ndim, nsamples) ~ dist(x) 
        1. array of x samples (ndim, nsamples), y = M(x)
        2. ndarray of time series
    - t: time 
    - M: deterministic solver, defined by solver_name
    - theta_m: parameters defining solver M of shape (m,n)(sys_def_params)
        - m: number of set, 
        - n: number of system parameters per set
    - y: variable to solve 
    - s: excitation source function, defined by source_func (string) 
    - theta_s: parameters defining excitation function s 

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

    def __init__(self,solver_name, x, theta_m=None, error=None, source_func=None, theta_s=None, *args, **kwargs):
        self.solver_name= solver_name
        self.output     = []
        self.input_x    = x
        self.error      = error if error else ErrorType()
        self.source_func= source_func
        self.theta_s    = theta_s

        print('------------------------------------------------------------')
        print('►►► Initialize Solver Obejct...')
        print('------------------------------------------------------------')
        print(' ► Solver (system) properties:')
        print('   ♦ {:<17s} : {:15s}'.format('Solver name', solver_name))

        if theta_m is None or theta_m[0] is None:
            print('   ♦ Solver parameters: NA ' )
        else:
            print('   ♦ Solver parameters: ndim={:d}, nsets={:d}'.format(theta_m.shape[1], theta_m.shape[0]))
        print('   ♦ System excitation functions:')
        print('     ∙ {:<15s} : {}'.format('function'   , self.source_func))
        print('     ∙ {:<15s} : {}'.format('parameters' , self.theta_s))
        ###------------- Error properties ----------------------------
        self.error.disp()


    def run(self, sim_parameters):
        output = [] # a list saving simulation results
        print('------------------------------------------------------------')
        print('►►► Run solver...')
        print('------------------------------------------------------------')
        print(' ► Solver setting ...')
        if sim_parameters.time_params:
            print('   ♦ {:<15s} : '.format('time parameters'))
            print('     ∙ {:<8s} : {:.2f} ∙ {:<8s} : {:.2f}'.format('start', sim_parameters.time_start, 'end', sim_parameters.time_max )
            print('     ∙ {:<8s} : {:.2f} ∙ {:<8s} : {:.2f}'.format('ramp ', sim_parameters.time_ramp , 'dt ', sim_parameters.dt )

        print(' ► Running Simulation...')
        print('   ♦ Job list: [{:^20} {:^20}]'.format('# solver parameter sets', '# DoE sets'))
        print('   ♦ Target  : [{:^20d} {:^20d}]'.format(len(theta_m), sim_parameters.ndoe))
        print('   ' + '·'*55)

        for isys_done, itheta_m in enumerate(theta_m): 
            run_sim_1doe_res = []
            for idoe, sys_input_x_1doe in enumerate(sim_parameters.sys_input_x):
                if sim_parameters.doe_method.upper() == 'QUADRATURE':
                    input_vars, input_vars_weights = sys_input_x_1doe
                else:
                    input_vars, input_vars_weights = sys_input_x_1doe, None
                ### Run simulations
                y = solver_wrapper(sim_parameters.model_name,input_vars,\
                        error_type      = sim_parameters.error,\
                        sys_excit_params= sim_parameters.sys_excit_params,\
                        sys_def_params  = itheta_m)

                print('   ♦ Achieved: [{:^20d} {:^20d}]'.format(isys_done+1,idoe+1))
                run_sim_1doe_res.append(y)
            output.append(run_sim_1doe_res)
        ## 
        if len(output) ==1:
            output = output[0]
        if len(output) ==1:
            output = output[0]
        print(' ► Simulation Done, Output shape: {}'.format(np.array(output).shape) )
        return output



