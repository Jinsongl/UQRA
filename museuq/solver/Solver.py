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

from .dynamic_models import lin_oscillator, duffing_oscillator, linear_oscillator
from .benchmark import bench1, bench2, bench3, bench4, ishigami
from ..utilities.classes import ErrorType
from ..utilities.helpers import num2print
from ..utilities import constants as const

ALL_SOLVERS = {
    'ISHIGAMI'  : ishigami,
    'BENCH1'    : bench1,
    'BENCH2'    : bench2,
    'BENCH3'    : bench3,
    'BENCH4'    : bench4,
    'DUFFING'   : duffing_oscillator,
    'SDOF'      : linear_oscillator,
    'LINEAR_OSCILLATOR' : linear_oscillator,
    }

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

    def __init__(self,solver_name, x, theta_m=None, error=None, source_func=None, theta_s=None):
        self.solver_name= solver_name
        self.input_x    = x
        self.theta_m    = [theta_m] if theta_m is None else theta_m
        self.error      = error if error else ErrorType()
        self.source_func= source_func
        self.theta_s    = theta_s
        self.output     = []

        print('------------------------------------------------------------')
        print('►►► Initialize Solver Obejct...')
        print('------------------------------------------------------------')
        print(' ► Solver (system) properties:')
        print('   ♦ {:<17s} : {:15s}'.format('Solver name', solver_name))

        if self.theta_m is None or self.theta_m[0] is None:
            print('   ♦ Solver parameters: NA ' )
        else:
            print('   ♦ Solver parameters: ndim={:d}, nsets={:d}'.format(self.theta_m.shape[1], self.theta_m.shape[0]))
        print('   ♦ System excitation functions:')
        print('     ∙ {:<15s} : {}'.format('function'   , self.source_func))
        print('     ∙ {:<15s} : {}'.format('parameters' , self.theta_s))
        ###------------- Error properties ----------------------------
        self.error.disp()


    def run(self, doe_obj, *args, **kwargs):
        self.output = [] # a list saving simulation results
        print(' ► Running Simulation...')
        print('   ♦ Job list: [{:^20} {:^20}]'.format('# solver parameter sets', '# DoE sets'))
        print('   ♦ Target  : [{:^20d} {:^20d}]'.format(len(self.theta_m), doe_obj.ndoe))
        print('   ' + '·'*55)

        for isys_done, itheta_m in enumerate(self.theta_m): 
            run_sim_1doe_res = []
            for idoe, isamples_x in enumerate(doe_obj.mapped_samples):
                if const.DOE_METHOD_FULL_NAMES[doe_obj.method.lower()] == 'QUADRATURE':
                    input_vars, input_vars_weights = isamples_x
                else:
                    input_vars, input_vars_weights = isamples_x, None
                ### Run simulations
                y = self._solver_wrapper(input_vars, theta_m = itheta_m, *args, **kwargs)
                print('   ♦ Achieved: [{:^20d} {:^20d}]'.format(isys_done+1,idoe+1))
                run_sim_1doe_res.append(y)
            self.output.append(run_sim_1doe_res)
        ## 
        if len(self.output) ==1:
            self.output = self.output[0]
        if len(self.output) ==1:
            self.output = self.output[0]
        print(' ► Simulation Done, Output shape: {}'.format(np.array(self.output).shape) )
        return self.output

    def _solver_wrapper(self, x, theta_m=None, *args, **kwargs):
        """
        a wrapper for solvers

        solver_name: string 
        M(x,t,theta_m;y) = s(x,t; theta_s)
        - x: array -like (ndim, nsamples) ~ dist(x) 
            1. array of x samples (ndim, nsamples), y = M(x)
            2. ndarray of time series
        - t: time 
        - M: deterministic solver, defined by solver_name
        - theta_m: parameters defining solver M of shape (m,n)(theta_m)
            - m: number of set, 
            - n: number of system parameters per set
        - y: variable to solve 
        - s: excitation source function, defined by source_func (string) 
        - theta_s: parameters defining excitation function s 

        Return:
            return results from one solver run
            of shape(nsample, nqoi), each column represents a full time series
            or just a number for single output solvers

        """

        # solver_name, sterm_dist = model_def #if len(model_def) == 2 else model_def[0], None
        # print(solver_name)
        try:
            solver = ALL_SOLVERS[self.solver_name.upper()]
        except KeyError:
            print(f"{self.solver_name.upper()} is not defined" )
        assert (callable(solver)), '{:s} not callable'.format(solver.__name__)
        
        if self.solver_name.upper() == 'ISHIGAMI':
            p = theta_m if theta_m else [7,0.1]
            y = solver(x, p=p)

        elif self.solver_name.upper()[:5] == 'BENCH':
            y = solver(x, error)

        elif self.solver_name.upper() == 'LIN_OSCILLATOR':
            time_max= simParameters.time_max
            dt      = simParameters.dt
            ## Default initial condition [0,0]
            if theta_m:
                x0, v0, zeta, omega0 = theta_m
            else:
                x0, v0, zeta, omega0 = (0,0, 0.2, 1)

            y = solver(time_max,dt,x0,v0,zeta,omega0,add_f=x[0],*x[1:])

        elif self.solver_name.upper() == 'LINEAR_OSCILLATOR':
            tmax,dt =1000, 0.1
            t = np.arange(0,tmax, dt)
            y = [linear_oscillator(t,ix) for ix in x.T]


        elif self.solver_name.upper() ==  'DUFFING_OSCILLATOR':
            # x: [source_func, *arg, *kwargs]

            time_max  = simParameters.time_max
            dt        = simParameters.dt
            normalize = simParameters.normalize

            if len(x) == 3:
                source_func, source_kwargs, source_args = x
            elif len(x) == 2:
                source_func, source_kwargs, source_args = x[0], None, x[1]

            ## Default initial condition [0,0]
            if theta_m is not None:
                x0, v0, zeta, omega0, mu = theta_m
            else:
                x0, v0, zeta, omega0, mu = (0,0,0.02,1,1)

            # print(solver)
            # print(source_func, source_kwargs, source_args)
            # y = duffing_oscillator(time_max,dt,x0,v0,zeta,omega0,mu, *source_args,source_func=source_func,source_kwargs=source_kwargs)
            y,dt,pstep= solver(time_max,dt,x0,v0,zeta,omega0,mu, *source_args,source_func=source_func,source_kwargs=source_kwargs,normalize=normalize)

        else:
            raise ValueError('Function {} not defined'.format(solver.__name__)) 
        
        return y



