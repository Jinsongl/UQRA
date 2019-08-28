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
from solver.dynamic_models import lin_oscillator
from solver.dynamic_models import duffing_oscillator
from solver.benchmark import ishigami
from solver.benchmark import bench1, bench2, bench3, bench4

all_solvers = {
        'ISHIGAMI'  : ishigami,
        'BENCH1'    : bench1,
        'BENCH2'    : bench2,
        'BENCH3'    : bench3,
        'BENCH4'    : bench4,
        'DUFFING'   : duffing_oscillator,
        }



def solver_wrapper(solver_name, sys_input_x, error=None, sys_input_params=None, sys_def_params=None, *args, **kwargs):
    """
    a wrapper for solvers

    Arguments:
        solver_name: string 
        sys_input_x: ndarray of shape(ndim, nsamples)
            - sys_def_params: list of parameter sets defining the solver
                if no sys_def_params is required, sys_def_params = [None]
                e.g. for duffing oscillator:
                sys_def_params = [np.array([0,0,1,1,1]).reshape(5,1)] # x0,v0, zeta, omega_n, mu 
            - sys_input_params = [sys_excit_func_name, sys_excit_funys_kwargs, sys_input_x]
                [None,None,sys_input_x] if sys_excit_func_name, sys_excit_func_kwargs are not available
    Return:
        return results from one solver run
        of shape(nsample, nqoi), each column represents a full time series
        or just a number for single output solvers

    """

    # solver_name, sterm_dist = model_def #if len(model_def) == 2 else model_def[0], None
    # print(solver_name)
    solver = all_solvers[solver_name.upper()]

    assert (callable(solver)), '{:s} not callable'.format(solver.__name__)
    

    if solver.__name__.upper() == 'LIN_OSCILLATOR':
        time_max= simParams.time_max
        dt      = simParams.dt
        ## Default initial condition [0,0]
        if sys_def_params:
            x0, v0, zeta, omega0 = sys_def_params
        else:
            x0, v0, zeta, omega0 = (0,0, 0.2, 1)

        y = solver(time_max,dt,x0,v0,zeta,omega0,add_f=sys_input_x[0],*sys_input_x[1:])

    elif solver.__name__.upper() == 'ISHIGAMI':
        p = sys_def_params if sys_def_params else [7,0.1]
        y = solver(sys_input_x, p=p)

    elif solver.__name__.upper()[:5] == 'BENCH':
        y = solver(sys_input_x, error)

    elif solver.__name__.upper() ==  'DUFFING_OSCILLATOR':
        # sys_input_x: [source_func, *arg, *kwargs]

        time_max  = simParams.time_max
        dt        = simParams.dt
        normalize = simParams.normalize

        if len(sys_input_x) == 3:
            source_func, source_kwargs, source_args = sys_input_x
        elif len(sys_input_x) == 2:
            source_func, source_kwargs, source_args = sys_input_x[0], None, sys_input_x[1]

        ## Default initial condition [0,0]
        if sys_def_params is not None:
            x0, v0, zeta, omega0, mu = sys_def_params
        else:
            x0, v0, zeta, omega0, mu = (0,0,0.02,1,1)

        # print(solver)
        # print(source_func, source_kwargs, source_args)
        # y = duffing_oscillator(time_max,dt,x0,v0,zeta,omega0,mu, *source_args,source_func=source_func,source_kwargs=source_kwargs)
        y,dt,pstep= solver(time_max,dt,x0,v0,zeta,omega0,mu, *source_args,source_func=source_func,source_kwargs=source_kwargs,normalize=normalize)

    else:
        raise ValueError('Function {} not defined'.format(solver.__name__)) 
    
    return y


        







