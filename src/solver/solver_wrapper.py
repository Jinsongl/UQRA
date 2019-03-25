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
        'ishigami'  : ishigami,
        'bench1'    : bench1,
        'bench2'    : bench2,
        'bench3'    : bench3,
        'bench4'    : bench4,
        }



def solver_wrapper(solver_name, simParams, sys_input_vars, error_type, sys_def_params=None):
    """
    a wrapper for solvers

    Arguments:
        solver_name: string 
        simParams: general settings for simulation. simParameter class obj
        sys_input_vars: contains either [source_func, *args, **kwargs] 
            or input signal indexed with time
        sys_def_params: one system parameter set

    Return:
        return results from one solver run
        of shape(nsample, nqoi), each column represents a full time series
        or just a number for single output solvers

    """

    # solver_name, sterm_dist = model_def #if len(model_def) == 2 else model_def[0], None
    # print(solver_name)
    solver = all_solvers[solver_name]

    assert (callable(solver)), '{:s} not callable'.format(solver.__name__)
    

    if solver.__name__.upper() == 'LIN_OSCILLATOR':
        time_max= simParams.time_max
        dt      = simParams.dt
        ## Default initial condition [0,0]
        if sys_def_params:
            x0, v0, zeta, omega0 = sys_def_params
        else:
            x0, v0, zeta, omega0 = (0,0, 0.2, 1)

        y = solver(time_max,dt,x0,v0,zeta,omega0,add_f=sys_input_vars[0],*sys_input_vars[1:])

    elif solver.__name__.upper() == 'ISHIGAMI':
        p = sys_def_params if sys_def_params else 2
        x = sys_input_vars
        y = solver(x, p=p)

    elif solver.__name__.upper()[:5] == 'BENCH':
        x = sys_input_vars
        y = solver(x, error_type)
        # if sterm_dist:
            # mu = 0 * x
            # sigma = dist_error_params(x)
            # y = solver(x, sterm_dist, mu, sigma)
        # else:

    elif solver.__name__.upper() ==  'DUFFING_OSCILLATOR':
        # sys_input_vars: [source_func, *arg, *kwargs]

        time_max  = simParams.time_max
        dt        = simParams.dt
        normalize = simParams.normalize

        if len(sys_input_vars) == 3:
            source_func, source_kwargs, source_args = sys_input_vars
        elif len(sys_input_vars) == 2:
            source_func, source_kwargs, source_args = sys_input_vars[0], None, sys_input_vars[1]

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


        







