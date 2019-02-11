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

def solver_wrapper(solver_func, simParams, sys_input_vars, sys_params=None):
    """
    a wrapper for solvers

    Arguments:
        solver_func: callable object
        simParams: general settings for simulation. simParameter class obj
        sys_input_vars: contains either [source_func, *args, **kwargs] 
            or input signal indexed with time
        sys_params: one system parameter set

    Return:
        return results from one solver_func run
        of shape(nsample, nqoi), each column represents a full time series
        or just a number for single output solvers

    """
    assert (callable(solver_func)), '{:s} not callable'.format(solver_func.__name__)
    

    if solver_func.__name__.upper() == 'LIN_OSCILLATOR':
        time_max= simParams.time_max
        dt      = simParams.dt
        ## Default initial condition [0,0]
        if sys_params:
            x0, v0, zeta, omega0 = sys_params
        else:
            x0, v0, zeta, omega0 = (0,0, 0.2, 1)

        y = solver_func(time_max,dt,x0,v0,zeta,omega0,add_f=sys_input_vars[0],*sys_input_vars[1:])

    elif solver_func.__name__.upper() == 'ISHIGAMI':
        p = sys_params if sys_params else 2
        x = sys_input_vars
        y = solver_func(x, p=p)

    elif solver_func.__name__.upper() == 'BENCHMARK1_NORMAL':
        mu, sigma = sys_params if sys_params else (0, 0.5)
        x = sys_input_vars
        y = solver_func(x, mu=mu, sigma=sigma)

    elif solver_func.__name__.upper() == 'POLY5':
        x = sys_input_vars
        y = solver_func(x)

    elif solver_func.__name__.upper() ==  'DUFFING_OSCILLATOR':
        # sys_input_vars: [source_func, *arg, *kwargs]

        time_max  = simParams.time_max
        dt        = simParams.dt
        normalize = simParams.normalize

        if len(sys_input_vars) == 3:
            source_func, source_kwargs, source_args = sys_input_vars
        elif len(sys_input_vars) == 2:
            source_func, source_kwargs, source_args = sys_input_vars[0], None, sys_input_vars[1]

        ## Default initial condition [0,0]
        if sys_params is not None:
            x0, v0, zeta, omega0, mu = sys_params
        else:
            x0, v0, zeta, omega0, mu = (0,0,0.02,1,1)

        # print(solver_func)
        # print(source_func, source_kwargs, source_args)
        # y = duffing_oscillator(time_max,dt,x0,v0,zeta,omega0,mu, *source_args,source_func=source_func,source_kwargs=source_kwargs)
        y,dt,pstep= solver_func(time_max,dt,x0,v0,zeta,omega0,mu, *source_args,source_func=source_func,source_kwargs=source_kwargs,normalize=normalize)

    else:
        raise ValueError('Function {} not defined'.format(solver_func__name__)) 
    
    return y


        







