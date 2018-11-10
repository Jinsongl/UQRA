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
from .dynamic_models import duffing_oscillator
from .dynamic_models import duffing_equation

def solver_wrapper(solver_func, simParams, sys_source, sys_param=None):
    """
    a wrapper for solvers

    Arguments:
        solver_func: callable object
        simParams: general settings for simulation. simParameter class obj
        sys_source: contains either [source_func, *args, **kwargs] 
            or input signal indexed with time
        sys_param: list of system parameters

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
        if sys_param:
            x0, v0, zeta, omega0 = sys_param
        else:
            x0, v0, zeta, omega0 = (0,0, 0.2, 1)

        y = solver_func(time_max,dt,x0,v0,zeta,omega0,add_f=sys_source[0],*sys_source[1:])

    elif solver_func.__name__.upper() == 'ISHIGAMI':
        p = sys_param if sys_param else 2
        x = sys_source
        y = solver_func(x, p=p)

    elif solver_func.__name__.upper() == 'POLY5':
        x = sys_source
        y = solver_func(x)

    elif solver_func.__name__.upper() ==  'DUFFING_OSCILLATOR':
        # sys_source: [source_func, *arg, *kwargs]

        time_max= simParams.time_max
        dt      = simParams.dt

        if len(sys_source) == 3:
            source_func, source_args, source_kwargs = sys_source
        elif len(sys_source) == 2:
            source_func, source_args, source_kwargs = sys_source, None

        ## Default initial condition [0,0]
        if sys_param is not None:
            x0, v0, zeta, omega0, mu = sys_param
        else:
            x0, v0, zeta, omega0, mu = (0,0,0.2,1,1)

        # print(solver_func)
        # print(source_func, source_kwargs, source_args)
        # y = duffing_oscillator(time_max,dt,x0,v0,zeta,omega0,mu, *source_args,source_func=source_func,source_kwargs=source_kwargs)
        y,dt,pstep= solver_func(time_max,dt,x0,v0,zeta,omega0,mu, *source_args,source_func=source_func,source_kwargs=source_kwargs)

    else:
        raise ValueError('Function {} not defined'.format(solver_func__name__)) 
    
    return y


        







