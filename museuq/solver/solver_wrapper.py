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
from .dynamic_models import * 
from .benchmark import * 

ALL_SOLVERS = {
    'ISHIGAMI'  : ishigami,
    'BENCH1'    : bench1,
    'BENCH2'    : bench2,
    'BENCH3'    : bench3,
    'BENCH4'    : bench4,
    'DUFFING'   : duffing_oscillator,
    }



def solver_wrapper(solver_name, x, theta_m=None, error=None, source_func=None,  theta_s=None, *args, **kwargs):
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
        solver = ALL_SOLVERS[solver_name.upper()]
    except KeyError:
        print(f"{solver} is not defined" )
    assert (callable(solver)), '{:s} not callable'.format(solver.__name__)
    
    if solver.__name__.upper() == 'ISHIGAMI':
        p = theta_m if theta_m else [7,0.1]
        y = solver(x, p=p)

    elif solver.__name__.upper()[:5] == 'BENCH':
        y = solver(x, error)

    elif solver.__name__.upper() == 'LIN_OSCILLATOR':
        time_max= simParameters.time_max
        dt      = simParameters.dt
        ## Default initial condition [0,0]
        if theta_m:
            x0, v0, zeta, omega0 = theta_m
        else:
            x0, v0, zeta, omega0 = (0,0, 0.2, 1)

        y = solver(time_max,dt,x0,v0,zeta,omega0,add_f=x[0],*x[1:])

    elif solver.__name__.upper() ==  'DUFFING_OSCILLATOR':
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


        







