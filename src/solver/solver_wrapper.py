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

def solver_wrapper(solver_func, simParams, x, sys_param=None):
    """
    a wrapper for solvers

    Arguments:
        solver_func: callable object
        simParams: general settings for simulation. simParameter class obj
        *args: containing arguments for the solver
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
            x0, v0 = sys_param['x0']
            zeta, omega0 = sys_param['c']
        else:
            x0, v0 = (0,0)
            zeta, omega0 = (0.2, 1)

        y = solver_func(time_max,dt,x0,v0,zeta,omega0,x)

    elif solver_func.__name__.upper() == 'ISHIGAMI':
        p = sys_param['p'] if sys_param else 2
        y = solver_func(x, p=p)
    elif solver_func.__name__.upper() == 'POLY5':
        y = solver_func(x)
    elif solver_func.__name__.upper() ==  'DUFFING_OSCILLATOR':
        time_max= simParams.time_max
        dt      = simParams.dt
        ## Default initial condition [0,0]
        if sys_param:
            x0, v0 = sys_param['x0']
            zeta, omega0, mu = sys_param['c']
        else:
            x0, v0 = (0,0)
            zeta, omega0, mu = (0.2, 1, 1)

        y = solver_func(time_max,dt,x0,v0,zeta,omega0,mu,x)
    else:
        pass
    
    return y


        







