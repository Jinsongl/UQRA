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

def solver_wrapper(solver_func, *args):
    """
    a wrapper for solvers

    Arguments:
        solver_func: callable object
        # simParams: general settings for simulation. simParameter class obj
        *args: containing arguments for the solver
    Return:
        return results from one solver_func run
        of shape(nsample, nqoi), each column represents a full time series
        or just a number for single output solvers

    """
    assert (callable(solver_func)), '{:s} not callable'.format(solver_func.__name__)

    if solver_func.__name__.upper() == 'LIN_OSCILLATOR':
        assert (len(args) == 8 or len(args)==9), 'Expecting 8 or 9 arguments for {:s} but {:d} were given'.format(solver_func.__name__, len(args))
        Hs,Tp   = args
        Tmax    = simParams.time_max
        dt      = simParams.dt
        seed    = simParams.seed
        y = solver_func(Hs,Tp,Tmax,dt,seed)
    elif solver_func.__name__.upper() == 'ISHIGAMI':
        x,p = args if len(args) == 2 else (args[0], None)
        y = solver_func(x, p=p)
    elif solver_func.__name__.upper() == 'POLY5':
        x = args[0]
        y = solver_func(x)
    elif solver_func.__name__.upper() == 
    else:
        pass
    
    return y


    


        







