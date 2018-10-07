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

def solver_wrapper(solver_func, simParams, *args):
    """
    a wrapper for solvers

    Arguments:
        solver_func: callable object
        simParams: general settings for simulation. simParameter class obj
        **kwargs: dictionary containing arguments for the solver
    Return:
        return results from one solver_func run

    """
    assert (callable(solver_func)), '{:s} not callable'.format(solver_func.__name__)
    
    # kwargs = {key.upper(): value for key, value in kwargs.items()}
    if solver_func.__name__.upper() == 'DETERMINISTIC_LIN_SDOF':
        assert len(args) == 2
        Hs,Tp   = args
        Tmax    = simParams.time_max
        dt      = simParams.dt
        seed    = simParams.seed
        # isDisplay = 0
        # if not isDisplay: 
            # if seed[0]:
                # print('\tFixed seed with {:d}'.format(seed[1]))
                # np.random.seed(int(seed[1]))
            # else:
                # print('\tRunning with random seed')
            # print ("\tRunning deterministic SDOF system ")
            # isDisplay = 1

        y = solver_func(Hs,Tp,Tmax,dt,seed)
    else:
        pass
    
    return y


    


        







