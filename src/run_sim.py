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
from solver.solver_wrapper import solver_wrapper
NDOE_DONE=1
NSOURCE_DONE=1
NSYS_DONE=1
def run_sim(solver_func, simParams,sys_source, *args):
    """
    Run simulation with given "solver" (real model) and predefined environment
    Arguments:
        solver_func: solver. Return [nChannel, nTimeSample] array. Each row is a time series for specified channel
        simParams: simParameter class object
        sys_source: list containing [source_func, source_args, source_kwargs]

    Optional:
        sys_param: parameters for solver system of shape [ndim, nsamples], ndim is the number of system parameters

    Return:
        List of simulation results
        Different doe_order, sample size would change, so return list

        1. len(res) = len(doe_order)
        2. For each element in res
            [#source args sets, # sys parameter sets, returns from solver]
        3. Return from solver 
            a). [*, ndofs]
                each column represents a time series for one dof
            b). 

        f_obsx: list of length len(doe_order)
            each element contains input sample variables both in zeta and physical space
            of shape (ndim, nsamples), each column includes [zeta_0,...zeta_n[,w], phy_0,...,phy_n].T, 
            weight is optional

        f_obsy: list of length len(doe_order)
            each element contains solver output for certain amount of samples corresponding to doe_order
        
        f_obsy_stats: list of length len(doe_order)
            each element contains (nsamples, nstats, nqois)

    """
# sys_source: contains either [source_func, *args, **kwargs] or input signal indexed with time
# sys_param: ndarray of system parameters of shape[ndim, nsamples]

    # if len(args) == 1:
        # sys_param = args
    # else:
        # sys_param = None
    sys_param = args[0] if len(args) == 1 else None
    print('************************************************************')
    print('Start running: {:s}, Tmax={:.1e}, dt={:.2f}'.format(solver_func.__name__,simParams.time_max, simParams.dt))
    if sys_param is not None:
        print('System parameters: ndim={:d}, nsets={:d}'.format(sys_param.shape[0], sys_param.shape[1]))

    resall = []

    global NDOE_DONE
    global NSOURCE_DONE
    global NSYS_DONE

    if isinstance(sys_source, np.ndarray):
        print('Source parameters: signal of shape {}'.format(sys_source.shape))
        ## only need inner loop for changing physical system
        res_inner = _run_fixsource_loop(solver_func, simParams, sys_source, sys_param)
        resall.append(res_inner)

    elif isinstance(sys_source, list) and callable(sys_source[0]):
        print('Source parameters:')
        # Three loops: first loop will go through different doe orders
        # second loop will go through different source_func arguments
        # third loop will evaluate at given source_func argument, system response with different parameters.

        ## First loop for different doe order sys_sources
        if len(sys_source) == 3:
            source_func, source_args, source_kwargs = sys_source
        elif len(sys_source) == 2:
            source_func, source_args, source_kwargs = sys_source, None
        ndoe = len(source_args)
        nsouce_dim = source_args[0][0].shape[0]
        nsets_per_doe = source_args[0][0].shape[1]
        print('   function : {:s}'.format(source_func.__name__))
        print('   arguments: ndoe={:d}, ndim={:d}, nsets={:d}'\
                .format(ndoe, nsouce_dim, nsets_per_doe))
        print('   kwargs   : {}'.format(source_kwargs))
        print('   ------------------------------------')
        print('   Job list: [# DOE sets, # Souce args, # Sys params]')
        print('   Target  : [{:8d}, {:8d}, {:8d}]'.format(ndoe, nsets_per_doe, sys_param.shape[1]))
        print('   --------')
        for i, source_args_1doe in enumerate(source_args):
            sys_source_1doe = [source_func, source_args_1doe, source_kwargs]
            y = _run_fixdoeorder_loop(solver_func, simParams, sys_source_1doe,sys_param) 
            resall.append(y)
            NDOE_DONE +=1

    return resall
           

def _run_fixsource_loop(solver_func,simParams,sys_source_fixed,sys_param_all):
    global NSYS_DONE
    res = []
    ## system parameters don't change
    if sys_param_all is None:
        y = solver_wrapper(solver_func,simParams,sys_source_fixed)
        res.append(y)
    ## different setzs of system parameters
    else:
        for i , isys_param in enumerate(sys_param_all.T):
            y = solver_wrapper(solver_func,simParams,sys_source_fixed,sys_param=isys_param)
            res.append(y)
            NSYS_DONE = i+1
            # print('   Target  : [{:8d}, {:8d}, {:8d}]'.format(ndoe, nsets_per_doe, sys_param.shape[1]))
            print('   Achieved: [{:8d}, {:8d}, {:8d}]'.format(NDOE_DONE, NSOURCE_DONE, NSYS_DONE))
    # print(np.array(res).shape)
    return np.array(res)

def _run_fixdoeorder_loop(solver_func,simParams,sys_source_1doe, sys_param_all):
    global NSOURCE_DONE
    res = []
    source_func, source_args, source_kwargs = sys_source_1doe
    if simParams.doe_method.upper() in ['QUAD', 'GQ']:
        source_args, source_args_weights = source_args 
    else:
        source_args, source_args_weights = source_args, None
    ## source_args must of shape(ndim, nsampes)
    for i, isource_args in enumerate(source_args.T):

        # print(' [{:8d}'.format(i),end='')
        isys_source = [source_func, isource_args, source_kwargs]
        res_inner = _run_fixsource_loop(solver_func, simParams, isys_source, sys_param_all)
        res.append(res_inner)
        NSOURCE_DONE +=1

    # print(np.array(res).shape)
    return np.array(res)


    # if sys_source:

        # if simParams.doe_method.upper() in ['QUAD', 'GQ']:
            # phy_cor, phy_weights = sys_in
        # else:
            # phy_cor = sys_in

        # for iphy_cor in phy_cor.T:

            # tmax, dt = simParams.time_max, simParams.dt

            # psd_name    = sys_source['name']
            # psd_method  = sys_source['method']
            # psd_sides   = sys_source['sides']
            # sys_source = [gen_gauss_time_series, iphy_cor, ]

            # t, sys_input_signal = gen_gauss_time_series(tmax, dt, psd_name, iphy_cor,\
                    # method=psd_method, sides=psd_sides)
            # add_f = lambda t: 

            # f_obs_i = solver_wrapper(solver_func,simParams,sys_input_signal,sys_param=sys_params)

            # f_obsy.append(f_obs_i)
    # else:
        # f_obsy = solver_wrapper(solver_func,simParams,sys_in,sys_param=sys_params)
    # return np.array(f_obsy)

