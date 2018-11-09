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

def run_sim(solver_func, simParams,sys_source, *args):
    """
    Run simulation with given "solver" (real model) and predefined environment
    Arguments:
        solver_func: solver. Return [nChannel, nTimeSample] array. Each row is a time series for specified channel
        simParams: simParameter class object
        args: 
    Optional:
        sys_param: parameters for solver system
        solver_initconds: initial conditions

    Return:
        zafor different doe_order, sample size would change, so return list

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
    if isinstance(sys_source, np.ndarray):
        print('Source parameters: signal of shape {}'.format(sys_source.shape))
        ## only need inner loop for changing physical system
        res_inner = _run_fixsource_loop(solver_func, simParams, sys_source, sys_param)
        resall.append(*res_inner)

    elif isinstance(sys_source, list) and callable(sys_source[0]):
        print('Source parameters:')
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
        for source_args_1doe in source_args:
            sys_source_1doe = [source_func, source_args_1doe, source_kwargs]
            y = _run_fixdoeorder_loop(solver_func, simParams, sys_source_1doe,sys_param) 
            resall.append(*y)

    return np.array(resall)
           

def _run_fixsource_loop(solver_func,simParams,sys_source_fixed,sys_param_all):
    res = []
    ## system parameters don't change
    if sys_param_all is None:
        y = solver_wrapper(solver_func,simParams,sys_source_fixed)
        res.append(y)
    ## different sets of system parameters
    else:
        for _ , isys_param in enumerate(sys_param_all.T):
            y = solver_wrapper(solver_func,simParams,sys_source_fixed,sys_param=isys_param)
            res.append(y)
    return res

def _run_fixdoeorder_loop(solver_func,simParams,sys_source_1doe, sys_param_all):
    res = []
    source_func, source_args, source_kwargs = sys_source_1doe
    if simParams.doe_method.upper() in ['QUAD', 'GQ']:
        source_args, source_args_weights = source_args 
    else:
        source_args, source_args_weights = source_args, None
    ## source_args must of shape(ndim, nsampes)
    for isource_args in source_args.T:
        isys_source = [source_func, isource_args, source_kwargs]
        res_inner = _run_fixsource_loop(solver_func, simParams, isys_source, sys_param_all)
        res.append(*res_inner)

    return res


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

