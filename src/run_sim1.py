#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import chaospy as cp
import doe
from doe.doe_generator import samplegen
from solver.solver_wrapper import solver_wrapper
from utilities.get_stats import get_stats
from utilities.gen_gauss_time_series import gen_gauss_time_series
import numpy as np
from envi.environment import environment

import sys

def run_sim(solver_func, vars_in, simParams, sys_params=None, psd_params=None):
    """
    Run simulation with given "solver" (real model) and predefined environment
    Inputs:
        solver_func: solver. Return [nChannel, nTimeSample] array. Each row is a time series for specified channel
        vars_in: input variables in physical space. Will have [cor, weights] if quadrature method used,
        simParams: simParameter class object
    Optional:
        sys_params: parameters for solver system
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

    f_obsy     = []

    if simParams.doe_method.upper() in ['QUAD', 'GQ']:
        phy_cor, phy_weights = vars_in
    else:
        phy_cor = vars_in

    if psd_params:
        for iphy_cor in phy_cor.T:

            tmax, dt = simParams.time_max, simParams.dt

            psd_name    = psd_params['name']
            psd_method  = psd_params['method']
            psd_sides   = psd_params['sides']

            t, sys_input_signal = gen_gauss_time_series(tmax, dt, psd_name, iphy_cor,\
                    method=psd_method, sides=psd_sides)

            f_obs_i = solver_wrapper(solver_func,simParams,sys_input_signal,sys_param=sys_params)

            f_obsy.append(f_obs_i)
    else:
        f_obsy = solver_wrapper(solver_func,simParams,phy_cor,\
                sys_param=sys_params)

    return np.array(f_obsy)


    # for ivar_input in vars_in: 
       # # ED for different selected doe_order 


        # # #1 Run solver for each iphyrvs(input variable in physical space) and get stats



        # # if solver return is just a number or (1, nqoi) or (nsamples, 1), vectorization could be used
        # # otherwise, solver need to run for input variables one by one
        # if_obs = solver_wrapper(solver_func, simParams, phy_cor.T[0])
        # solvery_shape = None if np.isscalar(if_obs) else if_obs.shape
        # print(solvery_shape)

        # # isvectorization = 1 if np.isscalar(if_obs) or if_obs.shape[0] == 1 or if_obs.shape[1] == 1 else 0 

        # if np.isscalar(if_obs):
            # isvectorization = 1
        # else:
            # m,n = solvery_shape
            # if m == 1 or n == 1:
                # isvectorization = 1
            # else:
                # isvectorization = 0
        # if np.isscalar(if_obs):
            # isreturnstats = 0
        # else:
            # m,n = if_obs.shape
            # if n == 1:
                # isreturnstats = 0
            # else:
                # isreturnstats = 1

        # if isvectorization:
            # if_obs = solver_wrapper(solver_func, simParams, phy_cor)
            # if solvery_shape is None:
                # if_obs = if_obs.reshape(np.prod(if_obs.shape), 1)
            # else:
                # newshape = (phy_cor.shape[1], *solvery_shape)
                # assert np.prod(if_obs.shape) == np.prod(newshape)
                # if_obs = if_obs.reshape(newshape)
            # if isreturnstats:
                # if_stats = get_stats(if_obs[:,:, qoi2analysis], stats = simParams.stats)
                # f_obsy_stats.append(np.array(if_stats))
            # f_obsy.append(np.array(if_obs))
        # else:

            # _f_obsy = []
            # # return from each solver run must be full matrix of shape(ntimeseries, nqoi)
            # for i, iphyrvs in enumerate(phy_cor.T):
                # if_obs   = solver_wrapper(solver_func, simParams, *iphyrvs)
                # _f_obsy.append(if_obs[:,qoi2analysis])
                # print('\r\tRunning solver: {:s}, {:d} out of {:d}'\
                        # .format(solver_func.__name__, i+1, phy_cor.shape[1]), end='')
            # print('\n')
            # _f_obsy_stats = get_stats(_f_obsy, stats=simParams.stats)
            # f_obsy.append(np.array(_f_obsy))
            # f_obsy_stats.append(np.array(_f_obsy_stats))
        # # Convert output to ndarray of dtype float
        # if zeta_weights is None:
            # f_obsx.append(np.vstack((zeta_cor, phy_cor)))
        # else:
            # f_obsx.append(np.vstack((zeta_cor, zeta_weights, phy_cor)))


    # Save inputs 
    # simParams.update_filename(simParams.doe_method + 'Inputs')
    # filenames = dataIO.setfilename(simParams)
    # dataIO.saveData(samples_rvs, filenames[0], simParams.outdirName)
    # # Save outputs
    # simParams.update_filename(simParams.doe_method + 'Outputs')
    # filenames = dataIO.setfilename(simParams)
    # for i in xrange(qoi2analysis):
        # dataIO.saveData(f_obsy_stats[i],filenames[i+1], simParams.outdirName)
    # assert len(f_obsx) == len(f_obsy)  == len(doe_order)
    # assert isreturnstats == len(f_obsy_stats) 

    # return (f_obsx, f_obsy, f_obsy_stats) if isreturnstats else (f_obsx, f_obsy)


# def transform_zeta2phy(dist_phy, dist_zeta, zeta):
    # """
    # Transform zeta sample values from zeta space (slected Wiener-Askey scheme) to physical space

    # Arguments:
        # dist_phy: list of marginal distributions of physical random variables,
            # distributions must be mutually independent
        # dist_zeta: list of marginal distributions of zeta random variables
        # zeta: samples in zeta space of shape(ndim, nsamples)

    # Return:
        # phy: same shape as zeta, samples in physical space
    # """

    # print('Transforming samples from Wiener-Askey scheme to physical space')
    # print('   marginal physical distributions are assumed independent')
    # assert len(dist_phy) == len(dist_zeta)
    # if len(dist_phy) == 1:
        # dist_phy = [dist_phy,]
        # dist_zeta = [dist_zeta,]

    # zeta = np.asfarray(zeta)
    # vals = np.empty(zeta.shape)
    # assert len(dist_zeta) == zeta.shape[0]
    # for i, val in enumerate (zeta.T):
        # q = list(map(lambda dist, x: float(dist.cdf(x)), dist_zeta, val))
        # iphy = list(map(lambda dist, x: float(dist.inv(x)), dist_phy, q))
        # vals[:,i] = np.array(iphy) 
    # assert vals.shape == zeta.shape
    # return vals




