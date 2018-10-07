#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
from doe_generator import samplegen
from solver_wrapper import solver_wrapper
from utility.get_stats import get_stats
import numpy as np

import sys

def run_sim(siteEnvi, solver_func, simParams, metaParams):
    """
    Run simulation with given "solver" (real model) and predefined environment
    Inputs:
        siteEnvi: environment class object
        simParams: simParameter class object
        solver: solver. Return [nChannel, nTimeSample] array. Each row is a time series for specified channel
    Return:
        for different doe_order, sample size would change, so return list

        f_obsx: list of length len(doe_order)
            each element contains input sample variables both in zeta and physical space
            of shape (ndim, nsamples), each column includes [zeta_0,...zeta_n[,w], phy_0,...,phy_n].T, 
            weight is optional

        f_obsy: list of length len(doe_order)
            each element contains solver output for certain amount of samples corresponding to doe_order
        
        f_obsy_stats: list of length len(doe_order)
            each element contains (nsamples, nstats, nqois)

    """
    qoi2analysis= simParams.qoi2analysis
    doe_method  = simParams.doe_method
    doe_order   = simParams.doe_order
    outfile_name= simParams.outfile_name
    outdir_name = simParams.outdir_name
    f_obsx     = []
    f_obsy     = []
    f_obsy_stats = [] 

    for idoe_order in doe_order: 
       # ED for different selected doe_order 
        samp_zeta = samplegen(doe_method, idoe_order, metaParams.distJ)

        if doe_method.upper() in ['QUAD', 'GQ']:
            zeta_cor, zeta_weights = samp_zeta
        else:
            zeta_cor, zeta_weights = samp_zeta, None
        # Transform input sample values from zeta space to physical space
        phyrvs = siteEnvi.zeta2phy(metaParams.dist,zeta_cor)
        _f_obsy = []
        _f_obsy_stats = []
        # idoe_order_stats = np.empty((phyrvs.shape[1], sum(simParams.stats), len(simParams.qoi2analysis)))
        # run solve for each iphyrvs(input variable in physical space) and get stats
        for i, iphyrvs in enumerate(phyrvs.T):
            if_obs   = solver_wrapper(solver_func, simParams, *iphyrvs)
            if_stats = get_stats(if_obs[:,qoi2analysis], stats=simParams.stats)
            _f_obsy.append(if_obs)
            _f_obsy_stats.append(if_stats)
            # idoe_order_stats[i,:,:] = if_stats
            print('\r\tRunning solver: {:s}, {:d} out of {:d}'\
                    .format(solver_func.__name__, i+1, phyrvs.shape[1]), end='')
        print('\n')
        f_obsy.append(np.array(_f_obsy))
        f_obsy_stats.append(np.array(_f_obsy_stats))
        # Convert output to ndarray of dtype float
        if zeta_weights is None:
            f_obsx.append(np.vstack((zeta_cor, phyrvs)))
        else:
            f_obsx.append(np.vstack((zeta_cor, zeta_weights, phyrvs)))


    # Save inputs 
    # simParams.update_filename(simParams.doe_method + 'Inputs')
    # filenames = dataIO.setfilename(simParams)
    # dataIO.saveData(samples_rvs, filenames[0], simParams.outdirName)
    # # Save outputs
    # simParams.update_filename(simParams.doe_method + 'Outputs')
    # filenames = dataIO.setfilename(simParams)
    # for i in xrange(qoi2analysis):
        # dataIO.saveData(f_obsy_stats[i],filenames[i+1], simParams.outdirName)
    assert len(f_obsx) == len(f_obsy) == len(f_obsy_stats)  == len(doe_order)
    return (f_obsx, f_obsy, f_obsy_stats)



