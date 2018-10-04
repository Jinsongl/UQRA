#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
def run_sim(siteEnvi, solver, simParams, metaParams):
    """
    Run simulation with given "solver" (real model) and predefined environment
    Inputs:
        siteEnvi: environment class object
        simParams: simParameter class object
        solver: solver. Return [nChannel, nTimeSample] array. Each row is a time series for specified channel
    Return:
        inputVars: (nsamples * ndim), each sample including [zeta_0,...zeta_n, phy_0,...,phy_n] 
        outputStats:(nchannels * nsamples * nstats)

    """
    numRows     = simParams.numRows
    scheme      = simParams.scheme 
    schemePts   = simParams.schemePts
    outfileName = simParams.outfileName
    outdirName  = simParams.outdirName
    outputStats = [[] for _ in range(numRows)] # each element store the statistics for one output channel 
    # for i in numRows[:-1]:
        # outputStats.append([])

    inputVars   = []
    for schemePt in schemePts: 
        samp_zeta = genVar.genVar(schemePt,simParams, metaParams)
        if scheme == "QUAD":
            zeta_cor, zeta_weights = samp_zeta[:-1,:], samp_zeta[-1,:]
        else:
            zeta_cor = samp_zeta

        args = siteEnvi.zeta2phy(metaParams.dist,zeta_cor).T.tolist()
        for i, arg in enumerate(args):
            inputVars.append(samp_zeta[:,i].T.tolist() + arg)
            f_obs   = solver(simParams, *arg)
            f_stats = getStats.getStats(f_obs, stats=simParams.stats)
            for j in xrange(numRows):
                outputStats[j].append(f_stats[j,:])
    outputStats = np.array(outputStats)
    inputVars   = np.array(inputVars)
    # Save inputs 
    simParams.updateFilename(simParams.scheme + 'Inputs')
    filenames = dataIO.setfilename(simParams)
    dataIO.saveData(inputVars, filenames[0], simParams.outdirName)
    # Save outputs
    simParams.updateFilename(simParams.scheme + 'Outputs')
    filenames = dataIO.setfilename(simParams)
    for i in xrange(numRows):
        dataIO.saveData(outputStats[i],filenames[i+1], simParams.outdirName)
        
    return inputVars, outputStats



