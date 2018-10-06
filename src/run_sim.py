#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
def run_sim(siteEnvi, solver_func, simParams, metaParams):
    """
    Run simulation with given "solver" (real model) and predefined environment
    Inputs:
        siteEnvi: environment class object
        simParams: simParameter class object
        solver: solver. Return [nChannel, nTimeSample] array. Each row is a time series for specified channel
    Return:
        samples_rvs: (ndim * nsamples), each sample including [zeta_0,...zeta_n, phy_0,...,phy_n].T
        output_stats:(nchannels * nsamples * nstats)

    """
    qoi2analysis= simParams.qoi2analysis
    doe_method  = simParams.doe_method
    doe_order   = simParams.doe_order
    outfile_name= simParams.outfile_name
    outdir_name = simParams.outdir_name
    output_stats= [] 
    #[[] for _ in range(qoi2analysis)]#each element store the statistics for one output channel 
    # for i in qoi2analysis[:-1]:
        # output_stats.append([])

    samples_rvs   = []
    for idoe_order in doe_order: 
        
        samp_zeta = samplegen(doe_method, idoe_order, metaParams.distJ)

        if doe_method.upper() in ['QUAD', 'GQ']:
            zeta_cor, zeta_weights = samp_zeta
        else:
            zeta_cor = samp_zeta

        phyrvs = siteEnvi.zeta2phy(metaParams.dist,zeta_cor)
        # idoe_order_stats (iphyrvs, istats, iqoi2analysis)
        idoe_order_stats = np.empty(phyrvs.shape[1], sum(simParams.stats), len(simParams.qoi2analysis))
        for i, iphyrvs in enumerate(phyrvs.T):
            samples_rvs.append(samp_zeta[:,i].T.tolist() + iphyrvs.tolist())
            ## Need to implement
            f_obs   = solver_wrapper(solver_func, simParams, *iphyrvs)
            # f_obs   = solver(simParams, *iphyrvs)
            f_stats = get_stats.get_stats(f_obs[:,qoi2analysis], stats=simParams.stats)
            idoe_order_stats[i,:,:] = f_stats
        output_stats.append(idoe_order_stats)
    output_stats = np.asfarray(output_stats)
    samples_rvs  = np.asfarray(samples_rvs)
    # Save inputs 
    simParams.updateFilename(simParams.doe_method + 'Inputs')
    filenames = dataIO.setfilename(simParams)
    dataIO.saveData(samples_rvs, filenames[0], simParams.outdirName)
    # Save outputs
    simParams.updateFilename(simParams.doe_method + 'Outputs')
    filenames = dataIO.setfilename(simParams)
    for i in xrange(qoi2analysis):
        dataIO.saveData(output_stats[i],filenames[i+1], simParams.outdirName)
        
    return samples_rvs, output_stats



