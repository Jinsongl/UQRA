#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import chaospy as cp
import numpy as np
import sampling.genVar as genVar
import utility.dataIO as dataIO
import utility.getStats as getStats
## Define parameters class
class simParameter(object):
    """
    Define parameters needed to run "solver". Different between solver and solver
    General options:
        T,dT: total simulation time, time step
        stats: list, indicator of statistics to be calculated, [mean, std, skewness, kurtosis, absmax, absmin, up_crossing]
        numRows: list, rows of the output data from solver to be analyzed. 
        pts: int or a list of percentiles for QUANT
        outfileName, outdirName: string, file name and directory used to save output data.  
    """
    def __init__(self,site,scheme,pts,T=1000,dT=0.1,stats=[1,1,1,1,1,1,0]):
        self.site       = site
        self.T          = T
        self.dT         = dT
        self.stats      = stats
        self.rule       = None
        self.seed       = [0,100]
        self.numRows    = 100
        self.scheme     = scheme 
        if self.scheme == 'QUANT':
            self.schemePts = pts
        elif self.scheme == 'MC':
            self.nsamplesNeed = int(pts)
            self.schemePts = xrange(self.nsamplesNeed)
        elif self.scheme == 'QUAD' or self.scheme == 'ED':
            self.schemePts = [pts,]
        else:
            raise ValueError('scheme not defined')
        self.nsamplesDone = 0
        # self.nsamplesNeed = 0
        self.outdirName = ''
        self.outfileName= ''

    def setRule(self,newrule):
        self.rule = newrule

    def setScheme(self,scheme):
        self.scheme = scheme

    def setSchemePts(self,p):
        self.schemePts = p

    def AddSamplesDoneNum(self,n):
        self.nsamplesDone += n

    def setSeed(self, s):
        self.seed = s

    def updateFilename(self,filename):
        self.outfileName = filename

    def updateDir(self, newDir):
        self.outdirName = newDir

    def setNumOutputs(self, n):
        self.numRows = n

    def setNsamplesNeed(self, n):
        self.nsamplesNeed= n
        self.scheme = xrange(self.nsamplesNeed)



class Data(object):
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y  

def RunSim(siteEnvi, solver, simParams, metaParams):
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



