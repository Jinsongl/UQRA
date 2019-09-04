#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""


"""
import itertools
import chaospy as cp
import numpy as np
import os

def genVar(schemePt, simParameters, metaParams):
    """
    Generating samples from zeta space (underlying random variables for selected Wiener-Askey polynomial).
        simParameters: simParameters() object, scheme, schemePt, rule will be provided

            (QUAD, schemePt, rule): 
                QUAD: Quadrature rule,  
                schemePt: number of quadrature points in one dimension,
                rule: rules to generate quadrature points. Refer to chaospy for details. Default: G
                
            (QUANT, schemePt, rule): 
                QUANT: Sampling with predifined quantiles, 
                schemePt: list of specified arbitrary quantiles, number of points don't need to be the same at different dimension.
                rule:  

            (MC, schemePt, rule):
                MC: Monte Carlo method or its derivatives
                schemePt: number of points in Monte Carlo
                rule: rules to generate MC samples, including Random, Latin Hypercube and other quasi-monte carlo methods. 
                      Refer to chaospy for details, default is R

            (ED, schemePt, rule):
                ED: Experimental Design. Generate points based on specified rule, like Latin Hypercube
                schemePt: number of points for each dimension 
                rule:

        metaParams: metaParameter() object, dist_zeta will be provided
    """

    ####  Create Joint distribution for marginal distributions  
    dists      = metaParams.dist
    distJ_zeta = metaParams.distJ
    scheme = simParameters.scheme

    #########################################################################
    ###  Quandrature Rule 
    #########################################################################
    if scheme.upper() == "QUAD":
        if simParameters.rule is None:
            simParameters.setRule('G')
        # simParameters.rule = "G" if simParameters.rule is None else simParameters.rule
        # if isSave and filename is None:
            # filename,NEWDIR = genfilename(scheme,numPts,rule)
        print '************************************************************'
        print "Generating Quadrature samples..."
        print "Rule: " + str(simParameters.rule) + ", Number of points: " + str(schemePt) 
        zeta_cor, zeta_weights = cp.generate_quadrature(schemePt-1, distJ_zeta, rule=simParameters.rule) 
        zeta_weights = zeta_weights.reshape((1,len(zeta_weights)))
        zeta_cor = np.concatenate((zeta_cor,zeta_weights), axis=0)
        simParameters.AddSamplesDoneNum(int(zeta_cor.shape[1])) 

        print "... Sampling completed. Total number:" + str(simParameters.nsamplesDone)


    #########################################################################
    ###  Quantile (zeta space ) based
    #########################################################################
    elif scheme.upper() == "QUANT":
        """
        QUANT method generate variables in zeta space based on given percentile
        """
        if simParameters.rule is None:
            simParameters.setRule('Permu')
        string = 'Number of points at each dimension:'
        varDim = []
        quantile_cor = []
        for var in schemePt:
            varDim.append(len(var))
            quantile_cor.append(np.arange(len(var)))
        quantile_cor = itertools.product(*quantile_cor)

        print '************************************************************'
        print "Generating Quantile Based samples..."
        print "Dimensions: ", len(varDim), "Rule: ", str(simParameters.rule)
        print string, varDim

        zeta_cor=[]
        for samp in list(quantile_cor):
            v = []
            for i in range(len(varDim)):
                dist = dists[i]
                quantles = schemePt[i]
                j = samp[i]
                val = dist.inv(quantles[j])
                v.append(float(val))
            zeta_cor.append(v)

        zeta_cor = np.array(zeta_cor).T
        simParameters.AddSamplesDoneNum(int(zeta_cor.shape[1])) 

        print "... Sampling completed. Total number:" + str(simParameters.nsamplesDone)


    #########################################################################
    ###  Monte Carlo Sampling
    #########################################################################
    elif scheme.upper() == "MC":
        """
        Monte Carlo Points are generated one by one by design, avoiding possible large memory requirement
        """
        if simParameters.rule is None:
            simParameters.setRule('R')
        zeta_cor = distJ_zeta.sample(1,rule=simParameters.rule)
        simParameters.AddSamplesDoneNum(int(zeta_cor.shape[1])) 
        if simParameters.nsamplesDone == 1:
            print '************************************************************'
            print "Generating Monte Carlo samples..."
            print "Rule: " + str(simParameters.rule) + ", Number of points: " + '{:2.1E}'.format(simParameters.nsamplesNeed)
        if simParameters.nsamplesDone % int(simParameters.nsamplesNeed/10) == 0:
            print "     > generating:   " + '{:2.1E}'.format(simParameters.nsamplesDone)



    #########################################################################
    ###  Experimental Design 
    #########################################################################
    elif scheme.upper() == "ED":
        """
        schemePt could be generalized, if schemePt is a number, then "rule" will be used
        if schemePt is specified points, schemePt will be used and rule will be ignored
        """
        if simParameters.rule is None:
            simParameters.setRule('L')

        print '************************************************************'
        print "Generating Experimental Design samples..."
        print "Rule: " + str(simParameters.rule) + ", Number of points: " + str(schemePt) 
        zeta_cor = distJ_zeta.sample(int(schemePt),rule=simParameters.rule)
        # print zeta for:_cor
        simParameters.AddSamplesDoneNum(int(zeta_cor.shape[1])) 
        print "... Sampling completed. Total number:" + str(simParameters.nsamplesDone) 


    else:
        raise NotImplementedError("scheme not defined")
    return zeta_cor

def getQuant(size,ndim = 2.0, base=2.0, alpha=1):
    size1D = (int(size**(1.0/ndim)) + 1) 
    dist = cp.Uniform()
    unisamp = dist.sample(size1D, rule='L') 
    unisamp.sort()
    stop = int(np.log(size1D)/ np.log(base))  + 1
    endInd = np.logspace(0,stop,num = stop+1, base=base, dtype=int)[:-1]
    q = []
    for i,ind in enumerate(endInd):
        if i == 0:
            q.append(unisamp[0])
            q.append(unisamp[-1])
        else:
            delta = int(-endInd[i-1] + ind )/alpha
            j = endInd[i-1]
            for k in xrange(int(alpha)):
                j += delta 
                q.append(unisamp[j])
                q.append(unisamp[-j-1])
    return np.array(q)
