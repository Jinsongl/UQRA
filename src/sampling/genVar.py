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

def genVar(schemePt, simParams, metaParams):
    """
    Generating samples from zeta space  underlying random variables for selected Wiener-Askey polynomial.
        simParams: simParameter() object, scheme, schemePt, rule will be provided

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
    scheme = simParams.scheme

    #########################################################################
    ###  Quandrature Rule 
    #########################################################################
    if scheme.upper() == "QUAD":
        if simParams.rule is None:
            simParams.setRule('G')
        # simParams.rule = "G" if simParams.rule is None else simParams.rule
        # if isSave and filename is None:
            # filename,NEWDIR = genfilename(scheme,numPts,rule)
        print '************************************************************'
        print "Generating Quadrature samples..."
        print "Rule: " + str(simParams.rule) + ", Number of points: " + str(schemePt) 
        zeta_cor, zeta_weights = cp.generate_quadrature(schemePt-1, distJ_zeta, rule=simParams.rule) 
        zeta_weights = zeta_weights.reshape((1,len(zeta_weights)))
        zeta_cor = np.concatenate((zeta_cor,zeta_weights), axis=0)
        simParams.AddSamplesDoneNum(int(zeta_cor.shape[1])) 

        print "... Sampling completed. Total number:" + str(simParams.nsamplesDone)


    #########################################################################
    ###  Quantile (zeta space ) based
    #########################################################################
    elif scheme.upper() == "QUANT":
        """
        QUANT method generate variables in zeta space based on given percentile
        """
        if simParams.rule is None:
            simParams.setRule('Permu')
        string = 'Number of points at each dimension:'
        varDim = []
        quantile_cor = []
        for var in schemePt:
            varDim.append(len(var))
            quantile_cor.append(np.arange(len(var)))
        quantile_cor = itertools.product(*quantile_cor)

        print '************************************************************'
        print "Generating Quantile Based samples..."
        print "Dimensions: ", len(varDim), "Rule: ", str(simParams.rule)
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
        simParams.AddSamplesDoneNum(int(zeta_cor.shape[1])) 

        print "... Sampling completed. Total number:" + str(simParams.nsamplesDone)


    #########################################################################
    ###  Monte Carlo Sampling
    #########################################################################
    elif scheme.upper() == "MC":
        """
        Monte Carlo Points are generated one by one by design, avoiding possible large memory requirement
        """
        if simParams.rule is None:
            simParams.setRule('R')
        zeta_cor = distJ_zeta.sample(1,rule=simParams.rule)
        simParams.AddSamplesDoneNum(int(zeta_cor.shape[1])) 
        if simParams.nsamplesDone == 1:
            print '************************************************************'
            print "Generating Monte Carlo samples..."
            print "Rule: " + str(simParams.rule) + ", Number of points: " + '{:2.1E}'.format(simParams.nsamplesNeed)
        if simParams.nsamplesDone % int(simParams.nsamplesNeed/10) == 0:
            print "     > generating:   " + '{:2.1E}'.format(simParams.nsamplesDone)



    #########################################################################
    ###  Experimental Design 
    #########################################################################
    elif scheme.upper() == "ED":
        """
        schemePt could be generalized, if schemePt is a number, then "rule" will be used
        if schemePt is specified points, schemePt will be used and rule will be ignored
        """
        if simParams.rule is None:
            simParams.setRule('L')

        print '************************************************************'
        print "Generating Experimental Design samples..."
        print "Rule: " + str(simParams.rule) + ", Number of points: " + str(schemePt) 
        zeta_cor = distJ_zeta.sample(int(schemePt),rule=simParams.rule)
        # print zeta for:_cor
        simParams.AddSamplesDoneNum(int(zeta_cor.shape[1])) 
        print "... Sampling completed. Total number:" + str(simParams.nsamplesDone) 


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
