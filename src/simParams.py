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
import utility.dataIO as dataIO
import utility.getStats as getStats
## Define parameters class
class simParameter(object):
    """
    Define general settings to simulation running and post data analysis. Solver parameters will be different between solver and solver

    General options:
        T,dT: total simulation time, time step
        stats: list, indicator of statistics to be calculated, [mean, std, skewness, kurtosis, absmax, absmin, up_crossing]
        outfileName, outdirName: string, file name and directory used to save output data.  

        numRows: list, rows of the output data from solver to be analyzed. 
        pts: int or a list of percentiles for QUANT
    """
    def __init__(self,site,doe_method,pts,T=1000,dT=0.1,stats=[1,1,1,1,1,1,0]):
        self.site       = site
        self.T          = T
        self.dT         = dT
        self.stats      = stats
        self.rule       = None ## Need to be specific which rule ? sampling? quadrature? etc..
        self.seed       = [0,100]
        self.numRows    = 100
        self.doe_method     = doe_method 
        if self.doe_method == 'QUANT':
            self.doe_methodPts = pts
        elif self.doe_method == 'MC':
            self.nsamplesNeed = int(pts)
            self.doe_methodPts = xrange(self.nsamplesNeed)
        elif self.doe_method == 'QUAD' or self.doe_method == 'ED':
            self.doe_methodPts = [pts,]
        else:
            raise ValueError('doe_method not defined')
        self.nsamplesDone = 0
        # self.nsamplesNeed = 0
        self.outdirName = ''
        self.outfileName= ''

    def setRule(self,newrule):
        self.rule = newrule

    def setScheme(self,doe_method):
        self.doe_method = doe_method

    def setSchemePts(self,p):
        self.doe_methodPts = p

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
        self.doe_method = xrange(self.nsamplesNeed)


