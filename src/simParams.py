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
## Define parameters class
class simParameter(object):
    """
    Define general settings to simulation running and post data analysis. Solver parameters will be different between solver and solver

    General options:
        T,dt: total simulation time, time step
        stats: list, indicator of statistics to be calculated, [mean, std, skewness, kurtosis, absmax, absmin, up_crossing]
        outfile_name, outdir_name: string, file name and directory used to save output data.  

        qoi2analysis: list, rows of the output data from solver to be analyzed. 
        pts: int or a list of percentiles for QUANT
    """
    def __init__(self,site,doe_method,doe_order,qoi2analysis=[0,],time_start=0,time_ramp=0,time_max=1000,dt=0.1,stats=[1,1,1,1,1,1,0]):
        self.site       = site
        self.time_start = time_start
        self.time_ramp  = time_ramp 
        self.time_max   = time_max
        self.dt         = dt
        self.stats      = stats
        self.seed       = [0,100]
        self.doe_method = doe_method 
        self.doe_order  = []
        if not isinstance(doe_order, list):
            self.doe_order.append(int(doe_order))
        else:
            self.doe_order = list(int(x) for x in doe_order)

        self.qoi2analysis = qoi2analysis  ## default for single output solver
        self.nsamples_done = 0
        self.outdir_name = ''
        self.outfile_name= ''
 
    def set_doe_method(self,doe_method):
        self.doe_method = doe_method

    def set_doe_order(self,p):
        self.doe_order = p

    def add_nsamples_done(self,n):
        self.nsamples_done += n

    def set_seed(self, s):
        self.seed = s

    def update_filename(self,filename):
        self.outfile_name = filename

    def update_dir(self, newDir):
        self.outdir_name = newDir

    def set_qoi2analysis(self, qois):
        # what is thsi??
        self.qoi2analysis = qois

    def set_nsamples_need(self, n):
        self.nsamples_need= n
        self.doe_method = xrange(self.nsamples_need)


