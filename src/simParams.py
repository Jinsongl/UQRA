#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import numpy as np
from doe.doe_generator import samplegen
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
    def __init__(self,doe_method,doe_order,dist_zeta,qoi2analysis=[0,],time_start=0,time_ramp=0,time_max=1000,dt=0.1,stats=[1,1,1,1,1,1,0]):
        self.time_start = time_start
        self.time_ramp  = time_ramp 
        self.time_max   = time_max
        self.dt         = dt
        self.stats      = stats
        self.seed       = [0,100]
        self.doe_method = doe_method 
        self.doe_order  = []
        self.dist_zeta  = dist_zeta
        self.distJ      = dist_zeta if len(dist_zeta) == 1 else cp.J(*dist_zeta)
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
    def get_doe_samples(self, retphy=False, dist_phy=None):
        """
        Return design sample points both in physical and zeta spaces based on specified doe_method
        Return:
            Experiment samples at zeta and physical space (idoe_order,[samples])
        """
        doe_samples_zeta = []

        if retphy:

            doe_samples_phy = []
            assert dist_phy, 'To return samples in physical space, list of marginal distributions of physical random variables must be provided.'
            assert len(dist_phy) == len(self.dist_zeta)

        for idoe_order in self.doe_order: 
           # ED for different selected doe_order 
            samp_zeta = samplegen(self.doe_method, idoe_order, self.distJ)
            doe_samples_zeta.append(samp_zeta)

            # Transform input sample values from zeta space to physical space
            if retphy:

                if self.doe_method.upper() in ['QUAD', 'GQ']:
                    zeta_cor, zeta_weights = samp_zeta
                    phy_cor = self._dist_transform(self.dist_zeta, dist_phy, zeta_cor)
                    phy_weights = zeta_weights
                    samp_phy = np.array([phy_cor, phy_weights])
                else:
                    zeta_cor, zeta_weights = samp_zeta, None
                    phy_cor = self._dist_transform(self.dist_zeta, dist_phy, zeta_cor)
                    samp_phy = phy_cor

                doe_samples_phy.append(samp_phy)
        return doe_samples_zeta, doe_samples_phy if retphy else doe_samples_zeta



    def _dist_transform(self,dist1, dist2, var1):
        """
        Transform variables (var1) from list of dist1 to correponding variables in dist2. based on same icdf

        Arguments:
            dist1: list of independent distributions (source)
            dist2: list of independent distributions (destination)
            var1 : variables in dist1 of shape[ndim, nsamples]

        Return:
            
        """
        var1 = np.array(var1)

        dist1  = [dist1,] if len(dist1) == 1 else dist1
        dist2  = [dist2,] if len(dist2) == 1 else dist2
        assert len(dist1) == len(dist2) == var1.shape[0]
        var2 = []
        for i in range(var1.shape[0]):
            _dist1 = dist1[i]
            _dist2 = dist2[i]
            _var = _dist2.inv(_dist1.cdf(var1[i,:]))
            var2.append(_var)
        var2 = np.array(var2)
        assert var1.shape == var2.shape
        return var2
