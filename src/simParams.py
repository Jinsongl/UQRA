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
    def __init__(self,dist_zeta, doe_params=['GQ','lag',[2,3]], time_params=[0,0,1000,0.1], \
            post_params=[[0,], [1,1,1,1,1,1,0]], sys_params=None, sys_source=None, normalize=False):

        self.seed       = [0,100]
        self.dist_zeta  = dist_zeta
        self.distJ      = dist_zeta if len(dist_zeta) == 1 else cp.J(*dist_zeta)

        self.doe_method, self.doe_rule, self.doe_order = doe_params[0], doe_params[1], []
        if not isinstance(doe_params[2], list):
            self.doe_order.append(int(doe_params[2]))
        else:
            self.doe_order = list(int(x) for x in doe_params[2])
        self.ndoe = len(self.doe_order)
        self.nsouce_dim = 0
        self.nsets_per_doe = []

        self.time_start, self.time_ramp, self.time_max, self.dt = time_params
        self.qoi2analysis, self.stats = post_params
        self.normalize  = normalize 
        self.nsamples_done = 0
        self.outdir_name = ''
        self.outfile_name= ''

        # if normalize:
            # self._normalize_sys()
        # else:
        self.sys_source = sys_source
        self.sys_params = sys_params

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
            samp_zeta = samplegen(self.doe_method, idoe_order, self.distJ,rule=self.doe_rule)
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

            if self.doe_method.upper() in ['QUAD', 'GQ']:
                self.nsouce_dim = samp_zeta[0].shape[0] 
                self.nsets_per_doe.append(samp_zeta[0].shape[1])
            else:
                self.nsouce_dim = samp_zeta.shape[0] 
                self.nsets_per_doe.append(samp_zeta.shape[1])

        self.sys_source.append(doe_samples_phy) if retphy else self.sys_source.append(doe_samples_zeta)

        return doe_samples_zeta, doe_samples_phy if retphy else doe_samples_zeta

    # def _normalize_sys(self):
        # pass
    
    # def __normalize_source_func(self, norm_t, norm_y):

        # def wrapper(*args, **kwargs):
            # t, y = source_func(*args, **kwargs)
            # return  t*norm_t, y*norm_y/ norm_t**2
        # return wrapper 


    # def _cal_norm_values(self, zeta,omega0,source_kwargs, *source_args):
        # TF = lambda w : 1.0/np.sqrt((w**2-omega0**2)**2 + (2*zeta*omega0)**2)
        # spec_dict = spec_coll.get_spec_dict() 
        # spec_name = source_kwargs.get('name','JONSWAP') if source_kwargs else 'JONSWAP'
        # spec_side = source_kwargs.get('sides', '1side') if source_kwargs else '1side'
        # spec_func = spec_dict[spec_name]

        
        # nquads = 100
        # if spec_side.upper() in ['2','2SIDE','DOUBLE','2SIDES']:
            # x, w = np.polynomial.hermite_e.hermegauss(nquads)
        # elif spec_side.upper() in ['1','1SIDE','SINGLE','1SIDES']:
            # x, w = np.polynomial.laguerre.laggauss(nquads)
        # else:
            # raise NotImplementedError("Spectrum side type '{:s}' is not defined".format(spec_side))
        # _,spec_vals = spec_func(x, *source_args)
        # spec_vals = spec_vals.reshape(nquads,1)
        # TF2_vals  = (TF(x)**2).reshape(nquads,1)
        # w = w.reshape(nquads,1)
        # norm_y = np.sum(w.T *(spec_vals*TF2_vals))/(2*np.pi)
        # norm_y = 1/norm_y**0.5

        # norm_t = omega0

        # return norm_t, norm_y

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
