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
DOE_METHOD_NAMES = {
    "GQ"    : "Quadrature"  , "QUAD"  : "Quadrature",
    "MC"    : "Monte Carlo" , "FIX"   : "Fixed point"
    } 

QUAD_SHORT_NAMES = {
    "c": "clenshaw_curtis"  , "e"   : "gauss_legendre"  , "p"   : "gauss_patterson",
    "z": "genz_keister"     , "g"   : "golub_welsch"    , "j"   : "leja",
    "h": "gauss_hermite"    ,"lag"  : "gauss_laguerre"  , "cheb": "gauss_chebyshev",
    "hermite"   :"gauss_hermite",
    "legendre"  :"gauss_legendre",
    "jacobi"    :"gauss_jacobi",
    }
class simParameter(object):
    """
    Define general parameter settings for simulation running and post data analysis. 
    Solver parameters will be different between solver and solver

    Arguments:
        dist_zeta: list of selected marginal distributions from Wiener-Askey scheme
        *OPTIONAL:
        doe_params  = [doe_method, doe_rule, doe_order]
        time_params = [time_start, time_ramp, time_max, dt]
        post_params = [qoi2analysis=[0,], stats=[1,1,1,1,1,1,0]]
            stats: [mean, std, skewness, kurtosis, absmax, absmin, up_crossing]
        sys_params: list of parameter sets defining the solver
            if no sys_params is required, sys_params = [None]
            e.g. for duffing oscillator:
            sys_params = [np.array([0,0,1,1,1]).reshape(5,1)] # x0,v0, zeta, omega_n, mu 

        normalize: 

    """
    def __init__(self,dist_zeta, doe_params=['GQ','lag',[2,3]],\
            time_params=None, post_params=[[0,], [1,1,1,1,1,1,0]],\
            sys_params=[None], normalize=False):

        self.seed       = [0,100]
        self.dist_zeta  = dist_zeta
        self.distJ      = dist_zeta if len(dist_zeta) == 1 else cp.J(*dist_zeta) 

        self.doe_method, self.doe_rule, self.doe_order = doe_params[0], doe_params[1], []
        if np.isscalar(doe_params[2]): 
            self.doe_order.append(int(doe_params[2]))
            # self.doe_order = np.array(int(doe_params[2]))
        else:
            # self.doe_order = list(int(x) for x in doe_params[2])
            self.doe_order = np.array(doe_params[2])
        # print(self.doe_order)

        self.ndoe               = len(self.doe_order)   # number of doe sets
        self.nsys_input_vars_dim= 0                     # dimension of solver inputs 
        self.nsamples_per_doe   = []                    # number of samples for each doe sets
        
        self.time_start, self.time_ramp, self.time_max, self.dt = time_params if time_params else [None]*4
        self.qoi2analysis, self.stats = post_params
        self.normalize      = normalize 
        self.nsamples_done  = 0
        self.outdir_name    = ''
        self.outfile_name   = ''

        # if normalize:
            # self._normalize_sys()
        # else:

        self.get_doe_samples_zeta() 
        self.sys_params = sys_params
        self.sys_input_params = []

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
        # what is this??
        self.qoi2analysis = qois
    def set_sys_input_params(self, sys_input_func_name, sys_input_kwargs):
        self.sys_input_params.insert(0,sys_input_kwargs)
        self.sys_input_params.insert(0,sys_input_func_name)

    def set_nsamples_need(self, n):
        self.nsamples_need= n
        self.doe_method = xrange(self.nsamples_need)
    def get_doe_samples_zeta(self):
        """
        Return design sample points both in zeta spaces based on specified doe_method
        Return:
            Experiment samples at zeta space (idoe_order,[samples for each doe_order])
        """

        print('------------------------------------------------------------')
        print('►►► Design of Experiment with {} method'.format(DOE_METHOD_NAMES[self.doe_method]))
        print('------------------------------------------------------------')
        print(' ► Quadrature rule               : {:s}'.format(QUAD_SHORT_NAMES[self.doe_rule]))
        print(' ► Number of quadrature points   : {}'.format(self.doe_order))

        self.sys_input_zeta = []
        for idoe_order in self.doe_order: 
           # DoE for different selected doe_order 
            samp_zeta = samplegen(self.doe_method, idoe_order, self.distJ,rule=self.doe_rule)
            self.sys_input_zeta.append(samp_zeta)

            if self.doe_method.upper() in ['QUAD', 'GQ']:
                self.nsys_input_vars_dim = samp_zeta[0].shape[0] 
                self.nsamples_per_doe.append(samp_zeta[0].shape[1])
            else:
                self.nsys_input_vars_dim = samp_zeta.shape[0] 
                self.nsamples_per_doe.append(samp_zeta.shape[1])

        # print(' ► ----   Done (Design of Experiment)   ----')
        print(' ► Summary (Design of Experiment) ')
        print('   ♦ Number of sample sets : {:d}'.format(len(self.sys_input_zeta)))
        print('   ♦ Sample shape: ')
        for isample_zeta in self.sys_input_zeta:
            if self.doe_method.upper() in ['QUAD', 'GQ']:
                print('     ∙ Coordinate: {}; weights: {}'\
                        .format(isample_zeta[0].shape, isample_zeta[1].shape))
            else:
                print(' ♦ Sample shape: {}'.format(isample_zeta.shape))

    def get_doe_samples(self, dist_phy):
        """
        Return design sample points both in physical spaces based on specified doe_method
        Return:
        sys_input_vars: parameters defining the inputs for the solver
            General format of a solver
                y =  M(x,sys_params)
                M: system solver, taking a set of system parameters and input x
                x: system inputs, ndarray of shape (ndim, nsamples)
                    1. M takes x[:,i] as input, y = M(x.T)
                    2. M takes time series generated from input_func(x[:,i]) as input
            Experiment samples at physical space (idoe_order,[samples for each doe_order])
        """

        self.sys_input_vars = []
        assert len(dist_phy) == len(self.dist_zeta)

        for isample_zeta in self.sys_input_zeta:
            if self.doe_method.upper() in ['QUAD', 'GQ']:
                zeta_cor, zeta_weights = isample_zeta 
                phy_cor = self._dist_transform(self.dist_zeta, dist_phy, zeta_cor)
                phy_weights = zeta_weights
                samp_phy = np.array([phy_cor, phy_weights])
            else:
                zeta_cor, zeta_weights = isample_zeta, None
                phy_cor = self._dist_transform(self.dist_zeta, dist_phy, zeta_cor)
                samp_phy = phy_cor
            self.sys_input_vars.append(samp_phy)
        self.sys_input_params.append(self.sys_input_vars)

    def _dist_transform(self,dist1, dist2, var1):
        """
        Transform variables (var1) from list of dist1 to correponding variables in dist2. based on same icdf

        Arguments:
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

