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
import os
from doe.doe_generator import samplegen
## Define parameters class
DOE_METHOD_NAMES = {
    "GQ"    : "Quadrature"  , "QUAD"  : "Quadrature",
    "MC"    : "Monte Carlo" , "FIX"   : "Fixed point"
    } 

DOE_RULE_NAMES = {
    "c": "clenshaw_curtis"  , "e"   : "gauss_legendre"  , "p"   : "gauss_patterson",
    "z": "genz_keister"     , "g"   : "golub_welsch"    , "j"   : "leja",
    "h": "gauss_hermite"    ,"lag"  : "gauss_laguerre"  , "cheb": "gauss_chebyshev",
    "hermite"   :"gauss_hermite",
    "legendre"  :"gauss_legendre",
    "jacobi"    :"gauss_jacobi",
    "R": "Pseudo-Random", "RG": "Regular Grid", "NG": "Nested Grid", "L": "Latin Hypercube",
    "S": "Sobol", "H":"Halton", "M": "Hammersley",
    "FIX": "Fixed point"
    }
class simParameter(object):
    """
    Define general parameter settings for simulation running and post data analysis. 
    System parameters will be different between solver and solver

    Arguments:
        dist_zeta: list of selected marginal distributions from Wiener-Askey scheme
        *OPTIONAL:
        doe_params  = [doe_method, doe_rule, doe_order]
        time_params = [time_start, time_ramp, time_max, dt]
        post_params = [qoi2analysis=[0,], stats=[1,1,1,1,1,1,0]]
            stats: [mean, std, skewness, kurtosis, absmax, absmin, up_crossing]
        sys_def_params: array of parameter sets defining the solver
            if no sys_def_params is required, sys_def_params = None
            e.g. for duffing oscillator:
            sys_def_params = np.array([0,0,1,1,1]).reshape(1,5) # x0,v0, zeta, omega_n, mu 
        normalize: 
    """

    def __init__(self,dist_zeta, doe_params=['GQ','lag',[2,3]],\
            time_params=None, post_params=[[0,], [1,1,1,1,1,1,0]],\
            sys_def_params=None, normalize=False):

        self.seed       = [0,100]
        self.dist_zeta  = dist_zeta
        self.distJ      = dist_zeta if len(dist_zeta) == 1 else cp.J(*dist_zeta) 
        self.doe_params = doe_params
        self.doe_method = DOE_METHOD_NAMES.get(doe_params[0])
        self.doe_rule   = doe_params[1]
        self.doe_order  = []
        if np.isscalar(doe_params[2]): 
            self.doe_order.append(int(doe_params[2]))
        else:
            self.doe_order = np.array(doe_params[2])

        self.ndoe               = len(self.doe_order)   # number of doe sets
        self.nsamples_per_doe   = []                    # number of samples for each doe sets
        
        self.time_start, self.time_ramp, self.time_max, self.dt = time_params if time_params else [None]*4
        self.qoi2analysis, self.stats = post_params
        self.normalize      = normalize 
        self.nsamples_done  = 0
        self.outdir_name    = ''
        self.outfile_name   = ''
        self.sys_def_params = sys_def_params if sys_def_params else [None]
        self.nsys_input_vars_dim= 0                     # dimension of solver inputs 
        # sys_input_params = [sys_input_func_name, sys_input_kwargs, sys_input_vars]
        self.sys_input_vars = []
        self.sys_input_params = [[None], [None], self.sys_input_vars]
        self.input_vars_phy = []


    def get_doe_samples_zeta(self, samp_phy=None, dist_phy=None):
        """
        Return design sample points both in zeta spaces based on specified doe_method
        Return:
            Experiment samples in zeta space (idoe_order,[samples for each doe_order])
        """

        print('------------------------------------------------------------')
        print('►►► Design of Experiments (DoEs)')
        print('------------------------------------------------------------')
        print(' ► DoE parameters: ')
        print('   ♦ {:<15s} : {:<15s}'.format('DoE method', self.doe_method))
        print('   ♦ {:<15s} : {:<15s}'.format('DoE rule', DOE_RULE_NAMES[self.doe_rule]))
        print('   ♦ {:<15s} : {}'.format('DoE order', self.doe_order))
        print(' ► Running DoEs: ')

        self.sys_input_zeta = []
        if self.doe_method.upper() == 'FIXED POINT':
            for isamp_phy in samp_phy:
                self.sys_input_zeta.append(self._dist_transform(dist_phy, self.dist_zeta, isamp_phy))
        else: ## could be MC or quadrature
            for idoe_order in self.doe_order: 
               # DoE for different selected doe_order 
                samp_zeta = samplegen(self.doe_method, idoe_order, self.distJ,rule=self.doe_rule)
                self.sys_input_zeta.append(samp_zeta)

                if self.doe_method.upper() == 'QUADRATURE':
                    ## if quadrature, samp_zeta = [coord(ndim, nsamples), weights(nsamples,)]
                    self.nsys_input_vars_dim = samp_zeta[0].shape[0] 
                    self.nsamples_per_doe.append(samp_zeta[0].shape[1])
                else:
                    ## if mc , samp_zeta = [coord(ndim, nsamples)]
                    self.nsys_input_vars_dim = samp_zeta.shape[0] 
                    self.nsamples_per_doe.append(samp_zeta.shape[1])

        # print(' ► ----   Done (Design of Experiment)   ----')
        print(' ► DoE Summary:')
        print('   ♦ Number of sample sets : {:d}'.format(len(self.sys_input_zeta)))
        print('   ♦ Sample shape: ')
        for isample_zeta in self.sys_input_zeta:
            if self.doe_method.upper() == 'QUADRATURE':
                print('     ∙ Coordinate: {}; weights: {}'\
                        .format(isample_zeta[0].shape, isample_zeta[1].shape))
            else:
                print('     ∙ Coordinate: {}'.format(isample_zeta.shape))

    def get_doe_samples(self, dist_phy, is_both=True, filename='inputs'):
        """
        Return and save design sample points both in physical spaces based on specified doe_method
        Return:
        sys_input_vars: parameters defining the inputs for the solver
            General format of a solver
                y =  M(x,sys_def_params)
                M: system solver, taking a set of system parameters and input x
                x: system inputs, ndarray of shape (ndim, nsamples)
                    1. M takes x[:,i] as input, y = M(x.T)
                    2. M takes time series generated from input_func(x[:,i]) as input
            Experiment samples in physical space (idoe_order,[samples for each doe_order])
        """

        # # self.sys_input_vars = []
        assert len(dist_phy) == len(self.dist_zeta)

        ## Get samples in zeta space first
        self.get_doe_samples_zeta() 

        # n_sys_input_func_name = len(self.sys_input_params[0]) if self.sys_input_params[0] else 1
        n_sys_input_func_name = len(self.sys_input_params[0]) 
        n_sys_input_kwargs    = len(self.sys_input_params[1]) 
        n_sys_def_params      = len(self.sys_def_params) 
        n_total_simulations = n_sys_input_func_name * n_sys_input_kwargs *n_sys_def_params 


        ## Then calcuate corresponding samples in physical variable space 
        for idoe_order, isample_zeta in enumerate(self.sys_input_zeta):
            filename = '_'.join([filename, self.doe_params[0], self.doe_params[1],'DoE'+'{:d}'.format(idoe_order)]) + '.txt'
            filename = os.path.join(self.outdir_name, filename)
            if self.doe_method.upper() == 'QUADRATURE':
                zeta_cor, zeta_weights = isample_zeta 
                phy_cor = self._dist_transform(self.dist_zeta, dist_phy, zeta_cor)
                phy_weights = zeta_weights#.reshape(zeta_cor.shape[1],1)
                samp_phy = np.array([phy_cor, phy_weights])
            else:
                zeta_cor, zeta_weights = isample_zeta, None
                phy_cor = self._dist_transform(self.dist_zeta, dist_phy, zeta_cor)
                samp_phy = phy_cor
            n_samp_phy = samp_phy.shape[1] 
            n_total_simulations *= n_samp_phy 
            # print('samp_phy shape: {}, coord shape: {}, weight shape {}'.format(samp_phy.shape, samp_phy[0].shape, samp_phy[1].shape))

            self.sys_input_vars.append(samp_phy)
            ### save input variables to file

            with open(filename, 'w') as file_input_vars:
                input_var = []
                for isys_def_params in self.sys_def_params:
                    l1 = list(isys_def_params) if isys_def_params else [None]
                    for isys_input_func_name in self.sys_input_params[0]:
                        l2 = list(isys_input_func_name) if isys_input_func_name else [None]
                        for isys_input_kwargs in self.sys_input_params[1]:
                            l3 = list(isys_input_kwargs ) if isys_input_kwargs else [None]
                            for isys_input_vars in samp_phy.T:
                                l4 = list(isys_input_vars ) 
                                input_var = l1 + l2 + l3 + l4
                                file_input_vars.write('\t'.join(map(str, input_var)))
                                # file_input_vars.write(input_var)
                                file_input_vars.write("\n")
        print('   ♦ Total number of simulations: {:d}'.format(int(n_total_simulations)))
        print(' ► Data saved {}'.format(filename))

        self.sys_input_params[2] = self.sys_input_vars

    def set_doe_samples(self, samp_phy, dist_phy):

        ## Set samples in physical space first
        self.sys_input_vars = samp_phy
        self.get_doe_samples_zeta(samp_phy, dist_phy) 

        self.sys_input_params[2].append(self.sys_input_vars)


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
        self.sys_input_params[0] = sys_input_func_name
        self.sys_input_params[1] = sys_input_kwargs

    # def set_nsamples_need(self, n):
        # self.nsamples_need= n
        # self.doe_method = xrange(self.nsamples_need)



    def _dist_transform(self, dist1, dist2, var1):
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
        assert (len(dist1) == len(dist2) == var1.shape[0]), 'Dimension of variable not equal. dist1.ndim={}, dist2.ndim={}, var1.ndim={}'.format(len(dist1), len(dist2), var1.shape[0])
        var2 = []
        for i in range(var1.shape[0]):
            _dist1 = dist1[i]
            _dist2 = dist2[i]
            _var = _dist2.inv(_dist1.cdf(var1[i,:]))
            var2.append(_var)
        var2 = np.array(var2)
        assert var1.shape == var2.shape
        return var2

