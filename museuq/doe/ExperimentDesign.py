#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import sys, os
import chaospy as cp, numpy as np
import collections
from datetime import datetime
from tqdm import tqdm,tqdm_notebook
from ..utilities.classes import ObserveError, Logger
from ..utilities.helpers import num2print, make_output_dir, get_gdrive_folder_id 
from ..utilities import dataIO 
from ..utilities import constants as const

from .doe_generator import samplegen

## Define parameters class

class ExperimentDesign(object):
    """
    Experimental design class 
    Define general parameter settings for DoE

    Arguments:
        dist_zeta: list of selected marginal distributions from Wiener-Askey scheme
        *OPTIONAL:
        params  = [method, rule, orders]
        time_params = [time_start, time_ramp, time_max, dt]
        post_params = [qoi2analysis=[0,], stats=[1,1,1,1,1,1,0]]
            stats: [mean, std, skewness, kurtosis, absmax, absmin, up_crossing]
        sys_def_params: array of parameter sets defining the solver
            if no sys_def_params is required, sys_def_params = None
            e.g. for duffing oscillator:
            sys_def_params = np.array([0,0,1,1,1]).reshape(1,5) # x0,v0, zeta, omega_n, mu 
        normalize: 
    """

    def __init__(self, method, rule, orders, space=None, **kwargs):
        self.params = [] 
        self.method = method
        self.rule   = rule
        self.orders = [int(orders),] if np.isscalar(orders) else sorted(orders)
        self.space  = space
        self.ndoe   = len(self.orders)   # number of doe sets
        self.samples=[]  # DoE output in space1
        self.samples_env = None # DoE output in space2 after calling mappingto method
        self.filename  = 'DoE_{}{}'.format(self.method.capitalize(), self.rule.capitalize())
        self.filename_tags = []
        for item, count in collections.Counter(self.orders).items():
            if count == 1:
                itag = [ num2print(item)]
            else:
                itag = [ num2print(item) + 'R{}'.format(i) for i in range(count)]
            self.filename_tags += itag

        # if self.method == 'MC':
            # self.filename_tags = [ num2print(iorder) + 'R{}'.format(i) for i, iorder in enumerate(self.orders)] 
        # else:
            # self.filename_tags = [ num2print(iorder) for iorder in self.orders] 

        print(r'------------------------------------------------------------')
        print(r'>>> Initialize Experiment Design:')
        print(r'------------------------------------------------------------')
        print(r' > DoE parameters: ')
        print(r'   * {:<15s} : {:<15s}'.format('DoE method', const.DOE_METHOD_FULL_NAMES[self.method.lower()]))
        print(r'   * {:<15s} : {:<15s}'.format('DoE rule',const.DOE_RULE_FULL_NAMES[self.rule.lower()]))
        print(r'   * {:<15s} : {}'.format('DoE order', list(map(num2print, self.orders)) ))

    def get_samples(self, space=None):
        """
        Return design sample points in specified spaces (dist) based on specified method
        space: 
            1. cp.distributions
        Return:
            Experiment samples in space (idoe_order,[samples for each orders])
        """
        print(' > Running DoEs: ')
        self.samples = []
        ## one can update new space here
        self.space = space if space else self.space
        # if self.method.lower() == 'FIXED POINT':
            # assert fix_point is not None, " For 'fixed point' method, samples in physical space must be given" 
            # for isample_x in fix_point:
                # self.samples.append(self._space_transform(self.dist_x, self.dist_zeta, isample_x))
        if const.DOE_METHOD_FULL_NAMES[self.method.lower()] in ['QUADRATURE', 'MONTE CARLO']:
            for i, idoe_order in enumerate(tqdm(self.orders, ascii=True, desc='   *')): 
                # DoE for different selected orders 
                # doe samples, array of shape:
                #    - Quadrature: res.shape = (2,) 
                #       res[0]: coord of shape (ndim, nsamples) 
                #       res[1]: weights of shape (nsamples,) 
                #    - MC: res.shape = (ndim, nsamples)
                isamples = samplegen(self.method, idoe_order, self.space, rule=self.rule)
                self.samples.append(isamples)
            tqdm._instances.clear()
                # print('\r   * {:<15s}: {}'.format( 'DoE completed', list(map(num2print, self.orders[:i+1]))), end='')
        else:
            raise ValueError('DoE method: {:s} not implemented'.format(const.DOE_METHOD_FULL_NAMES[self.method.lower()]))

        self.samples = self.samples[0] if len(self.samples)==1 else self.samples
        return self.samples

    def mappingto(self, space2):
        """
        Mapping the DOE results from original space (self.space) to another specified space  
        """
        self.space_env   = space2
        self.samples_env = []
        ## if self.samples is not a list (in the case of only 1 DOE set), change it to 1 elememnt list first 
        self.samples = [self.samples,] if not isinstance(self.samples, list) else self.samples

        # # n_sys_excit_func_name = len(self.sys_excit_params[0]) if self.sys_excit_params[0] else 1
        # n_sys_excit_func_name = len(self.sys_excit_params[0]) 
        # n_sys_excit_func_kwargs= len(self.sys_excit_params[1]) 
        # n_sys_def_params      = len(self.sys_def_params) 
        # n_total_simulations   = n_sys_excit_func_name * n_sys_excit_func_kwargs *n_sys_def_params 

        ## Then calcuate corresponding samples in physical variable space 
        # with open(os.path.join(self.data_dir, self.filename), 'w') as filename:

        for idoe, isamples in enumerate(self.samples):
            if const.DOE_METHOD_FULL_NAMES[self.method.lower()] == 'QUADRATURE':
                zeta_cor, zeta_weights = isamples[:-1,:], isamples[-1,:] 
                x_cor = self._space_transform(self.space, self.space_env, zeta_cor)
                x_weights = zeta_weights#.reshape(zeta_cor.shape[1],1)
                isample_x = np.concatenate((x_cor, x_weights[np.newaxis,:]), axis=0)
                # isample_x = np.array([x_cor, x_weights])
            else:
                zeta_cor, zeta_weights = isamples, None
                x_cor = self._space_transform(self.space, self.space_env, zeta_cor)
                isample_x = x_cor
            # print(r'isample_x shape: {}, coord shape: {}, weight shape {}'.format(isample_x.shape, isample_x[0].shape, isample_x[1].shape))

            self.samples_env.append(isample_x)

        self.samples = self.samples[0] if len(self.samples)==1 else self.samples
        self.samples_env = self.samples_env[0] if len(self.samples_env)==1 else self.samples_env
        return self.samples_env

    def set_samples(self, **kwargs):
        """
        TO FINISH, not yet in practice
        Set samples in physical space, used for FIXED POINT scheme
        """
        self.samples     = kwargs.get('zeta', self.samples)
        self.samples_env = kwargs.get('env', self.samples_env) 

    def set_method(self,params= ['QUAD','hem',[2,3]]):
        """
        Define DoE parameters and properties
        """
        self.params = params
        self.method = params[0] 
        self.rule   = params[1]
        self.orders = params[2] 
        self.filename= [] 
        if np.isscalar(params[2]): 
            self.orders.append(int(params[2]))
        else:
            self.orders = np.array(params[2])

        self.filename  = 'DoE_{}{}'.format(self.method.capitalize(), self.rule.capitalize())
        self.ndoe        = len(self.orders)   # number of doe sets
        if self.method == 'MC':
            self.filename_tags = [ num2print(riorder) + 'R{}'.format(i) for i, iorder in enumerate(self.orders)] 
        else:
            self.filename_tags = [ num2print(riorder) for iorder in self.orders] 

    def save_data(self, data_dir):
        ### save input variables to file
        print(r' > Saving DoE to: {}'.format(data_dir))
        if self.samples_env is not None:
            if isinstance(self.samples, (np.ndarray, np.generic)):
                data = np.concatenate((self.samples, self.samples_env), axis=0)
            else:
                data = []
                for idata in zip(self.samples, self.samples_env):
                    data.append(np.concatenate(idata, axis=0))
        else:
            data = self.samples

        dataIO.save_data(data, self.filename, data_dir, self.filename_tags)

    def info(self, decimals=4, nsamples2print=0):
        print(r' > DoE Summary:')
        print(r'   * Number of sample sets : {:d}'.format( 1 if isinstance(self.samples, np.ndarray) else len(self.samples)))
        if const.DOE_METHOD_FULL_NAMES[self.method.lower()] == 'QUADRATURE':
            print(r'   * {:10s} & {:<10s}'.format('Abscissae', 'Weights'))
            if isinstance(self.samples, np.ndarray):
                print(r'     ∙ {} & {}'.format(self.samples[:-1,:].shape,self.samples[-1,:].shape))
            else:
                for isamples in self.samples:
                    print(r'     ∙ {} & {}'.format(isamples[:-1,:].shape,isamples[-1,:].shape))
        elif const.DOE_METHOD_FULL_NAMES[self.method.lower()] == 'MONTE CARLO':
            if isinstance(self.samples, np.ndarray):
                for jcor in self.samples[:,:nsamples2print].T:
                    print(r'         {}'.format(np.around(jcor,decimals)))
            else:
                for isamples in self.samples:
                    print(r'       - sample size: {:d} '.format(isamples.shape[1]))
                    for jcor in isamples[:,:nsamples2print].T:
                        print(r'         {}'.format(np.around(jcor,decimals)))

    def _space_transform(self, dist1, dist2, var1):
        """
        Transform variables (var1) from list of dist1 to correponding variables in dist2. F1(x) = F2(x) , same cdf

        Arguments:
            dist2: list of independent distributions (destination)
            var1 : variables in dist1 of shape[ndim, nsamples]

        Return:
            
        """
        var1  = np.array(var1)
        dist1 = [dist1,] if len(dist1) == 1 else dist1
        dist2 = [dist2,] if len(dist2) == 1 else dist2
        assert (len(dist1) == len(dist2)), "No. of original and target distributions must be equal, but origianl sets: {:d}, target sets: {:}".format(len(dist1), len(dist2))

        var1 = var1.reshape(1,-1) if var1.ndim == 1 else var1
        assert (len(dist1) == var1.shape[0]), 'Dimension of variable not equal. dist1.ndim={}, dist2.ndim={}, var1.ndim={}'.format(len(dist1), len(dist2), var1.shape[0])

        var2 = []
        for i in range(var1.shape[0]):
            _dist1 = dist1[i]
            _dist2 = dist2[i]
            _var = _dist2.inv(_dist1.cdf(var1[i,:]))
            var2.append(_var)
        var2 = np.array(var2)
        assert var1.shape == var2.shape
        return var2



