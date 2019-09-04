#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import sys
import chaospy as cp
import numpy as np
import os
from datetime import datetime
from doe.doe_generator import samplegen
from utilities import make_output_dir, get_gdrive_folder_id 
import settings

## Define parameters class

def _num2str(n):
    if n<100:
        return '{:2d}'.format(n)
    else:
        __str ='{:.0E}'.format(n) 
        return __str[0]+'E'+__str[-1] 

class ErrorType():
    def __init__(self, name=None, params=None, size=None):
        """
        name:   string, error distribution name or None if not defined
        params: list of error distribution parameters, float or array_like of floats 
                [ [mu1, sigma1, ...]
                  [mu2, sigma2, ...]
                  ...]
        size:   list of [int or tuple of ints], optional
            -- Output shape:
            If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. 
            If size is None (default), a single value is returned if loc and scale are both scalars. 
            Otherwise, np.broadcast(loc, scale).size samples are drawn.
        """
        assert name is None or isinstance(name, str) 
        if name is None:
            self.name   = 'Free' 
            self.params = None 
            self.size   = None 
        elif len(params) == 1:
            self.name   = name
            self.params = params[0]
            self.size   = size[0] if size else None
        else:
            self.name   = [] 
            for _ in range(len(params)):
                self.name.append(name) 
            self.size   = size if size else [None] * len(params)
            self.params = params

    def tolist(self, ndoe):
        """

        """
        if not isinstance(self.name, list):
            return [ErrorType(self.name, [self.params], [self.size])] * ndoe
        else:
            assert len(self.name) == ndoe,"ErrorType.name provided ({:2d}) doesn't match number of DoE expected ({:2d})".format(len(self.name), ndoe)
            error_type_list = []
            for i in range(ndoe):
                error_type_list.append(ErrorType(self.name[i], [self.params[i]], self.size[i]))
            return error_type_list

    def disp(self):

        if self.name.upper() == 'FREE':
            print('   ♦ Short-term/error distribution parameters: noise free')
        else:
            print('   ♦ Short-term/error distribution parameters:')
            print('     ∙ {:<15s} : {}'.format('dist_name', self.name))
            for i, ierror_params in enumerate(self.params):
                ierror_params_shape = []
                for iparam in ierror_params:
                    if np.isscalar(iparam):
                        ierror_params_shape.append(1)
                    else:
                        ierror_params_shape.append(np.array(iparam).shape)

                if i == 0:
                    print('     ∙ {:<15s} : {}'.format('dist_params',ierror_params_shape))
                else:
                    print('     ∙ {:<15s} : {}'.format('',ierror_params_shape))
            print('     ∙ {:<15s} : {}'.format('dist_size', self.size))

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

class simParameters(object):
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

    def __init__(self, model_name, dist_zeta, dist_x, prob_fails=1e-3):
        sys.stdout = Logger()
        ###---------- Random system properties ------------------------
        self.seed       = [0,100]
        self.model_name = model_name
        self.prob_fails = prob_fails
        self.dist_zeta  = dist_zeta
        self.dist_zeta_J= dist_zeta #if len(dist_zeta) == 1 else cp.J(*dist_zeta) 
        self.dist_x     = dist_x
        self.dist_x_J   = dist_x    #if len(dist_zeta) == 1 else cp.J(*dist_zeta) 
        assert len(self.dist_x) == len(self.dist_zeta)
        assert len(self.dist_x_J) == len(self.dist_zeta_J)

        ###-------------Directories setting -----------------------------
        self.pwd            = ''
        self.data_dir       = ''
        self.data_dir_id    = ''
        self.figure_dir     = ''
        self.outfilename    = ''

        ###-------------Output file names -----------------------------
        self.fname_doe_out  = ''
        ###-------------Systerm input params ----------------------------
        ### sys_def_params is of shape (m,n)
        ##  - m: number of set, 
        ##  - n: number of system parameters per set
        self.sys_def_params   = []
        self.sys_excit_params = [] ### sys_excit_params = [sys_excit_func_name, sys_excit_func_kwargs]
        self.sys_input_x      = []
        self.sys_input_zeta   = []

        ###-------------Error type paramters ----------------------------
        self.error = ErrorType()

        ###-------------Others ------------------------------------------
        self.set_params()

    def set_error(self, params=None):
        """
        Set error distributions 
        """
        if params is None:
            self.error = ErrorType()
        else:
            name, params, size = params
            self.error = ErrorType(name=name, params=params, size=size)

    def set_seed(self, s):
        self.seed = s

    def update_outfilename(self,filename):
        self.outfilename = filename

    def update_dir(self, **kwargs):
        """
        update directories for working and data saving.
            Takes either a string argument or a dictionary argument

        self.update_dir(MDOEL_NAME)  (set as default and has priority).
            if MODEL_NAME is given, kwargs is ignored
        self.update_dir(pwd=, data_dir=, figure_dir=)

        Updating directories of
            pwd: present working directory, self.pwd
            data_dir: directory saving all data, self.data_dir
            figure_dir: directory saving all figures, self.figure_dir
        """
        data_dir_id, data_dir, figure_dir =  make_output_dir(self.model_name)
        self.pwd        = kwargs.get('pwd'          , os.getcwd()   )
        self.data_dir   = kwargs.get('data_dir'     , data_dir      )
        self.figure_dir = kwargs.get('figure_dir'   , figure_dir    )
        self.data_dir_id= kwargs.get('data_dir_id'  , data_dir_id   )

    def set_params(self, **kwargs):
        """
        Taking key word arguments to set parameters like time, post process etc.
        """

        ## define parameters related to time steps in simulation 
        self.time_params= kwargs.get('time_params'  , [0,0,0,0])
        self.time_start = kwargs.get('time_start'   , self.time_params[0])
        self.time_ramp  = kwargs.get('time_ramp'    , self.time_params[1])
        self.time_max   = kwargs.get('time_max'     , self.time_params[2])
        self.dt         = kwargs.get('dt'           , self.time_params[3])

        ## define parameters related to post processing
        self.post_params    = kwargs.get('post_params'  , [[0,], [1,1,1,1,1,1,0]])
        self.qoi2analysis   = kwargs.get('qoi2analysis' , self.post_params[0]) 
        self.stats          = kwargs.get('stats'        , self.post_params[1])

        ###-------------Systerm input params ----------------------------
        ### sys_def_params is of shape (m,n)
        ##  - m: number of set, 
        ##  - n: number of system parameters per set
        self.sys_def_params     = kwargs.get('sys_def_params'   , [None])

        ### sys_excit_params = [sys_excit_func_name, sys_excit_func_kwargs]
        self.sys_excit_params   = kwargs.get('sys_excit_params' , [[None], [None]])  
        self.sys_excit_params[0]= kwargs.get('sys_excit_func_name', self.sys_excit_params[0])
        self.sys_excit_params[1]= kwargs.get('sys_excit_func_kwargs', self.sys_excit_params[1])

        ## 
        ###-------------Other special params ----------------------------
        self.normalize = kwargs.get('normalize', False)


