#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import os, sys
import chaospy as cp
import numpy as np
from datetime import datetime
from .doe.doe_generator import samplegen
from .utilities.classes import ErrorType, Logger
from .utilities.helpers import num2print, make_output_dir, get_gdrive_folder_id 

## Define parameters class


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

    def update_outfilename(self,filename):
        self.outfilename = filename

    def update_dir(self, **kwargs):
        """
        update directories for working and data saving.
            Takes either a string argument or a dictionary argument

        self.update_dir(MDOEL_NAME)  (set as default and has priority).
            if model_name is given, kwargs is ignored
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
        self.time_params= kwargs.get('time_params'  , None)
        self.time_start = kwargs.get('time_start'   , self.time_params)
        self.time_ramp  = kwargs.get('time_ramp'    , self.time_params)
        self.time_max   = kwargs.get('time_max'     , self.time_params)
        self.dt         = kwargs.get('dt'           , self.time_params)

        ## define parameters related to post processing
        self.post_params    = kwargs.get('post_params'  , [[0,], [1,1,1,1,1,1,0]])
        self.qoi2analysis   = kwargs.get('qoi2analysis' , self.post_params[0]) 
        self.stats          = kwargs.get('stats'        , self.post_params[1])

        ###-------------Systerm input params ----------------------------
        ### sys_def_params is of shape (m,n)
        ##  - m: number of set, 
        ##  - n: number of system parameters per set
        self.sys_def_params     = kwargs.get('sys_def_params'   , None)

        ### sys_excit_params = [sys_excit_func_name, sys_excit_func_kwargs]
        self.sys_excit_params   = kwargs.get('sys_excit_params' , [None, None])  
        self.sys_excit_params[0]= kwargs.get('sys_excit_func_name', None)
        self.sys_excit_params[1]= kwargs.get('sys_excit_func_kwargs', None)

        ## 
        ###-------------Other special params ----------------------------
        self.normalize = kwargs.get('normalize', False)

    def set_error(self, params=None):
        """
        Set error distributions 
        """
        if params is None:
            self.error = ErrorType()
        else:
            name, params, size = params
            self.error = ErrorType(name=name, params=params, size=size)
    def disp(self):

        print('------------------------------------------------------------')
        print('►►► SimParameter setting for model: {}'.format(self.model_name))
        print('------------------------------------------------------------')
        print(' ► Required parameters:')
        print('   ♦ {:<25s} : {}'.format('Model Name:', self.model_name))
        print('   ♦ {:<25s} : {} '.format('Target Exceedance prob', self.prob_fails))
        print('   ♦ {:<25s} : {} '.format('Joint zeta distribution', self.dist_zeta_J))
        print('   ♦ {:<25s} : {} '.format('Joint x distribution', self.dist_x_J))

        print(' ► Working directory:')
        print('   WORKING_DIR: {}'.format(os.getcwd()))
        print('   +-- MODEL: {}'.format(self.figure_dir[:-7]))
        print('   |   +-- {:<6s}: {}'.format('FIGURE',self.figure_dir))
        print('   |   +-- {:<6s}: {}'.format('DATA',self.data_dir))

        print(' ► Optional parameters:')
        if self.time_params:
            print('   ♦ {:<15s} : '.format('time parameters'))
            print('     ∙ {:<8s} : {:.2f} ∙ {:<8s} : {:.2f}'.format('start', self.time_start, 'end', self.time_max ))
            print('     ∙ {:<8s} : {:.2f} ∙ {:<8s} : {:.2f}'.format('ramp ', self.time_ramp , 'dt ', self.dt ))
        if self.post_params:
            print('   ♦ {:<15s} '.format('post analysis parameters'))
            print('     ∙ {:<15s} : {} '.format('qoi2analysis', self.qoi2analysis ))
            print('     ∙ {:<15s} : {} '.format('statistics'  , self.stats ))
        # if self.sys_def_params is None :
            # print('   ♦ {:<15s} : {:s}'.format('system definition parameters', 'NA'))
        # else:
            # print('   ♦ {:<15s} '.format('system definition parameters'))
            # print('     ∙ {:<10s} : {:d}  ∙ {:<10s} : {:d} '.format('# dim', self.sys_def_params.shape[0],'# set', self.sys_def_params.shape[1] ))
        # if self.sys_excit_params is None or self.sys_excit_params[0] is None:
            # print('   ♦ {:<15s} : {:s}'.format('system excitation force', 'NA'))
        # else:
            # print('   ♦ {:<15s} '.format('system excitation force'))
            # print('     ∙ {:<15s} : {} '.format('Excitation function name', self.sys_excit_params[0] ))
            # print('     ∙ {:<15s} :    '.format('Excitation function kwargs'))
            # for ikey, ivalue in self.sys_excit_params[1].items():
                # print('     - {:<15s} : {}   '.format(ikey, ivalue))

