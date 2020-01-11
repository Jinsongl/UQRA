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
from .utilities.classes import Logger
from .utilities.helpers import num2print
from itertools import compress
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

    def __init__(self, model_name, dist_zeta_J):
        sys.stdout = Logger()
        ###---------- Random system properties ------------------------
        self.seed       = [0,100]
        self.model_name = model_name
        self.dist_zeta_J= dist_zeta_J   ## Joint distribution
        self.dist_zeta_M= None          ## List of marginal distributions
        # self.dist_x     = dist_x
        # self.dist_x_J   = dist_x    #if len(dist_zeta) == 1 else cp.J(*dist_zeta) 
        # assert len(self.dist_x) == len(self.dist_zeta)
        # assert len(self.dist_x_J) == len(self.dist_zeta_J)

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

        ###-------------Observation Error ----------------------------
        # self.set_error()
        ###-------------Others ------------------------------------------
        self.set_params()
        self.update_dir()

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
        data_dir_id, data_dir, figure_dir =  self._make_output_dir(self.model_name)
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
        self.post_params    = kwargs.get('post_params'  , [None, [1,1,1,1,1,1,0]])
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
        self.dist_zeta_M = kwargs.get('dist_zeta_M', None)

    def info(self):
        print(r'------------------------------------------------------------')
        print(r'>>> SimParameter setting for model: {}'.format(self.model_name))
        print(r'------------------------------------------------------------')
        print(r' > Required parameters:')
        print(r'   * {:<25s} : {}'.format('Model Name:', self.model_name))
        print(r'   * {:<25s} : {} '.format('Joint zeta distribution', self.dist_zeta_J))
        # print(r'   * {:<25s} : {} '.format('Joint x distribution', self.dist_x_J))

        print(r' > Working directory:')
        print(r'   WORKING_DIR: {}'.format(os.getcwd()))
        print(r'   +-- MODEL: {}'.format(self.figure_dir[:-7]))
        print(r'   |   +-- {:<6s}: {}'.format('FIGURE',self.figure_dir))
        print(r'   |   +-- {:<6s}: {}'.format('DATA',self.data_dir))

        print(r' > Optional parameters:')
        if self.dist_zeta_M:
            print(r'   * {:<15s} : '.format('Marginal distributions'))
        if self.time_params:
            print(r'   * {:<15s} : '.format('time parameters'))
            print(r'     - {:<8s} : {:.2f} - {:<8s} : {:.2f}'.format('start', self.time_start, 'end', self.time_max ))
            print(r'     - {:<8s} : {:.2f} - {:<8s} : {:.2f}'.format('ramp ', self.time_ramp , 'dt ', self.dt ))
        if self.post_params:
            print(r'   * {:<15s} '.format('post analysis parameters'))
            qoi2analysis = self.qoi2analysis if self.qoi2analysis is not None else 'All'
            print(r'     - {:<15s} : {} '.format('qoi2analysis', qoi2analysis))
            stats_list = ['mean', 'std', 'skewness', 'kurtosis', 'absmax', 'absmin', 'up_crossing']
            print(r'     - {:<15s} : {} '.format('statistics'  , list(compress(stats_list, self.stats)) ))
    

    def _make_output_dir(self, model_name):
        """
        WORKING_DIR/
        +-- MODEL_DIR
        |   +-- FIGURE_DIR

        /directory saving data depends on OS/
        +-- MODEL_DIR
        |   +-- DATA_DIR

        """
        WORKING_DIR     = os.getcwd()
        MODEL_DIR       = os.path.join(WORKING_DIR, model_name.capitalize())
        FIGURE_DIR= os.path.join(MODEL_DIR,r'Figures')
        # DATA_DIR  = os.path.join(MODEL_DIR,r'Data')
        current_os  = sys.platform
        if current_os.upper()[:3] == 'WIN':
            DATA_DIR= os.path.join('G:','My Drive','MUSE_UQ_DATA')
            MODEL_DIR_DATA_ID = self._get_gdrive_folder_id(model_name)
        elif current_os.upper() == 'DARWIN':
            DATA_DIR= '/Users/jinsongliu/External/MUSE_UQ_DATA'
            MODEL_DIR_DATA_ID = self._get_gdrive_folder_id(model_name)
        elif current_os.upper() == 'LINUX':
            MODEL_DIR_DATA_ID = None 
            DATA_DIR = WORKING_DIR
            # DATA_DIR= '/home/jinsong/Box/MUSE_UQ_DATA'
        else:
            raise ValueError('Operating system {} not found'.format(current_os))    
        
        DATA_DIR  = os.path.join(DATA_DIR,model_name,r'Data')
        # MODEL_DIR_DATA_ID = GDRIVE_DIR_ID[model_name.upper()] 

        # Create directory for model  
        try:
            os.makedirs(MODEL_DIR)
            os.makedirs(DATA_DIR)
            os.makedirs(FIGURE_DIR)
            # print(r'Data, Figure directories for model {} is created'.format(model_name))
        except FileExistsError:
            # one of the above directories already exists
            # print(r'Data, Figure directories for model {} already exist'.format(model_name))
            pass
        return MODEL_DIR_DATA_ID, DATA_DIR, FIGURE_DIR


    
    def _get_gdrive_folder_id(self, folder_name):
        """
        Check if the given folder_name exists in Google Drive. 
        If not, create one and return the google drive ID
        Else: return folder ID directly
        """
        # GDRIVE_DIR_ID = {
                # 'BENCH1': '1d1CRxZ00f4CiwHON5qT_0ijgSGkSbfqv',
                # 'BENCH4': '15KqRCXBwTTdHppRtDjfFZtmZq1HNHAGY',
                # 'BENCH3': '1TcVfZ6riXh9pLoJE9H8ZCxXiHLH_jigc',
                # }
        command = os.path.join('/Users/jinsongliu/Google Drive File Stream/My Drive/MUSE_UQ_DATA', folder_name)
        try:
            os.makedirs(command)
        except FileExistsError:
            pass
        command = 'gdrive list --order folder |grep ' +  folder_name
        folder_id = os.popen(command).read()
        return folder_id[:33]


