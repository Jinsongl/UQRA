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
from .utilities.classes import Logger
from itertools import compress

class simParameters(object):
    """
    Define general parameter settings for simulation running and post data analysis. 
    System parameters will be different depending on solver 

    Arguments:
        model_name: 
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

    def __init__(self, model_name,  **kwargs):
        sys.stdout = Logger()
        ###---------- Random system properties ------------------------
        self.seed       = [0,100]
        self.model_name = model_name.capitalize()

        ###------------- Adaptive setting -----------------------------
        self.is_adaptive= False
        self.plim       = kwargs.get('plim'     , None)   ## polynomial degree limit
        self.n_budget   = kwargs.get('n_budget' , None)
        self.min_r2     = kwargs.get('min_r2'   , None  )   ## minimum adjusted R-squared threshold value to take
        self.rel_mse    = kwargs.get('rel_mse'  , None  )   ## Relative mean square error 
        self.abs_mse    = kwargs.get('abs_mse'  , None  )   ## Absolute mean square error
        self.rel_qoi    = kwargs.get('rel_qoi'  , None  )   ## Relative error for QoI, i.e. percentage difference relative to previous simulation
        self.abs_qoi    = kwargs.get('abs_qoi'  , None  )   ## Absolute error for QoI, i.e. decimal accuracy 
        self.qoi_val    = kwargs.get('qoi_val'  , None  )   ## QoI value up to decimal accuracy 
        self.rel_cv     = kwargs.get('rel_cv'   , 0.05  )   ## percentage difference relative to previous simulation
        if self.plim is not None or self.n_budget is not None:
            self.is_adaptive = True
        
        ###-------------Directories setting -----------------------------

        self.pwd            = ''
        self.figure_dir     = ''
        self.data_dir_sample= ''
        self.data_dir_result= ''
        self.dir_id_sample  = ''
        self.dir_id_result  = ''
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
        self.update_dir(pwd=, data_dir_result=, figure_dir=)

        Updating directories of
            pwd: present working directory, self.pwd
            data_dir_result: directory saving all data, self.data_dir_result
            figure_dir: directory saving all figures, self.figure_dir
        """
        data_dir_sample, data_dir_result, dir_id_sample, dir_id_result, figure_dir =  self._make_output_dir()
        self.pwd            = kwargs.get('pwd'              , os.getcwd()    )
        self.figure_dir     = kwargs.get('figure_dir'       , figure_dir     )
        self.data_dir_result= kwargs.get('data_dir_result'  , data_dir_result)
        self.data_dir_sample= kwargs.get('data_dir_sample'  , data_dir_sample)
        self.dir_id_sample  = kwargs.get('dir_id_sample'    , dir_id_sample  )
        self.dir_id_result  = kwargs.get('dir_id_result'    , dir_id_result  )

    def set_adaptive_parameters(self, **kwargs):

        self.is_adaptive=True
        for ikey, ivalue in kwargs.items():
            try:
                setattr(self, ikey, ivalue)
            except:
                raise KeyError

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
        self.post_params    = kwargs.get('post_params'  , [None, ['mean', 'std', 'skewness', 'kurtosis', 'absmax', 'absmin', 'up_crossing']])
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

    def info(self):
        print(r'------------------------------------------------------------')
        print(r'>>> SimParameter setting for model: {}'.format(self.model_name))
        print(r'------------------------------------------------------------')
        print(r' > Required parameters:')
        print(r'   * {:<25s} : {}'.format('Model Name:', self.model_name))

        print(r' > Working directory:')
        print(r'   WORKING_DIR: {}'.format(os.getcwd()))
        print(r'   +-- MODEL: {}'.format(self.figure_dir[:-7]))
        print(r'   |   +-- {:<6s}: {}'.format('FIGURE',self.figure_dir))
        print(r'   |   +-- {:<6s}: {}'.format('DATA(RESULT)',self.data_dir_result))

        print(r' > Optional parameters:')
        if self.time_params:
            print(r'   * {:<15s} : '.format('time parameters'))
            print(r'     - {:<8s} : {:.2f} - {:<8s} : {:.2f}'.format('start', self.time_start, 'end', self.time_max ))
            print(r'     - {:<8s} : {:.2f} - {:<8s} : {:.2f}'.format('ramp ', self.time_ramp , 'dt ', self.dt ))
        if self.post_params:
            print(r'   * {:<15s} '.format('post analysis parameters'))
            qoi2analysis = self.qoi2analysis if self.qoi2analysis is not None else 'All'
            print(r'     - {:<23s} : {} '.format('qoi2analysis', qoi2analysis))
            print(r'     - {:<23s} : {} '.format('statistics'  , self.stats)) 
        if self.is_adaptive:
            print(r'   * {:<15s} '.format('Adaptive parameters'))
            print(r'     - {:<23s} : {} '.format('Simulation budget', self.n_budget))
            print(r'     - {:<23s} : {} '.format('Poly degree limit', self.plim))
            print(r'     - {:<23s} : {} '.format('Relvative CV error', self.rel_cv))
            if self.min_r2:
                print(r'     - {:<23s} : {} '.format('R2 bound', self.min_r2))
            if self.rel_mse:
                print(r'     - {:<23s} : {} '.format('Relative MSE', self.rel_mse))
            if self.abs_mse:
                print(r'     - {:<23s} : {} '.format('Absolute MSE', self.abs_mse))
            if self.rel_qoi:
                print(r'     - {:<23s} : {} '.format('Relative QoI', self.rel_qoi))
            if self.abs_qoi:
                print(r'     - {:<23s} : {} '.format('QoI decimal accuracy', self.abs_qoi))
            if self.qoi_val:
                print(r'     - {:<23s} : {} '.format('QoI=0, decimal accuracy', self.qoi_val))

    def _make_output_dir(self):
        """
        WORKING_DIR/
        +-- MODEL_DIR
        |   +-- FIGURE_DIR

        /directory saving data depends on OS/
        +-- MODEL_DIR
        |   +-- DATA_DIR

        """
        working_dir = os.getcwd()
        model_dir   = os.path.join(working_dir, self.model_name)
        figure_dir  = os.path.join(model_dir,r'Figures')
        current_os  = sys.platform
        if current_os.upper()[:3] == 'WIN':
            data_dir_sample = r'G:\My Drive\MUSE_UQ_DATA\Samples' 
            data_dir_result = os.path.join('G:','My Drive','MUSE_UQ_DATA')
            dir_id_result   = self._get_gdrive_folder_id(self.model_name)
            dir_id_sample   = None 
        elif current_os.upper() == 'DARWIN':
            data_dir_sample = r'/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples'
            data_dir_result = r'/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA'
            dir_id_result   = self._get_gdrive_folder_id(self.model_name)
            dir_id_sample   = None 
        elif current_os.upper() == 'LINUX':
            dir_id_result   = None 
            dir_id_sample   = None 
            data_dir_result = r'/home/jinsong/Documents/MUSE_UQ_DATA'
            data_dir_sample = r'/home/jinsong/Documents/MUSE_UQ_DATA/Samples'
        else:
            raise ValueError('Operating system {} not found'.format(current_os))    
        
        data_dir_result  = os.path.join(data_dir_result, self.model_name)
        # dir_id_result = GDRIVE_DIR_ID[self.model_name.upper()] 

        # Create directory for model  
        try:
            os.makedirs(model_dir)
            os.makedirs(data_dir_result)
            os.makedirs(figure_dir)
            # print(r'Data, Figure directories for model {} is created'.format(self.model_name))
        except FileExistsError:
            # one of the above directories already exists
            # print(r'Data, Figure directories for model {} already exist'.format(self.model_name))
            pass
        return data_dir_sample, data_dir_result, dir_id_sample, dir_id_result, figure_dir

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

    def check_overfitting(self, cv_error):
        """
        Return True if overfitting detected

        """

        ## Cross validation error used to check overfitting. 
        ## At least three cv error are needed to check overfitting, [cv1, cv2, cv3]
        ##  Return overfitting warning when 1: cv2 > (1+rel_cv)*cv1; 2: cv3 > cv2 
        ##  two consecutive increasing of cv error, and previous increment is larger than rel_cv

        if len(cv_error) < 3:
            return False
        else:
            cv_error = np.array(cv_error)
            if ((cv_error[-2]- cv_error[-3])/cv_error[-3] > self.rel_cv ) and (cv_error[-2] < cv_error[-1]):
                return True
            else:
                return False

    def is_adaptive_continue(self, nsim_completed, poly_order, **kwargs):
        """
        Stopping criteria for adaptive algorithm
            Algorithm will have a hard stop (return False) when one of following occurs:
                1. nsim_completed >= n_budget
                2. for PCE, poly_order exceeding the largest allowable, plim[-1]
        Arguments:
            nsim_completed: number of evaluations has been done (should be <= self.n_budget)
        Optional:
            poly_order: for PCE model, polynomial order (should be in range self.plim) 

        Return:
            Bool
            return true when the algorithm should continue. i.e.
                1. hard stop on n_budget and poly_order not met 
                2. at least one of the given metric criteria is NOT met
        """

        ## Algorithm stop when nsim_completed >= n_budget 
        if nsim_completed > self.n_budget:
            print(' >! Stopping... Reach simulation budget,  {:d} >= {:d} '.format(nsim_completed, self.n_budget))
            return False

        ## Algorithm stop when poly_order > self.plim[-1]
        ## If poly_order is not given, setting the poly_order value to -inf, which will not affect the checking of other criteria 
        if poly_order > self.plim[1]:
            print(' >! Stopping... Exceed max polynomial order p({:d}) > {:d}'.format(poly_order, self.plim[1]))
            return False


        ### For following metrics, algorithm stop (False) when all of these met.
        ### i.e. If any metric is True ( NOT met), algorithm will continue (return True) 
        is_any_metrics_not_met = []

        # for imetric_name, imetric_value in kwargs.items():
            # threshold_value = getattr(self, imetric_name)
            # if threshold_value is None:
                # print(' Warning: {:s} provided but threshold value was not given'.format(imetric_name))
                # continue


        ## Algorithm continue when r2 <= min_r2 (NOT met, return True)

        try: 
            r2 = kwargs.pop('r2')
        except KeyError:
            try:
                r2 = kwargs.pop('adj_r2')
            except KeyError:
                r2 = None

        if r2 is None:
            is_r2 = False  ## if not defined, then return False. is_any_metrics_not_met=[*, *, False, *, *].any() will not affect the continue of adaptive
        else:
            if self.min_r2 is None:
                raise ValueError(' R squared value provided but R2 threshold was not given. min_r2 = None')
            ## condition met when consecutive two runs meet condition
            ## [r2 is empty (initial step), not defined, not engouth data] or one of last two R2 is less than min_r2
            is_r2 = len(r2) < 2 or  r2[-2] < self.min_r2 or r2[-1] < self.min_r2 

        is_any_metrics_not_met.append(is_r2)

        ## Algorithm continue when mse continue when mse > mse_bound(NOT met, return True)
        mse = kwargs.pop('mse', None)
        if mse is None:
            is_mse = False
            is_any_metrics_not_met.append(is_mse)
        else:
            mse = np.array(mse)
            if self.abs_mse is None and self.rel_mse is None:
                raise ValueError(' MSE value provided but neither rel_mse or abs_mse was given')
            if self.abs_mse:
                is_mse = len(mse) < 2 or mse[-2] > self.abs_mse or mse[-1] > self.abs_mse
                is_any_metrics_not_met.append(is_mse)
            if self.rel_mse:
                if len(mse) < 3:
                    is_mse = True
                else:
                    rel_mse = abs((mse[1:] - mse[:-1])/mse[:-1])
                    is_mse = rel_mse[-2] > self.rel_mse or rel_mse[-1] > self.rel_mse
                is_any_metrics_not_met.append(is_mse)

        ## Algorithm stop when rel_qoi continue when qdiff > self.rel_qoi
        qoi = kwargs.pop('qoi', None)
        if qoi is None:
            is_qoi = False
            is_any_metrics_not_met.append(is_qoi)
        else:
            qoi = np.array(qoi)
            if self.abs_qoi is None and self.rel_qoi is None and self.qoi_val is None:
                raise ValueError(' QoI value provided but none of rel_qoi, abs_qoi, qoi_val was given')

            if self.qoi_val:
                if len(qoi) < 1:
                    is_qoi = True
                else:
                    is_qoi = qoi[-1] > self.qoi_val
                is_any_metrics_not_met.append(is_qoi)
            if self.abs_qoi:
                if len(qoi) < 3:
                    is_qoi = True
                else:
                    qoi_diff = abs((qoi[1:] - qoi[:-1]))
                    is_qoi = qoi_diff[-2] > self.abs_qoi or qoi_diff[-1] > self.abs_qoi
                is_any_metrics_not_met.append(is_qoi)
            if self.rel_qoi:
                if len(qoi) < 3:
                    is_qoi = True
                else:
                    rel_qoi = abs((qoi[1:] - qoi[:-1])/qoi[:-1])
                    is_qoi = rel_qoi[-2] > self.rel_qoi or rel_qoi[-1] > self.rel_qoi
                is_any_metrics_not_met.append(is_qoi)

        ### If any above metric is True ( NOT met), algorithm will continue (return True)
        if not kwargs:
            ### kwargs should be empty by now, otherwise raise valueerror
            is_adaptive = np.array(is_any_metrics_not_met).any()
            return is_adaptive
        else:
            raise ValueError('Given stopping criteria {} not defined'.format(kwargs.keys()))


