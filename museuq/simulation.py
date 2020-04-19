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
import numpy as np
import museuq
from scipy import sparse
from datetime import datetime
from .utilities.classes import Logger
from itertools import compress
import scipy.stats as stats

class Modeling(object):
    """

    """
    def __init__(self, solver, model, params):
        self.solver = solver
        self.model  = model
        self.params = params
        self.dist_x_name = solver.dist_name.lower()
        self.dist_u_name = model.basis.dist_name.lower()
        assert solver.ndim == model.ndim
        self.ndim = solver.ndim

    def get_init_samples(self, n, doe_method='lhs', random_state=None, **kwargs):
        """
        Get initial sample design, return samples in U,X space and results y

        Arguments:
            n: int, number of samples to return
            solver: solver used to return y
            pce_model: for LHS, provide the u distributions needed to be sampled
                        for cls, need to know PCE.deg such that in unbounded case to scale u

        """

        if doe_method.lower() == 'lhs':
            doe = museuq.LHS(self.model.basis.dist_u)
            z, u= doe.samples(n, random_state=random_state)
            x = self.solver.map_domain(u, z) ## z_train is the cdf of u_train
        else:
            raise NotImplementedError
            # u_cand = kwargs['u_cand']
            # optimality = kwargs.get('optimality', None)
            # p = kwargs['p']
            # sample_selected= self.sample_selected 
            # u_cand_p = p **0.5* u_cand if (doe_method.lower()=='cls' and self.model.basis.dist_name=='norm') else u_cand
            # _,u = get_train_data(n, u_cand_p, doe_method, optimality=optimality, sample_selected=sample_selected, basis=self.model.basis)
            # x = self.solver.map_domain(u, self.model.basis.dist_u)
        y = self.solver.run(x)
        return u, x, y

    def get_candidate_data(self, **kwargs):
        """
        Return canndidate samples in u space
        """
        try:
            n = kwargs['n']
            self.params.n_cand = n
        except KeyError:
            n = self.params.n_cand
        doe_method = self.params.doe_method.lower()
        data_dir = os.path.join(self.params.data_dir_sample, doe_method.upper(), self.dist_x_name.capitalize()) 
        try:
            self.filename_candidates = kwargs['filename']
            try:
                data = np.load(self.filename_candidates)
            except FileNotFoundError:
                data = np.load(os.path.join(data_dir, self.filename_candidates))
            u_cand = data[:self.ndim,:n].reshape(self.ndim, -1) ## will raise error when samples files smaller than n
            self.filename_optimality = kwargs.get('filename_optimality', None)

        except KeyError:
            if doe_method.startswith('mcs'):
                self.filename_candidates = r'DoE_McsE6R0.npy'
                data = np.load(os.path.join(data_dir, self.filename_candidates))
                u_cand = data[:self.ndim,:n].reshape(self.ndim, -1) ## will raise error when samples files smaller than n

            elif doe_method.startswith('cls') or doe_method == 'reference':
                if self.dist_x_name.startswith('norm'):
                    self.filename_candidates = r'DoE_ClsE6d{:d}R0.npy'.format(self.ndim)
                else:
                    self.filename_candidates = r'DoE_ClsE6dR0.npy'
                data  = np.load(os.path.join(data_dir, self.filename_candidates))
                u_cand = data[:self.ndim,:n].reshape(self.ndim, -1)
            else:
                raise ValueError('DoE method {:s} not defined'.format(doe_method))

        return u_cand

    def get_test_data(self, solver, pce_model, filename = r'DoE_McsE6R9.npy', **kwargs):
        """
        Return test data. 

        Test data should always be in X-space. The correct sequence is X->y->u

        To be able to generate MCS samples for X, we use MCS samples in Samples/MCS, noted as z here

        If already exist in simparams.data_dir_result, then load and return
        else, run solver

        """
        
        data_dir_result = os.path.join(self.params.data_dir_result, 'TestData')
        try: 
            os.makedirs(data_dir_result)
        except OSError as e:
            pass

        n = kwargs.get('n', self.params.n_test)
        assert solver.ndim == pce_model.ndim
        ndim = solver.ndim
        
        self.filename_test = '{:s}_{:d}{:s}_'.format(solver.nickname, ndim, pce_model.basis.nickname) + filename
        if self.params.doe_method.lower() == 'cls':
            self.filename_test = self.filename_test.replace('Mcs', 'Cls')

        try:
            data_set = np.load(os.path.join(data_dir_result, self.filename_test))
            print('   > Retrieving test data from {}'.format(os.path.join(data_dir_result, self.filename_test)))
            assert data_set.shape[0] == 2*ndim+1
            u_test = data_set[      :  ndim,:n] if n > 0 else data_set[     :  ndim,:]
            x_test = data_set[ndim  :2*ndim,:n] if n > 0 else data_set[ndim :2*ndim,:]
            y_test = data_set[-1,           :n] if n > 0 else data_set[-1,          :]

        except FileNotFoundError:
            ### 1. Get MCS samples for X
            if solver.dist_name.lower() == 'uniform':
                self.filename_test =  os.path.join(self.params.data_dir_sample, 'MCS','Uniform', filename)
                print('   > Solving test data from {} '.format(self.filename_test))
                data_set = np.load(self.filename_test)
                z_test = data_set[:ndim,:] 
                x_test = solver.map_domain(z_test, [stats.uniform(-1,2),] * ndim)
            elif solver.dist_name.lower().startswith('norm'):
                self.filename_test = os.path.join(self.params.data_dir_sample, 'MCS','Norm', filename)
                print('   > Solving test data from {} '.format(self.filename_test))
                data_set= np.load(self.filename_test)
                z_test  = data_set[:ndim,:] 
                x_test  = solver.map_domain(z_test, [stats.norm(0,1),] * solver.ndim)
            else:
                raise ValueError
            y_test = solver.run(x_test)

            ### 2. Mapping MCS samples from X to u
            ###     dist_u is defined by pce_model
            ### Bounded domain maps to [-1,1] for both mcs and cls methods. so u = z
            ### Unbounded domain, mcs maps to N(0,1), cls maps to N(0,sqrt(0.5))
            u_test = 0.0 + z_test * np.sqrt(0.5) if self.is_cls_unbounded() else z_test
            data   = np.vstack((u_test, x_test, y_test.reshape(1,-1)))
            np.save(os.path.join(data_dir_result, self.filename_test), data)
            print('   > Saving test data to {} '.format(data_dir_result))

            u_test = u_test[:,:n] if n > 0 else u_test
            x_test = x_test[:,:n] if n > 0 else x_test
            y_test = y_test[  :n] if n > 0 else y_test

        return u_test, x_test, y_test

    def get_train_data(self, size, u_cand, basis=None, active_basis=None):
        """
        Return train data from candidate data set. All sampels are in U-space
        size is how many MORE to be sampled besides those alrady existed in sample_selected

        Arguments:
            size        : size of samples, (r, n): size n repeat r times
            u_cand      : ndarray, candidate samples in U-space to be chosen from
            sample_selected: samples already selected, need to be removed from candidate set to get n sampels
            basis       : When optimality is 'D' or 'S', one need the design matrix in the basis selected
            active_basis: activated basis used in optimality design

        """
        size = tuple(np.atleast_1d(size))

        if len(size) == 1 or size[0] == 1:
            size = np.prod(size)
            u_new, u_all = self._choose_samples_from_candidates(size, u_cand,
                    selected=self.sample_selected, basis=basis, active_basis=active_basis)
            return u_new, u_all

        elif len(size) == 2:
            u_train_new = []
            u_train_all = []
            if len(self.sample_selected) == 0:
                self.sample_selected = [[] for _ in range(size[0])]
            for r in range(size[0]):
                u_new, u_all  = self._choose_samples_from_candidates(size[1], u_cand, 
                        selected=self.sample_selected[r], basis=basis, active_basis=active_basis)
                u_train_new.append(u_new)
                u_train_all.append(u_all)
            u_train_new = np.array(u_train_new)
            u_train_all = np.array(u_train_all)
            return u_train_new, u_train_all

    def _choose_samples_from_candidates(self, n, u_cand, selected=[], **kwargs):
        """
        Return train data from candidate data set. All sampels are in U-space

        Arguments:
            n           : int, size of new samples in addtion to selected elements
            u_cand      : ndarray, candidate samples in U-space to be chosen from
            selected    : samples already selected, need to be removed from candidate set to get n sampels
            basis       : When optimality is 'D' or 'S', one need the design matrix in the basis selected
            active_basis: activated basis used in optimality design

        """
        
        if self.params.optimality is None:
            rows_slct_new = []
            while len(rows_slct_new) < n:
                ### random index set
                random_idx = set(np.random.randint(0, u_cand.shape[1], size=n*10))
                ### remove selected passed
                random_idx = random_idx.difference(set(selected))
                ### remove selected chosen in this step
                random_idx = random_idx.difference(set(rows_slct_new))
                ### update new samples set
                rows_slct_new += list(random_idx)
            rows_slct_new = rows_slct_new[:n]
            selected += rows_slct_new

        elif self.params.optimality:
            basis = kwargs['basis']
            active_basis = kwargs.get('active_basis', None)

            doe = museuq.OptimalDesign(self.params.optimality, curr_set=selected)
            ### Using full design matrix, and precomputed optimality file exists only for this calculation
            if active_basis == 0 or active_basis is None:
                if self._check_precomputed_optimality(basis, selected):
                    print('     - {:<23s} : {}/{}, File: {}'.format('Optimal design based on ', basis.num_basis, basis.num_basis, self.filename_optimality))
                    rows_slct_new = self.optimality_rows[len(selected):len(selected)+n]
                    selected += list(rows_slct_new)
                else:
                    print('     - {:<23s} : {}/{}'.format('Optimal design based on ', basis.num_basis, basis.num_basis))
                    X = basis.vandermonde(u_cand)
                    if self.params.doe_method.lower().startswith('cls'):
                        X  = X.shape[1]**0.5*(X.T / np.linalg.norm(X, axis=1)).T
                    rows_slct_new = doe.samples(X, n, orth_basis=True)
            ### Using partial columns
            else:
                active_index = np.array([i for i in range(basis.num_basis) if basis.basis_degree[i] in active_basis])
                print('     - {:<23s} : {}/{}'.format('Optimal design based on ', len(active_index), basis.num_basis))
                X = X[:, active_index]
                if self.params.doe_method.lower().startswith('cls'):
                    X  = X.shape[1]**0.5*(X.T / np.linalg.norm(X, axis=1)).T
                rows_slct_new = doe.samples(X, n, orth_basis=True)

        self._check_duplicate_samples(selected)
        samples_new = u_cand[:,rows_slct_new]
        samples_all = u_cand[:,selected]

        return samples_new, samples_all

    def _check_precomputed_optimality(self, basis, selected):
        """

        """

        try:
            self.optimality_rows = np.load(self.filename_optimality)
        except (AttributeError, FileNotFoundError) as e:
            self.filename_optimality = 'DoE_{:s}E{:s}R0_{:d}{:s}{:d}_{:s}.npy'.format(self.params.doe_method.capitalize(),
                    '{:.0E}'.format(self.params.n_cand)[-1], self.ndim, self.model.basis.nickname, basis.deg, self.params.optimality)
            try:
                self.optimality_rows = np.squeeze(np.load(os.path.join(self.params.data_dir_precomputed_optimality, self.filename_optimality)))
                if len(selected):
                    if np.array_equal(self.optimality_rows[:len(selected)], np.array(selected)):
                        return True
                    else:
                        print('Precomputed optimality found but do not match selected_samples, will run Optimality Design directly')
                        return False
                else:
                    return True
            except FileNotFoundError: 
                print('FileNotFoundError: [Errno 2] No such file or directory: {}'.format(self.filename_optimality))
                return False

        except:
            return False

    def cal_cls_weight(self, u, basis):
        """
        Calculate weight for CLS based on Christoffel function evaluated in U-space
        """
        X = basis.vandermonde(u)
        ### reproducing kernel
        Kp = np.sum(X* X, axis=1)
        w  = basis.num_basis / Kp
        return w

    def is_cls_unbounded(self):
        return  self.params.doe_method.lower().startswith('cls') and self.dist_u_name.startswith('norm')

    def _check_duplicate_samples(self, samples):
        if len(samples) == 0:
            raise ValueError('Sample size is 0')
        if len(samples) != len(np.unique(samples)):
            raise ValueError('Find duplciate samples in x ')

    def rescale_data(self, X, sample_weight):
        """Rescale data so as to support sample_weight"""
        n_samples = X.shape[0]
        sample_weight = np.asarray(sample_weight)
        if sample_weight.ndim == 0:
            sample_weight = np.full(n_samples, sample_weight,
                                    dtype=sample_weight.dtype)
        sample_weight = np.sqrt(sample_weight)
        sw_matrix = sparse.dia_matrix((sample_weight, 0),
                                      shape=(n_samples, n_samples))
        X = sw_matrix @ X
        return X

    def info(self):
        pass

class Parameters(object):
    """
    Define general parameter settings for simulation running and post data analysis. 
    System parameters will be different depending on solver 

    Arguments:
        solver.nickname: 
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

    def __init__(self, solver, **kwargs):
        sys.stdout  = Logger()
        self.solver = solver
        self.optimality = None
        ###------------- Adaptive setting -----------------------------
        self.is_adaptive= False
        self.plim     = kwargs.get('plim'     , None  )  ## polynomial degree limit
        self.n_budget = kwargs.get('n_budget' , None  )
        self.min_r2   = kwargs.get('min_r2'   , None  )  ## minimum adjusted R-squared threshold value to take
        self.rel_mse  = kwargs.get('rel_mse'  , None  )  ## Relative mean square error 
        self.abs_mse  = kwargs.get('abs_mse'  , None  )  ## Absolute mean square error
        self.rel_qoi  = kwargs.get('rel_qoi'  , None  )  ## Relative error for QoI, i.e. percentage difference relative to previous simulation
        self.abs_qoi  = kwargs.get('abs_qoi'  , None  )  ## Absolute error for QoI, i.e. decimal accuracy 
        self.qoi_val  = kwargs.get('qoi_val'  , None  )  ## QoI value up to decimal accuracy 
        self.rel_cv   = kwargs.get('rel_cv'   , 0.05  )  ## percentage difference relative to previous simulation
        if self.plim is not None or self.n_budget is not None:
            self.is_adaptive = True
        
    def info(self):
        print(r'------------------------------------------------------------')
        print(r'>>> Parameters for Model: {}'.format(self.solver.name))
        print(r'------------------------------------------------------------')
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

    def update(self):
        self.set_params()
        self.update_dir()
        self._get_tag()
        self.data_dir_precomputed_optimality = os.path.join(self.data_dir_sample, 'OED')

    def update_dir(self, **kwargs):
        """
        update directories for working and data saving.
            Takes either a string argument or a dictionary argument

        self.update_dir(MDOEL_NAME)  (set as default and has priority).
            if solver.nickname is given, kwargs is ignored
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
        ### ['mean', 'std', 'skewness', 'kurtosis', 'absmax', 'absmin', 'up_crossing']
        self.post_params    = kwargs.get('post_params'  , [None, None])
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

    def set_adaptive_parameters(self, **kwargs):
        self.is_adaptive=True
        for ikey, ivalue in kwargs.items():
            try:
                setattr(self, ikey, ivalue)
            except:
                raise KeyError

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
        model_dir   = os.path.join(working_dir, self.solver.nickname)
        figure_dir  = os.path.join(model_dir,r'Figures')
        current_os  = sys.platform
        if current_os.upper()[:3] == 'WIN':
            data_dir_sample = r'G:\My Drive\MUSE_UQ_DATA\Samples' 
            data_dir_result = os.path.join('G:','My Drive','MUSE_UQ_DATA')
            dir_id_result   = self._get_gdrive_folder_id(self.solver.nickname)
            dir_id_sample   = None 
        elif current_os.upper() == 'DARWIN':
            data_dir_sample = r'/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples'
            data_dir_result = r'/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA'
            dir_id_result   = self._get_gdrive_folder_id(self.solver.nickname)
            dir_id_sample   = None 
        elif current_os.upper() == 'LINUX':
            dir_id_result   = None 
            dir_id_sample   = None 
            data_dir_result = r'/home/jinsong/Documents/MUSE_UQ_DATA'
            data_dir_sample = r'/home/jinsong/Documents/MUSE_UQ_DATA/Samples'
        else:
            raise ValueError('Operating system {} not found'.format(current_os))    
        data_dir_result  = os.path.join(data_dir_result, self.solver.nickname)
        # Create directory for model  
        try:
            os.makedirs(data_dir_result)
            os.makedirs(figure_dir)
        except FileExistsError:
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

    def _get_tag(self):
        if self.optimality:
            self.tag = '{:s}{:s}_{:s}'.format(self.doe_method.capitalize(), self.optimality, self.fit_method.capitalize())
        else:
            self.tag = '{:s}_{:s}'.format(self.doe_method.capitalize(), self.fit_method.capitalize())

