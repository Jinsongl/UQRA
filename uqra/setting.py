#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import os, sys, math, platform
import numpy as np
import uqra
from scipy import sparse
from .utilities.classes import Logger
from itertools import compress
import scipy.stats as stats
from tqdm import tqdm
import copy

### ----------------- Base Classes -----------------
class Data(object):
    def __init__(self):
        pass

class Parameters(object):
    def __init__(self):
        pass

### ----------------- Experiment Parameters() -----------------
class ExperimentParameters(Parameters):
    """
    UQRA Parameters for Experimental Design
    """
    def __init__(self, doe_sampling, optimality=None):
        self.doe_sampling = doe_sampling.upper()
        self.optimality   = optimality.upper() if isinstance(optimality, str) else optimality
        if self.doe_sampling == 'LHS' and self.optimality is not None:
            print(" [WARNING]: Optimality {:s} not applicable for LHS, set 'optimaltiy' to None")
            self.optimality = None

        self._default_data_dir()
        self._check_wiener_askey_polynomials()

    def update_filenames(self, filename_template=None, **kwargs):
        """
        Create/update filenames related to data in/output
        """
        ### Check for parameters
        self._check_wiener_askey_polynomials() ### also return distribution name to specified polynomials
        if self.optimality is not None:
            try:
                ndim, deg   = self.ndim, self.deg
                poly_name   = self.poly_name
                doe_sampling= self.doe_sampling.capitalize()
                num_cand    = self.num_cand
            except AttributeError:
                raise ValueError('ExperimentParameters.update_filenames: missing attributes:  \
                        [ndim, deg, poly_name, doe_sampling, num_cand] ')
        else:
            try:
                ndim   = self.ndim
                poly_name   = self.poly_name
                doe_sampling= self.doe_sampling.capitalize()
            except AttributeError:
                raise ValueError('ExperimentParameters.update_filenames: missing attributes:  \
                        [ndim, poly_name, doe_sampling] ')


        ## 1 user defined filenames: direct assign, 1st priority
        self.fname_cand  = kwargs.get('filename_cand'  , None)
        self.fname_design= kwargs.get('filename_design', None)

        isFileNameAssigned = np.array([self.fname_cand, self.fname_design]) != None
        if isFileNameAssigned.all():
            ## user defined filenames have first priority
            pass

        ## 2: filename template function is given
        elif filename_template:
            ### filenames are based on given template function
            ### but user defined filenames are first priority
            if self.fname_cand is None:
                self.fname_cand = filename_template

            if self.fname_design is None:
                def fname_design(s):
                    if callable(self.fname_cand):
                        fname_design = os.path.splitext(self.fname_cand(s))[0]
                    else:
                        fname_design = os.path.splitext(self.fname_cand)[0]  ## remove extension .npy if any
                    res = fname_design + '_{:d}{:s}{:s}.npy'.format(ndim, poly_name[:3], str(deg))
                    return res 
                self.fname_design = fname_design
            
        ## 3: if none of above are given, will return system defined filenames 
        else:
            if poly_name.lower().startswith('leg'):
                distname = 'uniform'
            elif poly_name.lower().startswith('hem'):
                distname = 'norm'
            else:
                raise NotImplementedError
            if doe_sampling.lower() == 'lhs':
                self.fname_cand   = lambda r: None
                self.fname_design = lambda n: r'DoE_Lhs{:d}_{:d}{:s}.npy'.format(n,ndim, distname)

            elif doe_sampling[:3].lower() == 'mcs':
                self.fname_cand   = lambda s: r'DoE_{:s}E6R{:d}_{:s}.npy'.format(doe_sampling, s, distname)
                self.fname_design = lambda s: r'DoE_{:s}E{:d}R{:d}_{:d}{:s}{:s}.npy'.format(
                        doe_sampling, math.ceil(np.log10(num_cand)), s, ndim, poly_name[:3], str(deg))

            elif doe_sampling[:3].lower() == 'cls':
                self.fname_cand   = lambda s: r'DoE_{:s}E6D{:d}R{:d}.npy'.format(doe_sampling, ndim, s)
                self.fname_design = lambda s: r'DoE_{:s}E{:d}R{:d}_{:d}{:s}{:s}.npy'.format(
                        doe_sampling, math.ceil(np.log10(num_cand)), s, ndim, poly_name[:3], str(deg))
            else:
                self.fname_cand   = lambda s : r'DoE_{:s}E6R{:d}_{:s}.npy'.format(doe_sampling, s, distname)
                self.fname_design = lambda s : r'DoE_{:s}E{:d}R{:d}_{:d}{:s}{:s}.npy'.format(
                        doe_sampling, math.ceil(np.log10(num_cand)), s, ndim, poly_name[:3], str(deg))

    def update_nicknames(self):
        """
        Return a list of nickname(s) for doe_sampling and all the doe_optimality specified
        """
        try:
            doe_sampling = self.doe_sampling
            doe_optimality = self.optimality
            if not isinstance(doe_optimality, (list, tuple)):
                doe_optimality = [doe_optimality,]
        except AttributeError:
            raise ValueError(' doe_sampling and doe_optimality attributes must given to update nicknames')
        self.nicknames = [self.doe_nickname(doe_sampling, ioptimality) for ioptimality in doe_optimality]

    def doe_nickname(self):
        """
        Return DoE nickname for one specific doe_sampling and doe_optimality set
        """
        doe_sampling = self.doe_sampling
        doe_optimality=self.optimality
        if str(doe_optimality).lower() == 'none':
            nickname = str(doe_sampling).capitalize()
        else:
            assert doe_optimality.isalpha() and len(doe_optimality) ==1
            nickname = str(doe_sampling).capitalize()+str(doe_optimality).upper()
        return nickname

    def update_output_dir(self, **kwargs):
        """
        update directories for working and data saving.
            Takes either a string argument or a dictionary argument

        self.update_output_dir(MDOEL_NAME)  (set as default and has priority).
            if solver.nickname is given, kwargs is ignored
        self.update_output_dir(pwd=, data_dir_result=, figure_dir=)

        Updating directories of
            pwd: present working directory, self.pwd
            data_dir_result: directory saving all data, self.data_dir_result
            figure_dir: directory saving all figures, self.figure_dir
        """
        self.data_dir_cand    = kwargs.get('data_dir_cand'   , self.data_dir_cand    )
        self.data_dir_optimal = kwargs.get('data_dir_optimal', self.data_dir_optimal )

    def _default_data_dir(self):
        """
        WORKING_DIR/
        +-- MODEL_DIR
        |   +-- FIGURE_DIR

        /directory saving data depends on OS/
        +-- MODEL_DIR
        |   +-- DATA_DIR

        """
        current_os  = sys.platform
        if current_os.upper()[:3] == 'WIN':
            data_dir_optimal= os.path.join('G:\\','My Drive','MUSE_UQ_DATA', 'ExperimentalDesign', 'Random_Optimal')
            data_dir_cand   = os.path.join('G:\\','My Drive','MUSE_UQ_DATA', 'ExperimentalDesign', 'Random')
        elif current_os.upper() == 'DARWIN':
            data_dir_optimal= r'/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/ExperimentalDesign/Random_Optimal'
            data_dir_cand   = r'/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/ExperimentalDesign/Random'
        elif current_os.upper() == 'LINUX':
            data_dir_optimal= r'/home/jinsong/Documents/MUSE_UQ_DATA/ExperimentalDesign/Random_Optimal'
            data_dir_cand   = r'/home/jinsong/Documents/MUSE_UQ_DATA/ExperimentalDesign/Random'
        else:
            raise ValueError('Operating system {} not found'.format(current_os))    
        try:
            if self.doe_sampling.lower() == 'lhs':
                data_dir_optimal = os.path.join(os.path.split(data_dir_optimal)[0], 'LHS')
                data_dir_cand    = None
            else:
                pass
        except AttributeError as message:
            print(message)

        self.data_dir_cand    = data_dir_cand
        self.data_dir_optimal = data_dir_optimal 

    def _check_wiener_askey_polynomials(self):
        """
        check and set underlying Wiener-Askey distributions
        """
        try:
            doe_sampling = self.doe_sampling.upper()
            poly_name    = self.poly_name.upper()

            if doe_sampling == 'MCS' and poly_name == 'LEG':
                self.dist_xi     = stats.uniform(-1,2)
                self.xi_distname = 'uniform'

            elif doe_sampling == 'MCS' and poly_name == 'HEME':
                self.dist_xi     = stats.norm(0,1)
                self.xi_distname = 'norm'

            elif doe_sampling == 'CLS1' and poly_name == 'LEG':
                self.dist_xi     = stats.uniform(-1,2)
                self.xi_distname = 'uniform'

            elif doe_sampling == 'CLS4' and poly_name == 'HEM':
                self.dist_xi     = stats.norm(0,np.sqrt(0.5))
                self.xi_distname = 'norm'

            elif doe_sampling == 'LHS'and poly_name == 'LEG':
                self.dist_xi     = stats.uniform(-1,2)
                self.xi_distname = 'uniform'

            elif doe_sampling == 'LHS'and poly_name == 'HEME':
                self.dist_xi     = stats.norm(0,1)
                self.xi_distname = 'norm'
            else:
                raise ValueError(' {:s}-{:s} is either not compatible or defined'.format(doe_sampling, poly_name))
        except:
            pass

    def sampling_weight(self, w=None):
        """
        Return a weight function due to sampling
        If w is given, then has the highest priority. 
        otherwise, return based on doe_sampling
        """
        if w is not None:
            # if callable(w):
                # return w
            # else:
                # w = lambda x: w
            return w
        elif self.doe_sampling.lower().startswith('cls'):
            return 'christoffel'
        elif self.doe_sampling.lower()[:3] in ['mcs', 'lhs']:
            return None
        else:
            raise ValueError('Sampling weight function is not defined')

    def get_samples(self, x, poly, n, x0=[], active_index=None,  
            initialization='RRQR', return_index=False):
        """

        return samples based on UQRA
        """
        ### check arguments
        n = uqra.check_int(n)
        if self.doe_sampling.lower().startswith('lhs'):
            assert self.optimality is None
            dist_xi = poly.weight
            doe = uqra.LHS([dist_xi, ] *self.ndim)
            x_optimal = doe.samples(size=n)
            res = x_optimal

        else:
            x = np.array(x, copy=False, ndmin=2)
            d, N = x.shape
            assert d == self.ndim 
            ## expand samples if it is unbounded cls
            x = poly.deg**0.5 * x if self.doe_sampling in ['CLS4', 'CLS5'] else x
            if len(x0) == 0:
                x0 = np.empty((d,0))
            else:
                x0 = np.array(x0, copy=False, ndmin=2)
                assert d == x0.shape[0]

            x_optimal      = []
            idx_selected   = list(uqra.common_vectors(x0, x))
            initialization = initialization if len(idx_selected) == 0 else idx_selected

            if self.doe_sampling.lower().startswith('mcs'):
                if str(self.optimality).lower() == 'none':
                    idx = list(set(np.arange(self.num_cand)).difference(set(idx_selected)))[:n] 
                else:
                    X = poly.vandermonde(x_)
                    X = X if active_index is None else X[:, active_index]
                    uqra.blockPrint()
                    doe = uqra.OptimalDesign(X)
                    idx = doe.samples(self.optimality, n, initialization=initialization) 
                    uqra.enablePrint()

                idx = uqra.list_diff(idx, idx_selected)
                assert len(idx) == n
                x_optimal = x[:, idx]

            elif self.doe_sampling.lower().startswith('cls'):
                if str(self.optimality).lower() == 'none':
                    idx = list(set(np.arange(self.num_cand)).difference(set(idx_selected)))[:n] 
                else:
                    X = poly.vandermonde(x_)
                    X = X if active_index is None else X[:, active_index]
                    X = poly.num_basis**0.5*(X.T / np.linalg.norm(X, axis=1)).T
                    uqra.blockPrint()
                    doe = uqra.OptimalDesign(X)
                    idx = doe.samples(self.optimality, n, initialization=initialization) 
                    uqra.enablePrint()

                idx = uqra.list_diff(idx, idx_selected)
                assert len(idx) == n
                x_optimal = x[:, idx]
            else:
                raise ValueError
            res = (x_optimal, idx) if return_index else x_optimal
        return res

### ----------------- Modeling Parameters() -----------------
class Modeling(Parameters):
    """

    """
    def __init__(self, name):
        self.name = name.upper()

    # def __init__(self, solver, model, params):
        # self.solver = solver
        # self.model  = model
        # self.params = params
        # assert solver.ndim == model.ndim
        # self.ndim = solver.ndim
        # self.xi_distname = params.xi_distname
        # self.x_distname = solver.dist_name
        # assert self.xi_distname == model.orth_poly.dist_name.lower()

    def get_train_data(self, size, u_cand, u_train=None, active_basis=None, orth_poly=None):
        """
        Return train data from candidate data set. All samples are in U-space (with pluripotential equilibrium measure nv(x))

        Arguments:
            n           : int, size of new samples in addtion to selected elements
            u_cand      : ndarray, candidate samples in U-space to be chosen from
            u_train     : selected train samples from candidate data
            active_basis: list of active basis degree for activated basis used in doe_optimality design

        """

        size = int(size)
        u_train_new = []
        u_train_all = []

        orth_poly    = self.model.orth_poly if orth_poly is None else orth_poly
        active_basis = orth_poly.basis_degree if active_basis is None else active_basis
        active_index = [i for i in range(orth_poly.num_basis) if orth_poly.basis_degree[i] in active_basis]
        assert len(active_index) == len(active_basis)
        assert len(active_index) > 0

        ### get theindices of already selected train samples in candidate samples 
        selected_index = list(uqra.common_vectors(u_train, u_cand))

        if self.params.doe_optimality is None:
            ### for non doe_optimality design, design matrix X is irrelative
            row_index_adding = []
            while len(row_index_adding) < size:
                ### random index set
                random_idx = set(np.random.randint(0, u_cand.shape[1], size=size*10))
                ### remove selected_index chosen in this step
                random_idx = random_idx.difference(set(row_index_adding))
                ### remove selected_index passed
                random_idx = random_idx.difference(set(selected_index))
                ### update new samples set
                row_index_adding += list(random_idx)
            row_index_adding = row_index_adding[:size]
            u_new = u_cand[:,row_index_adding]

        else:
            doe = uqra.OptimalDesign(self.params.doe_optimality, selected_index=selected_index)
            ### Using full design matrix, and precomputed doe_optimality file exists only for this calculation
            X = orth_poly.vandermonde(u_cand)
            X = X[:,active_index]
            if self.params.doe_candidate.lower().startswith('cls'):
                X  = X.shape[1]**0.5*(X.T / np.linalg.norm(X, axis=1)).T
            row_index_adding = doe.get_samples(X, size, orth_basis=True)
            u_new = u_cand[:,row_index_adding]
        return u_new

    def update_basis(self):
        if self.name == 'PCE':
            self._update_pce_dist(self.basis)
        else:
            raise NotImplementedError

    def _update_pce_dist(self, poly_name):
        """
        set xi distributions
        """
        poly_name = poly_name.upper()
        if poly_name == 'LEG':
            self.dist_xi     = stats.uniform(-1, 2) 
            self.xi_distname = 'uniform'
        elif poly_name == 'HEM':
            self.dist_xi     = stats.norm(0, np.sqrt(0.5))
            self.xi_distname = 'norm'
        elif poly_name == 'HEME':
            self.dist_xi     = stats.norm(0, 1)
            self.xi_distname = 'norm'
        else:
            raise NotImplementedError

    def cal_weight(self, u, active_basis=None, orth_poly=None):
        """
        Calculate weight for CLS based on Christoffel function evaluated in U-space
        """
        orth_poly = self.model.orth_poly if orth_poly is None else orth_poly
        active_basis = orth_poly.basis_degree if active_basis is None else active_basis
        active_index = [i for i in range(orth_poly.num_basis) if orth_poly.basis_degree[i] in active_basis]
        assert len(active_index) == len(active_basis)
        assert len(active_index) > 0

        if self.params.doe_candidate.startswith('cls'):
            X = orth_poly.vandermonde(u)
            X = X[:, active_index]
            ### reproducing kernel
            Kp = np.sum(X* X, axis=1)
            P  = len(active_index)
            w  = P / Kp
        else:
            w = None
        return w

    def info(self):
        pass


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

    def map_domain(self, u, dist_u):

        if self.dist_xi.dist.name == 'uniform' and dist_u.dist.name == 'uniform':
            ua, ub = dist_u.support()
            loc_u, scl_u = ua, ub-ua
            xa, xb = self.dist_xi.support()
            loc_x, scl_x = xa, xb-xa 
            x = (u-loc_u)/scl_u * scl_x + loc_x

        elif self.dist_xi.dist.name == 'norm' and dist_u.dist.name == 'norm':
            mean_u = dist_u.mean()
            mean_x = self.dist_xi.mean()
            std_u  = dist_u.std()
            std_x  = self.dist_xi.std()
            x = (u-mean_u)/std_u * std_x + mean_x
        else:
            x = self.dist_xi.ppf(dist_u.cdf(u))
        return x
 
### ----------------- Simulation Parameters() -----------------
class Simulation(Parameters):
    """
    Simulation class with settings to run UQRA modeling

    Arguments:
        solver: solver to be run, function
        doe_method: list/tuple of str/None specified [method to get candidate samples, doe_optimality]
        doe_params  = [doe_method, doe_rule, doe_order]
        time_params = [time_start, time_ramp, time_max, dt]
        post_params = [out_responses=[0,], stats=[1,1,1,1,1,1,0]]
            stats: [mean, std, skewness, kurtosis, absmax, absmin, up_crossing]
        sys_def_params: array of parameter sets defining the solver
            if no sys_def_params is required, sys_def_params = None
            e.g. for duffing oscillator:
            sys_def_params = np.array([0,0,1,1,1]).reshape(1,5) # x0,v0, zeta, omega_n, mu 
        normalize: 
    """
    def __init__(self, solver, model_params, doe_params):
        self.solver     = solver
        self.doe_params = doe_params
        self.model_params= model_params
        self._default_output_dir()

    def info(self):
        print(r'------------------------------------------------------------')
        print(r' > Parameters for Model: {}'.format(self.solver.name))
        print(r'   - DoE candidate  : {}'.format(self.doe_candidate.upper()))
        print(r'   - DoE optimality : {}'.format(self.doe_optimality.upper()))
        print(r'   - fit method     : {}'.format(self.fit_method.capitalize()))
        print(r'------------------------------------------------------------')
        print(r' > Distributions: U,X')
        print(r'   - X distribution : {}'.format(self.x_dist.name))
        print(r'   - U distribution : {}, (mu, std)=({:.2f},{:.2f}), support={}'.format(
            self.xi_distname,self.dist_xi[0].mean(), self.dist_xi[0].std(), self.dist_xi[0].support()))
        print(r'------------------------------------------------------------')
        print(r' > DIRECTORIES:')
        print(r'   - Working Dir: {}'.format(os.getcwd()))
        print(r'   - Figure  Dir: {}'.format(self.figure_dir))
        print(r'   - Result  Dir: {}'.format(self.data_dir_result))
        print(r'   - Samples Dir: {}'.format(self.data_dir_cand))

    def update_filenames(self, filename_template=None, **kwargs):
        """
        Create/update filenames for testing 
        """
        ndim        = self.solver.ndim
        poly_name   = self.doe_params.poly_name.capitalize()
        doe_sampling= self.doe_params.doe_sampling.capitalize()
        xi_distname = self.model_params.xi_distname

        ## 1 user defined filenames: direct assign, 1st priority
        self.fname_test  = kwargs.get('filename_test'   , None)
        self.fname_testin= kwargs.get('fllename_testin' , None)

        isFileNameAssigned = np.array([self.fname_test, self.fname_testin]) != None
        if isFileNameAssigned.all():
            ## user defined filenames have first priority
            pass

        ## 2: filename template function is given
        elif filename_template:
            ### filenames are based on given template function
            ### but user defined filenames are first priority
            if self.fname_testin is None:
                self.fname_testin = lambda s: filename_template(s)+'.npy' 

            if self.fname_test is None:
                def fname_test(s):
                    fname_testin = self.fname_testin(s) if callable(self.fname_testin) else self.fname_testin
                    res = '_'.join([self.solver.nickname, fname_testin])
                    return res 
                self.fname_test = fname_test

        ## 3: if none of above are given, will return system defined filenames 
        else:
            self.fname_testin= lambda s: r'DoE_McsE6R{:d}_{:s}.npy'.format((s+1)%10, xi_distname)
            self.fname_test  = lambda s: r'{:s}_McsE6R{:d}.npy'.format(self.solver.nickname, (s+1) %10)

    def update_output_dir(self, **kwargs):
        """
        update directories for working and data saving.
            Takes either a string argument or a dictionary argument

        self.update_output_dir(MDOEL_NAME)  (set as default and has priority).
            if solver.nickname is given, kwargs is ignored
        self.update_output_dir(pwd=, data_dir_result=, figure_dir=)

        Updating directories of
            pwd: present working directory, self.pwd
            data_dir_result: directory saving all data, self.data_dir_result
            figure_dir: directory saving all figures, self.figure_dir
        """
        self.figure_dir      = kwargs.get('figure_dir'       , self.figure_dir     )
        self.data_dir_testin = kwargs.get('data_dir_testin'  , self.data_dir_testin)
        self.data_dir_test   = kwargs.get('data_dir_test'    , self.data_dir_test  )
        self.data_dir_result = kwargs.get('data_dir_result'  , self.data_dir_result)


    def get_init_samples(self, n, doe_candidate=None, random_state=None, **kwargs):
        """
        Get initial sample design, return samples in U space 

        Arguments:
            n: int, number of samples to return
            doe_candidate: method to get candidate samples if optimality is used
            pce_model: for LHS, provide the u distributions needed to be sampled
                        for cls, need to know PCE.deg such that in unbounded case to scale u

        """
        doe_candidate = self.doe_candidate if doe_candidate is None else doe_candidate

        if doe_candidate.lower() == 'lhs':
            doe = uqra.LHS(self.dist_xi)
            u   = doe.samples(size=n, random_state=random_state)
        else:
            raise NotImplementedError
        return u

    def update_num_samples(self, P, **kwargs):
        """
        return array number of samples based on given oversampling ratio alphas or num_samples
        alpha or num_samples = -1: 2 log(P) * P samples used as reference calculation
        """
        try:
            alphas = kwargs['alphas']
            self.alphas = np.array(alphas, dtype=np.float64).flatten()
            ### alpha = -1 for reference: 2 * P * log(P)
            if (self.alphas == -1).any():
                self.alphas[self.alphas==-1] = 2.0 * np.log(P)
            self.num_samples = np.array([math.ceil(P*ialpha) for ialpha in self.alphas])
            self.alphas = self.num_samples / P
        except KeyError:
            try:
                num_samples = kwargs['num_samples']
                self.num_samples = np.array(num_samples, dtype=np.int32).flatten()
                if (self.num_samples == -1).any():
                    self.num_samples[self.num_samples == -1] = int(math.ceil(2 * np.log(P) * P))
                self.alphas = self.num_samples /P
            except NameError:
                raise ValueError('Either alphas or num_samples should be defined')

    def get_basis(self, deg, **kwargs):
        if self.pce_type == 'legendre':
            basis = uqra.Legendre(d=self.ndim, deg=deg)
        elif self.pce_type == 'hermite_e':
            basis = uqra.Hermite(d=self.ndim, deg=deg, hem_type='probabilists')
        elif self.pce_type == 'hermite':
            basis = uqra.Hermite(d=self.ndim,deg=deg, hem_type='phy')
        elif self.pce_type == 'jacobi':
            a = kwargs['a']
            b = kwargs['b']
            basis = uqra.Jacobi(a, b, d=self.ndim, deg=deg)
        else:
            raise ValueError('UQRA.Parameters.get_basis error: undefined value {} for pce_type'.format(self.pce_type))
        return basis 

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
        self.out_responses   = kwargs.get('out_responses' , self.post_params[0]) 
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
        if nsim_completed >= self.n_budget:
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

    def _default_output_dir(self):
        """
        WORKING_DIR/
        +-- MODEL_DIR
        |   +-- FIGURE_DIR

        /directory saving data depends on OS/
        +-- MODEL_DIR
        |   +-- DATA_DIR

        """
        current_os  = sys.platform
        if current_os.upper()[:3] == 'WIN':
            data_dir        = os.path.join('G:\\','My Drive','MUSE_UQ_DATA', 'UQRA_Examples')
            data_dir_testin = os.path.join('G:\\','My Drive','MUSE_UQ_DATA', 'ExperimentalDesign', 'Random')
        elif current_os.upper() == 'DARWIN':
            data_dir        = r'/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/UQRA_Examples'
            data_dir_testin = r'/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/ExperimentalDesign/Random'
        elif current_os.upper() == 'LINUX':
            data_dir        = r'/home/jinsong/Documents/MUSE_UQ_DATA/UQRA_Examples'
            data_dir_testin = r'/home/jinsong/Documents/MUSE_UQ_DATA/ExperimentalDesign/Random'
        else:
            raise ValueError('Operating system {} not found'.format(current_os))    
        figure_dir      = os.path.join(data_dir, self.solver.nickname, 'Figures')
        data_dir_result = os.path.join(data_dir, self.solver.nickname, 'Data')
        data_dir_test   = os.path.join(data_dir, self.solver.nickname, 'TestData')
        # Create directory for model  
        try:
            os.makedirs(data_dir_result)
        except FileExistsError:
            pass
        try:
            os.makedirs(data_dir_test)
        except FileExistsError:
            pass
        try:
            os.makedirs(figure_dir)
        except FileExistsError:
            pass

        self.figure_dir     = figure_dir
        self.data_dir_test  = data_dir_test
        self.data_dir_result= data_dir_result
        self.data_dir_testin= data_dir_testin

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
        if self.doe_optimality:
            tag = '{:s}{:s}'.format(self.doe_candidate.capitalize(), self.doe_optimality.capitalize())
        else:
            tag = '{:s}'.format(self.doe_candidate.capitalize())
        return tag


    # def get_candidate_data(self, filename, sampling_domain=None, sampling_space='u', support=None):
        # """
        # Return canndidate samples 

        # Arguments:
            # filename: string of filename with candidate samples. data contains (u, x) values
            # sampling_domain: domain (in sampling_space) of samples
            # sampling_space:  'u' or 'x', defines
            # support: support of x_dist

        # """
        # data = np.load(os.path.join(self.data_dir_result, 'TestData', filename))
        # u_cand, x_cand = data[:self.solver.ndim], data[self.solver.ndim:2*self.solver.ndim]

    # def get_candidate_data(self, filename, sampling_domain=None, sampling_space='u', support=None):
        # """
        # Return canndidate samples 

        # Arguments:
            # filename: string of filename with candidate samples.
                # if cdf data is given, need to calculate corresponding u, x values
                # otherwise, filename should have (u_cdf, x) values
            # sampling_domain: domain (in sampling_space) of samples
            # sampling_space:  'u' or 'x', defines
            # support: support of x_dist

        # """
        # data = np.load(os.path.join(self.data_dir_result, 'TestData', filename))
        # u_cand, x_cand = data[:self.solver.ndim], data[self.solver.ndim:2*self.solver.ndim]
        # ## maping u->x, or x->u 
        # if sampling_space == 'u':
            # x_cand = uqra.inverse_rosenblatt(self.x_dist, u_cand, self.dist_xi, support=support)
        # elif sampling_space == 'x':
            # u_cand = uqra.rosenblatt(self.x_dist, x_cand, self.dist_xi, support=support)

        # ux_isnan = np.zeros(u_cand.shape[1])
        # for ix, iu in zip(x_cand, u_cand):
            # ux_isnan = np.logical_or(ux_isnan, np.isnan(ix))
            # ux_isnan = np.logical_or(ux_isnan, np.isnan(iu))
        # x_cand = x_cand[:, np.logical_not(ux_isnan)]
        # u_cand = u_cand[:, np.logical_not(ux_isnan)]
        # x_cand = x_cand[:, :self.n_cand]
        # u_cand = u_cand[:, :self.n_cand]

        # if sampling_domain is None:
            # u = u_cand
            # x = x_cand
        # else:
            # if sampling_space == 'u':
                # assert self.check_samples_inside_domain(u_cand, sampling_domain)
            # elif sampling_space == 'x':
                # assert self.check_samples_inside_domain(x_cand, sampling_domain)
            # else:
                # raise ValueError("Undefined value {} for UQRA.Parameters.get_test_data.sampling_space".format(sampling_space))
            # u = u_cand
            # x = x_cand
        # return u, x 

    # def check_samples_inside_domain(self, data, domain):
        # data = np.array(data, ndmin=2, copy=False)
        # if self.xi_distname == 'norm':
            # if np.ndim(domain) == 0:
                # r1, r2 = 0, domain
            # else:
                # r1, r2 = domain
            # radius = np.linalg.norm(data, axis=0)
            # res = r1 <= min(radius) and r2>= max(radius)

        # elif self.xi_distname == 'uniform':
            # if data.shape[0] != len(domain):
                # raise ValueError('Expecting {:d} intervals but only {:d} given'.format(data.shape[0], len(domain)))
            # min_, max_ = np.amin(data, axis=1), np.amax(data, axis=1)
            # res = True
            # for imin_, imax_, idomain in zip(min_, max_, domain):
                # if idomain is None:
                    # res = res and True
                # else:
                    # res = res and (idomain[0] <= imin_) and (imax_ <= idomain[1])
        # elif self.xi_distname == 'beta':
            # raise NotImplementedError
        # else:
            # raise ValueError('UQRA.Parameters.xi_distname {:s} not defined'.format(self.xi_distname)) 
        # return res

    # def separate_samples_by_domain(self, data, domain):
        # """
        # Return the index for data within the defined domain
        # """

        # data = np.array(data, ndmin=2, copy=False)
        # if self.xi_distname == 'norm':
            # if np.ndim(domain) == 0:
                # r1, r2 = 0, domain
            # else:
                # r1, r2 = domain
            # idx_inside, idx_outside = uqra.samples_within_circle(data, r1, r2) 

        # elif self.xi_distname == 'uniform':
            # if data.shape[0] != len(domain):
                # raise ValueError('Expecting {:d} intervals but only {:d} given'.format(data.shape[0], len(domain)))
            # idx_inside, idx_outside = uqra.samples_within_cubic(data, domain) 

        # elif self.xi_distname == 'beta':
            # raise NotImplementedError
        # else:
            # raise ValueError('UQRA.Parameters.xi_distname {:s} not defined'.format(self.xi_distname)) 

        # return idx_inside, idx_outside

    # def get_predict_data(self, filename, sampling_domain=None, sampling_space='u', support=None):
        # """
        # Return predict samples in x space
        # sampling_domain: sampling sampling_domain, used to draw samples
        # """
        # print(' > Loading predict data: {:s}'.format(filename))
        # if filename.lower().startswith('cdf'):
            # u_cdf = np.load(os.path.join(self.data_dir_cand, 'CDF', filename))
            # u_cdf_pred = u_cdf[:self.solver.ndim, :self.n_pred]
            # u = np.array([idist.ppf(iu_cdf) for idist, iu_cdf in zip(self.dist_xi, u_cdf_pred)])
            # x = uqra.inverse_rosenblatt(self.x_dist, u, self.dist_xi, support=support)
            # u = np.array(u, ndmin=2, copy=False)
            # x = np.array(x, ndmin=2, copy=False)
        # else:
            # data = np.load(filename)
            # assert data.shape[0] == 2 * self.solver.ndim
            # u = data[:self.solver.ndim]
            # x = data[self.solver.ndim:]
            # u = np.array(u, ndmin=2, copy=False)
            # x = np.array(x, ndmin=2, copy=False)

        # if sampling_domain is None:
            # ux_isnan = np.zeros(u.shape[1])
            # for iu, ix in zip(u, x):
                # ux_isnan = np.logical_or(np.isnan(iu))
                # ux_isnan = np.logical_or(np.isnan(ix))
            # if np.sum(ux_isnan):
                # print(' nan values found in predict U/X, dropping {:d} NAN'.format(np.sum(ux_isnan)))
            # u0 = u[:,np.logical_not(ux_isnan)]
            # x0 = x[:,np.logical_not(ux_isnan)]
            # u1 = u[:,ux_isnan]
            # x1 = x[:,ux_isnan]
        # else:
            # ux_isnan = np.zeros(u.shape[1])
            # for iu, ix in zip(u, x):
                # ux_isnan = np.logical_or(np.isnan(iu))
                # ux_isnan = np.logical_or(np.isnan(ix))
            # if sampling_space == 'u':
                # idx_inside, idx_outside = self.separate_samples_by_domain(u, sampling_domain)
                # u0 = u[:, idx_inside]
                # x0 = x[:, idx_inside]
                # # x0 = uqra.inverse_rosenblatt(self.x_dist, u0, self.dist_xi, support=support)
                # # if np.amax(abs(x0-x[:, idx_inside]), axis=None) > 1e-6:
                    # # print(np.amax(abs(x0-x[:, idx_inside]), axis=None))
                # u1 = u[:, np.logical_or(idx_outside, ux_isnan)] 
                # x1 = x[:, np.logical_or(idx_outside, ux_isnan)]

            # elif sampling_space == 'x':
                # idx_inside, idx_outside = self.separate_samples_by_domain(x, sampling_domain)
                # x0 = x[:, idx_inside]
                # u0 = u[:, idx_inside]
                # # u0 = uqra.rosenblatt(self.x_dist, x0, self.dist_xi, support=support)
                # # if np.amax(abs(u0-u[:, idx_inside]), axis=None) > 1e-6:
                    # # print(np.amax(abs(u0-u[:, idx_inside]), axis=None))

                # u1 = u[:, np.logical_or(idx_outside, ux_isnan)] 
                # x1 = x[:, np.logical_or(idx_outside, ux_isnan)]
                # # u1 = u[:, idx_outside] 
                # # x1 = x[:, idx_outside]
        # return u0, x0, u1, x1

    # def get_test_data(self, filename, sampling_domain=None, sampling_space='u', support=None):
        # data    = np.load(os.path.join(self.data_dir_result, 'TestData', filename))
        # u       = data[                   :    self.solver.ndim, :self.n_test]
        # x       = data[  self.solver.ndim : 2* self.solver.ndim, :self.n_test]
        # y       = np.squeeze(data[2*self.solver.ndim :        , :self.n_test])
        # u       = uqra.rosenblatt(self.x_dist, x, self.dist_xi, support=support)


        # if sampling_domain is None:
            # return u, x, y
        # else:
            # if sampling_space == 'u':
                # idx_inside, idx_outside = self.separate_samples_by_domain(u, sampling_domain)
                # u = u[:, idx_inside]
                # x = x[:, idx_inside]
                # y = np.squeeze(np.array(y, ndmin=2,copy=False)[:, idx_inside]) 
            # elif sampling_space == 'x':
                # idx_inside, idx_outside = self.separate_samples_by_domain(x, sampling_domain)
                # u = u[:, idx_inside]
                # x = x[:, idx_inside]
                # y = np.squeeze(np.array(y, ndmin=2,copy=False)[:, idx_inside]) 
            # else:
                # raise ValueError("Undefined value {} for UQRA.Parameters.get_test_data.sampling_space".format(sampling_space))

            # return u, x, y
        
        # doe_candidate = self.doe_candidate.lower()
        # data_dir = os.path.join(self.data_dir_cand, doe_candidate.upper(), self.xi_distname.capitalize()) 
        # try:
            # self.filename_candidates = kwargs['filename']
            # try:
                # data = np.load(self.filename_candidates)
            # except FileNotFoundError:
                # data = np.load(os.path.join(data_dir, self.filename_candidates))
            # u_cand = data[:self.ndim,:n].reshape(self.ndim, -1) ## will raise error when samples files smaller than n
            # self.filename_optimality = kwargs.get('filename_optimality', None)

        # except KeyError:
            # if doe_candidate.lower().startswith('mcs'):
                # self.filename_candidates = r'DoE_McsE6R0.npy'
                # data = np.load(os.path.join(data_dir, self.filename_candidates))
                # u_cand = data[:self.ndim,:n].reshape(self.ndim, -1) ## will raise error when samples files smaller than n

            # elif doe_candidate.lower().startswith('cls') or doe_candidate == 'reference':
                # if self.xi_distname.lower().startswith('norm'):
                    # self.filename_candidates = r'DoE_ClsE6d{:d}R0.npy'.format(self.ndim)
                # elif self.xi_distname.lower().startswith('uniform'):
                    # self.filename_candidates = r'DoE_ClsE6R0.npy'
                # else:
                    # raise ValueError('dist_x_name {} not defined'.format(self.dist_x_name))
                # data  = np.load(os.path.join(data_dir, self.filename_candidates))
                # u_cand = data[:self.ndim,:n].reshape(self.ndim, -1)
            # else:
                # raise ValueError('DoE method {:s} not defined'.format(doe_candidate))
            # self.filename_optimality = kwargs.get('filename_optimality', None)

        # return u_cand

    # def get_test_data(self, solver, pce_model, filename = r'DoE_McsE6R9.npy', **kwargs):
        # """
        # Return test data. 

        # Test data should always be in X-space. The correct sequence is X->y->u

        # To be able to generate MCS samples for X, we use MCS samples in Samples/MCS, noted as z here

        # If already exist in simparams.data_dir_result, then load and return
        # else, run solver

        # """
        
        # data_dir_result = os.path.join(self.params.data_dir_result, 'TestData')
        # try: 
            # os.makedirs(data_dir_result)
        # except OSError as e:
            # pass

        # n       = kwargs.get('n'   , self.params.n_test)
        # qoi     = kwargs.get('qoi' , solver.out_responses )
        # n       = int(n)
        # n_short_term = self.solver.n_short_term
        # assert solver.ndim == pce_model.ndim
        # ndim = solver.ndim
        # try:
            # nparams = solver.nparams
        # except AttributeError:
            # nparams = ndim
        # self.filename_test = '{:s}_{:d}{:s}_'.format(solver.nickname, ndim, pce_model.basis.nickname) + filename
        # if self.params.doe_candidate.lower() == 'cls':
            # self.filename_test = self.filename_test.replace('Mcs', 'Cls')

        # try:
            # u_test = None 
            # x_test = None
            # y_test = []
            # for iqoi in qoi:
                # filename_iqoi = self.filename_test[:-4] + '_y{:d}.npy'.format(iqoi)
                # data_set = np.load(os.path.join(data_dir_result, filename_iqoi))
                # print('   - Retrieving test data from {}'.format(os.path.join(data_dir_result, filename_iqoi)))
                # if not solver.nickname.lower().startswith('sdof'):
                    # assert data_set.shape[0] == 2*ndim+1

                # u_test_ = data_set[     :  ndim,:n] if n > 0 else data_set[     :  ndim, :]
                # x_test_ = data_set[ndim :ndim+nparams,:n] if n > 0 else data_set[ndim :ndim+nparams, :]

                # if u_test is None:
                    # u_test  = u_test_
                # else:
                    # assert np.array_equal(u_test_, u_test)

                # if x_test is None:
                    # x_test  = x_test_
                # else:
                    # assert np.array_equal(x_test_, x_test)

                # y_test_ = data_set[ndim+nparams: ndim+nparams+n_short_term,:n] if n > 0 else data_set[ndim+nparams: ndim+nparams+n_short_term, : ]
                # y_test.append(y_test_)
            # if len(y_test) == 1:
                # y_test = y_test[0]

        # except FileNotFoundError:
            # ### 1. Get MCS samples for X
            # if pce_model.basis.xi_distname.lower() == 'uniform':
                # data_dir_cand = os.path.join(self.params.data_dir_cand, 'MCS','Uniform')
                # print('    - Solving test data from {} '.format(os.path.join(data_dir_cand,filename)))
                # data_set = np.load(os.path.join(data_dir_cand,filename))
                # z_test = data_set[:ndim,:n] if n > 0 else data_set[:ndim,:]
                # x_test = solver.map_domain(z_test, [stats.uniform(-1,2),] * ndim)
            # elif pce_model.basis.xi_distname.lower().startswith('norm'):
                # data_dir_cand = os.path.join(self.params.data_dir_cand, 'MCS','Norm')
                # print('    - Solving test data from {} '.format(os.path.join(data_dir_cand,filename)))
                # data_set= np.load(os.path.join(data_dir_cand,filename))
                # z_test  = data_set[:ndim,:n] if n > 0 else data_set[:ndim,:]
                # x_test  = solver.map_domain(z_test, [stats.norm(0,1),] * ndim)
            # else:
                # raise ValueError
            # y_test = solver.run(x_test, out_responses='ALL', save_qoi=True, data_dir=data_dir_result)
            # np.save('y_test.npy', np.array(y_test))

            # ### 2. Mapping MCS samples from X to u
            # ###     dist_u is defined by pce_model
            # ### Bounded domain maps to [-1,1] for both mcs and cls methods. so u = z
            # ### Unbounded domain, mcs maps to N(0,1), cls maps to N(0,sqrt(0.5))
            # u_test = 0.0 + z_test * np.sqrt(0.5) if self.is_cls_unbounded() else z_test
            # u_test = u_test[:,:n] if n > 0 else u_test
            # x_test = x_test[:,:n] if n > 0 else x_test

            # if y_test.ndim == 1:
                # data   = np.vstack((u_test, x_test, y_test.reshape(1,-1)))
                # np.save(os.path.join(data_dir_result, self.filename_test), data)
                # print('   > Saving test data to {} '.format(data_dir_result))
                # y_test = y_test[  :n] if n > 0 else y_test
            # elif y_test.ndim == 2:
                # # if y_test.shape[0] == n:
                    # # y_test = y_test.T
                # # elif y_test.shape[1] == n:
                    # # y_test = y_test
                # # else:
                # raise ValueError('Solver output format not understood: {}, expecting has {:d} in 1 dimensino'.format(n))
            # elif y_test.ndim == 3:
                # ### (n_short_term, n, nqoi)
                # if solver.name.lower() == 'linear oscillator':
                    # y = []
                    # for i, iqoi_test in enumerate(y_test.T):
                        # data = np.vstack((u_test, x_test, iqoi_test.T))
                        # np.save(os.path.join(data_dir_result, self.filename_test[:-4]+'_y{:d}.npy'.format(i)), data)
                        # if i in solver.out_responses:
                            # y.append(iqoi_test[:n].T if n > 0 else iqoi_test)
                    # print('   > Saving test data to {} '.format(data_dir_result))
                    # y_test = y[0] if len(y) == 1 else y
                # elif solver.name.lower() == 'duffing oscillator':
                    # y = []
                    # for i, iqoi_test in enumerate(y_test.T):
                        # data = np.vstack((u_test, x_test, iqoi_test.T))
                        # np.save(os.path.join(data_dir_result, self.filename_test[:-4]+'_y{:d}.npy'.format(i)), data)
                        # if i in solver.out_responses:
                            # y.append(iqoi_test[:n].T if n > 0 else iqoi_test)
                    # print('   > Saving test data to {} '.format(data_dir_result))
                    # y_test = y[0] if len(y) == 1 else y
                # else:
                    # raise NotImplementedError
        # return u_test, x_test, y_test


        # # self.solver = solver
        # self.doe_optimality = None
        # ###------------- Adaptive setting -----------------------------
        # self.is_adaptive= False
        # self.plim     = kwargs.get('plim'     , None  )  ## polynomial degree limit
        # self.n_budget = kwargs.get('n_budget' , None  )
        # self.min_r2   = kwargs.get('min_r2'   , None  )  ## minimum adjusted R-squared threshold value to take
        # self.rel_mse  = kwargs.get('rel_mse'  , None  )  ## Relative mean square error 
        # self.abs_mse  = kwargs.get('abs_mse'  , None  )  ## Absolute mean square error
        # self.rel_qoi  = kwargs.get('rel_qoi'  , None  )  ## Relative error for QoI, i.e. percentage difference relative to previous simulation
        # self.abs_qoi  = kwargs.get('abs_qoi'  , None  )  ## Absolute error for QoI, i.e. decimal accuracy 
        # self.qoi_val  = kwargs.get('qoi_val'  , None  )  ## QoI value up to decimal accuracy 
        # self.rel_cv   = kwargs.get('rel_cv'   , 0.05  )  ## percentage difference relative to previous simulation
        # if self.plim is not None or self.n_budget is not None:
            # self.is_adaptive = True
        
        # print(r' > Optional parameters:')
        # if self.time_params:
            # print(r'   * {:<15s} : '.format('time parameters'))
            # print(r'     - {:<8s} : {:.2f} - {:<8s} : {:.2f}'.format('start', self.time_start, 'end', self.time_max ))
            # print(r'     - {:<8s} : {:.2f} - {:<8s} : {:.2f}'.format('ramp ', self.time_ramp , 'dt ', self.dt ))
        # if self.post_params:
            # print(r'   * {:<15s} '.format('post analysis parameters'))
            # out_responses = self.out_responses if self.out_responses is not None else 'All'
            # print(r'     - {:<23s} : {} '.format('out_responses', out_responses))
            # print(r'     - {:<23s} : {} '.format('statistics'  , self.stats)) 

        # if self.is_adaptive:
            # print(r'   * {:<15s} '.format('Adaptive parameters'))
            # print(r'     - {:<23s} : {} '.format('Simulation budget', self.n_budget))
            # print(r'     - {:<23s} : {} '.format('Poly degree limit', self.plim))
            # print(r'     - {:<23s} : {} '.format('Relvative CV error', self.rel_cv))
            # if self.min_r2:
                # print(r'     - {:<23s} : {} '.format('R2 bound', self.min_r2))
            # if self.rel_mse:
                # print(r'     - {:<23s} : {} '.format('Relative MSE', self.rel_mse))
            # if self.abs_mse:
                # print(r'     - {:<23s} : {} '.format('Absolute MSE', self.abs_mse))
            # if self.rel_qoi:
                # print(r'     - {:<23s} : {} '.format('Relative QoI', self.rel_qoi))
            # if self.abs_qoi:
                # print(r'     - {:<23s} : {} '.format('QoI decimal accuracy', self.abs_qoi))
            # if self.qoi_val:
                # print(r'     - {:<23s} : {} '.format('QoI=0, decimal accuracy', self.qoi_val))

    # def test_data_reference(self):
        # """
        # Validate the distributions of test data
        # """
        # if self.xi_distname.lower() == 'uniform':
            # u_mean = 0.0
            # u_std  = 0.5773
        # elif self.xi_distname.lower().startswith('norm'):
            # if self.params.doe_candidate.lower() == 'cls':
                # u_mean = 0.0
                # u_std  = np.sqrt(0.5)
            # elif self.params.doe_candidate.lower() == 'mcs':
                # u_mean = 0.0
                # u_std  = 1.0
            # else:
                # raise ValueError
        # else:
            # raise ValueError

        # return u_mean, u_std
            
    # def candidate_data_reference(self):
        # """
        # Validate the distributions of test data
        # """
        # if self.params.doe_candidate.lower() == 'mcs':
            # if self.xi_distname.lower() == 'uniform':
                # u_mean = 0.0
                # u_std  = 0.58
            # elif self.xi_distname.lower().startswith('norm'):
                # u_mean = 0.0
                # u_std  = 1.0
            # else:
                # raise ValueError
        # elif self.params.doe_candidate.lower() == 'cls':
            # if self.xi_distname.lower() == 'uniform':
                # u_mean = 0.0
                # u_std  = 0.71
            # elif self.xi_distname.lower().startswith('norm'):
                # u_mean = 0.0
                # if self.ndim == 1:
                    # u_std = 0.71
                # elif self.ndim == 2:
                    # u_std = 0.57
                # elif self.ndim == 3:
                    # u_std = 0.50
                # elif self.ndim == 4:
                    # u_std = 0.447 
                # else:
                    # raise NotImplementedError 
            # else:
                # raise ValueError
        # else:
            # raise ValueError

        # return u_mean, u_std

    # def sampling_density(self, u, p):
        # if self.params.doe_candidate.lower().startswith('mcs'):
            # if self.xi_distname.lower().startswith('norm'):
                # pdf = np.prod(stats.norm(0,1).pdf(u), axis=0)
            # elif self.xi_distname.lower().startswith('uniform'):
                # pdf = np.prod(stats.uniform(-1,2).pdf(u), axis=0)
            # else:
                # raise ValueError('{:s} not defined for MCS'.format(self.xi_distname))

        # elif self.params.doe_candidate.lower().startswith('cls'):
            # if self.xi_distname.lower().startswith('norm'):
                # pdf = 1.0/(self.ndim*np.pi * np.sqrt(p))*(2 - np.linalg.norm(u/np.sqrt(p),2, axis=0)**2)**(self.ndim/2.0) 
                # pdf[pdf<0] = 0
            # elif self.xi_distname.lower().startswith('uniform'):
                # pdf = 1.0/np.prod(np.sqrt(1-u**2), axis=0)/np.pi**self.ndim
            # else:
                # raise ValueError('{:s} not defined for CLS'.format(self.xi_distname))
        # else:
            # raise ValueError('{:s} not defined '.format(self.params.doe_candidate))
        # return pdf


    # def is_cls_unbounded(self):
        # return  self.params.doe_candidate.lower().startswith('cls') and self.xi_distname.lower().startswith('norm')
    # def _choose_samples_from_candidates(self, n, u_cand, u_selected=None, active_basis=None, precomputed=False, precomputed_index=None):
        # """
        # Return train data from candidate data set. All samples are in U-space (with pluripotential equilibrium measure nv(x))

        # Arguments:
            # n           : int, size of new samples in addtion to selected elements
            # u_cand      : ndarray, candidate samples in U-space to be chosen from
            # u_selected  : samples already are selected, need to be removed from candidate set to get n samples
            # basis       : When doe_optimality is 'D' or 'S', one need the design matrix in the basis selected
            # active_basis: activated basis used in doe_optimality design

        # """
        # ### get the list of indices in u_selected
        # selected_index = list(self._common_vectors(u_selected, u_cand))

        # if self.params.doe_optimality is None:
            # ### for non doe_optimality design, design matrix X is irrelative, so all columns are used
            # row_index_adding = []
            # while len(row_index_adding) < n:
                # ### random index set
                # random_idx = set(np.random.randint(0, u_cand.shape[1], size=n*10))
                # ### remove selected_index chosen in this step
                # random_idx = random_idx.difference(set(row_index_adding))
                # ### remove selected_index passed
                # random_idx = random_idx.difference(set(selected_index))
                # ### update new samples set
                # row_index_adding += list(random_idx)
            # row_index_adding = row_index_adding[:n]
            # u_new = u_cand[:,row_index_adding]
            # u_all = u_new if u_selected is None else np.hstack((u_selected, u_new))
            # duplicated_idx_in_all = self._get_duplicate_rows(u_all.T)
            # if len(duplicated_idx_in_all) > 0:
                # raise ValueError('Array have duplicate vectors: {}'.format(duplicated_idx_in_all))

        # elif self.params.doe_optimality:
            # doe = uqra.OptimalDesign(self.params.doe_optimality, selected_index=selected_index)
            # ### Using full design matrix, and precomputed doe_optimality file exists only for this calculation
            # if precomputed:
                # row_index_adding = []
                # try:
                    # for i in precomputed_index:
                        # if len(row_index_adding) >= n:
                            # break
                        # if i in selected_index:
                            # pass
                        # else:
                            # row_index_adding.append(i)
                # except AttributeError:
                    # raise AttributeError('Precomputed is True but precomputed_index was not found')
                # u_new = u_cand[:,row_index_adding]
                # u_all = u_new if u_selected is None else np.hstack((u_selected, u_new))
                # duplicated_idx_in_all = self._get_duplicate_rows(u_all.T)
                # if len(duplicated_idx_in_all) > 0:
                    # raise ValueError('Array have duplicate vectors: {}'.format(duplicated_idx_in_all))
            # else:
                # active_index = [i for i in range(basis.num_basis) if basis.basis_degree[i] in active_basis]
                # assert len(active_index) != 0
                # X = basis.vandermonde(u_cand)
                # X = X[:,active_index]
                # if self.params.doe_candidate.lower().startswith('cls'):
                    # X  = X.shape[1]**0.5*(X.T / np.linalg.norm(X, axis=1)).T
                # row_index_adding = doe.get_samples(X, n, orth_basis=True)
                # u_new = u_cand[:,row_index_adding]
                # u_all = u_new if u_selected is None else np.hstack((u_selected, u_new))
                # duplicated_idx_in_all = self._get_duplicate_rows(u_all.T)
                # if len(duplicated_idx_in_all) > 0:
                    # raise ValueError('Array have duplicate vectors: {}'.format(duplicated_idx_in_all))
            # # if active_basis == 0 or active_basis is None or len(active_basis) == 0:
                # # active_index = np.arange(basis.num_basis).tolist()
                # # if self._check_precomputed_optimality(basis) and precomputed:
                    # # row_index_adding = []
                    # # for i in self.precomputed_optimality_index:
                        # # if len(row_index_adding) >= n:
                            # # break
                        # # if i in selected_index:
                            # # pass
                        # # else:
                            # # row_index_adding.append(i)
                    # # u_new = u_cand[:,row_index_adding]
                    # # u_all = u_new if u_selected is None else np.hstack((u_selected, u_new))
                    # # duplicated_idx_in_all = self._get_duplicate_rows(u_all.T)
                    # # if len(duplicated_idx_in_all) > 0:
                        # # raise ValueError('Array have duplicate vectors: {}'.format(duplicated_idx_in_all))
                # # else:
                    # # X = basis.vandermonde(u_cand)
                    # # if self.params.doe_candidate.lower().startswith('cls'):
                        # # X  = X.shape[1]**0.5*(X.T / np.linalg.norm(X, axis=1)).T
                    # # row_index_adding = doe.get_samples(X, n, orth_basis=True)
                    # # u_new = u_cand[:,row_index_adding]
                    # # u_all = u_new if u_selected is None else np.hstack((u_selected, u_new))
                    # # duplicated_idx_in_all = self._get_duplicate_rows(u_all.T)
                    # # if len(duplicated_idx_in_all) > 0:
                        # # raise ValueError('Array have duplicate vectors: {}'.format(duplicated_idx_in_all))

        # return u_new, u_all

    # def _check_precomputed_optimality(self, active_basis):
        # """

        # """
        # ### Case: direct MCS and CLS without doe_optimality, basis could be None
        # if self.params.doe_optimality is None:
            # return False, [None,]

        # ### Case: Optimal design with only sigificant basis 
        # if len(active_basis) < basis.num_basis:
            # return False, [None,]

        # ### Case: Optimal design with all basis 
        # try:
            # ### If filename_optimality was given in get_candidate_data
            # precomputed_optimality_index = np.load(self.filename_optimality)
        # except (AttributeError, FileNotFoundError, TypeError) as e:
            # self.filename_optimality = 'DoE_{:s}E{:s}R0_{:d}{:s}{:d}_{:s}.npy'.format(self.params.doe_candidate.capitalize(),
                    # '{:.0E}'.format(self.params.n_cand)[-1], self.ndim, self.model.basis.nickname, basis.deg, self.params.doe_optimality)
            # try:
                # precomputed_optimality_index = np.squeeze(np.load(os.path.join(self.params.data_dir_precomputed_optimality, self.filename_optimality)))
                # precomputed_optimality_index = np.array(precomputed_optimality_index, ndmin=2)
                # ### For some S-Optimality designs, there exist more than one sets
                # ### wip: need to complete this feature. At this moment, if more than one exist, use the first one
                # # if self.precomputed_optimality_index.ndim == 2:
                    # # self.precomputed_optimality_index = self.precomputed_optimality_index
                # return True, precomputed_optimality_index
            # except FileNotFoundError: 
                # print('FileNotFound: No such file or directory: {}'.format(self.filename_optimality))
                # return False, [None,]
        # except:
            # return False, [None,]


    # def get_train_data(self, size, u_cand, u_train=None, active_basis=None, precomputed=True):
        # """
        # Return train data from candidate data set. All samples are in U-space
        # size is how many MORE to be sampled besides those alrady existed in u_train 

        # Arguments:
            # size        : size of samples, (r, n): size n repeats r times
            # u_cand      : ndarray, candidate samples in U-space to be chosen from
            # u_train     : samples already selected, need to be removed from candidate set to get n samples
            # active_basis: activated basis used in doe_optimality design

        # """
        # ###  formating the input arguments
        # u_cand  = np.array(u_cand, ndmin=2, copy=False)
        # size    = tuple(np.atleast_1d(size))
        # if len(size) == 1:
            # ## case: just a number if given n
            # repeats, n = 1, int(size[0])
        # elif len(size) == 2:
            # ## case: (1,n) or (r, n) 
            # repeats, n = int(size[0]), int(size[1])
        # else:
            # raise ValueError

        # ## u_train format checking
        # if u_train is None:
            # u_train = [None,] * repeats
        # else:
            # u_train = np.array(u_train, ndmin=2)
            # if u_train.ndim == 2:
                # ## just one train data set is given, of shape(ndim, nsamples)
                # assert u_train.shape[0] == self.ndim
                # u_train = [u_train,] * repeats
            # elif u_train.ndim == 3:
                # ## #repeats train data set are given, of shape(repeats, ndim, nsamples)
                # assert u_train.shape[0] == repeats
                # assert u_train.shape[1] == self.ndim
            # else:
                # ValueError('Wrong data type for u_train: {}'.format(u_train.shape))
        # assert len(u_train) == repeats, 'Expecting {:d} trianing set, but {:d} were given, u_train.shape={}'.format(repeats, len(u_train), u_train.shape )

        # u_train_new = []
        # u_train_all = []
        # active_basis= self.model.basis.basis_degree if active_basis is None or len(active_basis) == 0 else active_basis

        # ### Checking if the data is available to speed up the process
        # ### Precomputed datasets are only available to Optimality Design (S/D) with ALL basis 
        # if precomputed:
            # precomputed, precomputed_index = self._check_precomputed_optimality(active_basis) 

        # if precomputed:
            # tqdm.write('   - {:<20s} : {:s}'.format('Precomputed File ', self.filename_optimality))
        # else:
            # precomputed_index = [None,] * repeats

        # for r in tqdm(range(repeats), ascii=True,ncols=80, desc='   '):
            # u_new, u_all  = self._choose_samples_from_candidates(n, u_cand, 
                    # u_selected=u_train[r], active_basis=active_basis, precomputed=precomputed, precomputed_index=precomputed_index[r])
            # u_train_new.append(u_new)
            # u_train_all.append(u_all)
        # if len(size) == 1:
            # u_train_new = u_train_new[0] 
            # u_train_all = u_train_all[0]
        # elif len(size) == 2:
            # u_train_new = np.array(u_train_new)
            # u_train_all = np.array(u_train_all)

        # return u_train_new, u_train_all

    # def cal_adaptive_bias_weight(self, u, p, sampling_pdf):
        # sampling_pdf_p = self.sampling_density(u, p)
        # w = sampling_pdf_p/sampling_pdf
        # return w

    # def __init__(self, solver, doe_method=['MCS', None], fit_method='OLS'):
        # sys.stdout      = Logger()
        # self.solver     = solver
        # self.ndim       = self.solver.ndim
        # doe_method = [doe_method,] if not isinstance(doe_method, (list, tuple)) else doe_method
        # if len(doe_method) > 2 or len(doe_method) < 1:
            # raise ValueError('Initialize UQRA.Parameters error: given doe_method has {:d} elements'.format(len(doe_method)))
        # elif len(doe_method) == 1:
            # self.doe_candidate  = str(doe_method[0]).lower()
            # self.doe_optimality = None 
        # else:
            # self.doe_candidate  = str(doe_method[0]).lower()
            # self.doe_optimality = doe_method[1]

        # self.fit_method = str(fit_method).lower()
        # self.tag        = self._get_tag()
        # self.update_output_dir()
        # self.data_dir_precomputed_optimality = os.path.join(self.data_dir_cand, 'OED')
        # elif doe_candidate.lower() == 'mcs':


        # np.random.seed(random_state)
        # if self.doe_optimality is None:
            # u_cand = kwargs.get('u_cand', None)
            # if u_cand is None:
                # u_cdf = stats.uniform(0,1).rvs(size=(self.solver.ndim, n))
                # u   = np.array([ixi_dist.ppf(iu) for ixi_dist, iu in zip(self.dist_xi, u)]) 
            # else:
                # u = u_cand[:, np.random.randint(0, u_cand.shape[1], size=n)]
            # return u

        # else:
            # u_cand      = kwargs['u_cand']
            # pce_order   = kwargs['p']
            # ndim, n_cand= u_cand.shape

            # if self.pce_type.lower() == 'hermite_e':
                # orth_poly = uqra.Hermite(d=ndim,deg=pce_order, hem_type='prob')
            # elif self.pce_type.lower() == 'hemite':
                # orth_poly = uqra.Hermite(d=ndim,deg=pce_order, hem_type='phy')
            # elif self.pce_type.lower() == 'legendre':
                # orth_poly = uqra.Legendre(d=ndim,deg=pce_order)
            # elif self.pce_type.lower() == 'jacobi':
                # raise NotImplementedError
            # else:
                # raise NotImplementedError
            # design_matrix = orth_poly.vandermonde(u_cand)
            # doe = uqra.OptimalDesign(self.doe_optimality.upper(), selected_index=[])
            # doe_index= doe.get_samples(design_matrix, n, orth_basis=True)
            # u = u_cand[:, doe_index]
            # return u

        # if doe_candidate.lower() == 'cls':
            # raise NotImplementedError
