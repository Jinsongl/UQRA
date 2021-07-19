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
            print(" [WARNING]: Optimality {:s} not applicable for LHS, 'optimaltiy' is set to None")
            self.optimality = None
        self._default_data_dir()

    def update_poly_name(self, poly_name):
        self.poly_name = poly_name
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
        Return DoE nickname for one specific doe_sampling and doe_optimality set, e.g. MCS-D
        """
        doe_sampling   = self.doe_sampling
        doe_optimality = self.optimality
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

    def sampling_weight(self, w=None):
        """
        Return a weight function corresponding to the sampling scheme
        If w is given, the given weight has the highest priority. 
        otherwise, return based on doe_sampling
        """
        if w is not None:
            return w
        elif self.doe_sampling.lower().startswith('cls'):
            return 'christoffel'
        elif self.doe_sampling.lower()[:3] in ['mcs', 'lhs']:
            return None
        else:
            raise ValueError('Sampling weight function is not defined')

    def get_samples(self, x, poly, n, x0=[], active_index=None,  
            initialization='RRQR', return_index=False, decimals=8):
        """
        return samples based on UQRA

        x   : ndarray of shape (d,N), candidate samples
        poly: UQRA.polynomial object
        n   : int, number of samples to be added
        x0  : samples already selected (will be ignored when performing optimization)
            1. list of selected index
            2. selected samples
        initialization: methods to generate the initial samples
            1. string, 'RRQR', 'TSM'
            2. list of index

        active_index: list of active basis in poly 
        decimals: accuracy tolerance when comparing samples in x0 to x
        """
        ### check arguments
        n = uqra.check_int(n)
        if self.doe_sampling.lower().startswith('lhs'):
            assert self.optimality is None
            dist_xi = poly.weight
            doe = uqra.LHS([dist_xi, ] *self.ndim)
            x_optimal = doe.samples(size=n)
            res = x_optimal
            res = (res, None) if return_index else res 

        else:
            x = np.array(x, copy=False, ndmin=2)
            d, N = x.shape
            assert d == self.ndim 
            ## expand samples if it is unbounded cls
            # x = poly.deg**0.5 * x if self.doe_sampling in ['CLS4', 'CLS5'] else x

            ## define selected index set
            if len(x0) == 0:
                idx_selected = []
            elif isinstance(x0, (list, tuple)):
                idx_selected = list(x0)
            else:
                x0 = np.array(x0, copy=False, ndmin=2).round(decimals=decimals)
                x  = x.round(decimals)
                assert x.shape[0] == x0.shape[0]
                idx_selected = list(uqra.common_vectors(x0, x))

            ## define methods to get initial samples 
            initialization = initialization if len(idx_selected) == 0 else idx_selected

            x_optimal = []
            if self.doe_sampling.lower().startswith('mcs'):
                if str(self.optimality).lower() == 'none':
                    idx = list(set(np.arange(self.num_cand)).difference(set(idx_selected)))[:n] 
                else:
                    X = poly.vandermonde(x)
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
                    X = poly.vandermonde(x)
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

    def domain_of_interest(self, y0, data_xi, data_y, n_centroid=10, epsilon=0.1, random_state=None):
        ndim, deg = self.ndim, self.deg
        ## otbain the centroids of DoIs
        centroid_xi = np.array([data_xi[:, i] for i in np.argsort(abs(data_y-y0))[:n_centroid]]).T
        nsamples_each_centroid = np.zeros(n_centroid)
        DoI_cand_xi = [] 
        while True:
            np.random.seed(random_state)
            if self.doe_sampling.upper() in ['MCS', 'LHS']:
                xi_min = np.amin(centroid_xi, axis=1) - epsilon
                xi_max = np.amax(centroid_xi, axis=1) + epsilon
                assert len(xi_min) == ndim
                ## sampling from truncated dist_xi distribution with boundary [a,b]
                xi_cand = []
                for a, b in zip(xi_min, xi_max):
                    cdf_a = self.dist_xi.cdf(a)
                    cdf_b = self.dist_xi.cdf(b)
                    u = stats.uniform(cdf_a,cdf_b-cdf_a).rvs(100000)
                    xi_cand.append(self.dist_xi.ppf(u))
                xi_cand = np.array(xi_cand)


                # DoE = uqra.MCS([self.dist_xi, ] * ndim)
                # xi_cand = DoE.samples(10000000)
            elif self.doe_sampling.lower().startswith('cls'):
                DoE = uqra.CLS(self.doe_sampling, ndim)
                xi_cand = DoE.samples(size=1000000)
                if self.doe_sampling.upper() in ['CLS4', 'CLS5']:
                    xi_cand = xi_cand * deg ** 0.5
                xi_min = np.amin(centroid_xi, axis=1) - epsilon
                xi_max = np.amax(centroid_xi, axis=1) + epsilon
                assert len(xi_min) == ndim
                idx = np.ones((1000000), dtype=bool) 
                for ixi_cand, a, b in zip(xi_cand, xi_min, xi_max):
                    idx_= np.logical_and(ixi_cand >= a, ixi_cand <=b)
                    idx = np.logical_and(idx, idx_)
                xi_cand = xi_cand[:,idx]
            else:
                raise ValueError('{:s} not defined'.foramt(self.doe_sampling))
            idx_DoI_xi_cand = []

            for i, xi in enumerate(centroid_xi.T):
                xi = xi.reshape(ndim, 1)
                idx_DoI_xi_cand_icentroid = np.argwhere(np.linalg.norm(xi_cand-xi, axis=0) < epsilon).flatten().tolist()
                nsamples_each_centroid[i] = nsamples_each_centroid[i] + len(idx_DoI_xi_cand_icentroid)
                idx_DoI_xi_cand = list(set(idx_DoI_xi_cand+ idx_DoI_xi_cand_icentroid))
            DoI_cand_xi.append(xi_cand[:, idx_DoI_xi_cand])

            if np.sum(nsamples_each_centroid) > 1000:
                DoI_cand_xi = np.concatenate(DoI_cand_xi, axis=1)
                break
        return DoI_cand_xi

    def samples_nearby(self, y0, data_xi, data_y, data_cand, deg, n0=10, epsilon=0.1, return_index=True):
        data_cand_xi = data_cand
        ### locate samples close to estimated y0 (domain of interest)
        idx_DoI_data_test = np.argsort(abs(data_y-y0))[:n0] 
        idx_DoI_data_cand = []
        for idx_ in idx_DoI_data_test:
            xi = data_xi[:, idx_].reshape(-1, 1)
            idx_DoI_data_cand_ = np.argwhere(np.linalg.norm(data_cand_xi -xi, axis=0) < epsilon).flatten().tolist()
            ### xi is outside data cand 
            if len(idx_DoI_data_cand_) == 0:
                idx_DoI_data_cand_ = np.argsort(np.linalg.norm(data_cand_xi -xi, axis=0))[:100].tolist()
            idx_DoI_data_cand = list(set(idx_DoI_data_cand + idx_DoI_data_cand_))
        data_cand_DoI = data_cand[:, idx_DoI_data_cand] 
        if return_index:
            res = (data_cand_DoI, idx_DoI_data_cand)
        else:
            res = data_cand_DoI
        return res 

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
            raise ValueError(' Error: {:s}-{:s} is not defined'.format(doe_sampling, poly_name))

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

