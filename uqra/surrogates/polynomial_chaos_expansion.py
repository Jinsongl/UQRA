#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import numpy as np
from tqdm import tqdm
import copy
import scipy.stats as stats
import multiprocessing as mp
from sklearn import linear_model, preprocessing, model_selection
from sklearn.exceptions import ConvergenceWarning
from ._surrogatebase import SurrogateBase
import uqra, math
from scipy import sparse
class PolynomialChaosExpansion(SurrogateBase):
    """
    Class to build polynomial chaos expansion (PCE) model
    """

    def __init__(self, orth_poly=None, random_seed=None):
        super().__init__()
        self.name           = 'Polynomial Chaos Expansion'
        self.nickname       = 'PCE'
        self.orth_poly      = orth_poly  ### UQRA.polynomial object

        if orth_poly is None:
            self.ndim       = None 
            self.deg        = None
            self.cv_error   = np.inf
            self.num_basis  = None 
            self.active_index = None
            self.active_basis = None
            self.basis_degree = None
            self.sparsity   = 0
        else:
            self.ndim       = orth_poly.ndim 
            self.deg        = self.orth_poly.deg
            self.cv_error   = np.inf
            self.num_basis  = self.orth_poly.num_basis
            self.active_index = list(range(self.num_basis))
            self.active_basis = self.orth_poly.basis_degree 
            self.basis_degree = self.orth_poly.basis_degree 
            self.sparsity   = len(self.active_index)
            if self.deg is None:
                self.tag        = '{:d}{:s}0'.format(self.ndim, self.orth_poly.nickname[:3])
            else:
                self.tag        = '{:d}{:s}{:d}'.format(self.ndim, self.orth_poly.nickname[:3], self.deg)

    def info(self):
        print(r'   - {:<25s} : {:<20s}'.format('Surrogate Model Name', self.name))
        if self.deg is not None:
            print(r'     - {:<23s} : {}'.format('Askey-Wiener polynomial'   , self.orth_poly.name))
            print(r'     - {:<23s} : {}'.format('Polynomial dimension ', self.ndim))
            print(r'     - {:<23s} : {}'.format('Polynomial order (p)', self.deg ))
            print(r'     - {:<23s} : {:d}'.format('No. polynomial basis(P)', self.num_basis))
            print(r'     - {:<23s} : {:d}'.format('No. active basis (s)', len(self.active_index)))

    def set_degree(self, p):
        if self.orth_poly is None:
            raise ValueError('Basis is not defined')
        else:
            self.orth_poly.set_degree(p)
            self.num_basis = self.orth_poly.num_basis
            self.deg       = self.orth_poly.deg
            self.cv_error     = np.inf
            self.active_index = list(range(self.num_basis))
            self.active_basis = self.orth_poly.basis_degree 
            self.basis_degree = self.orth_poly.basis_degree 
            self.tag          = '{:d}{:s}{:d}'.format(self.ndim, self.orth_poly.nickname,self.deg)
    
    def weight_func(self, x, w, *args, **kwargs):
        if callable(w):
            return w(x, *args, **kwargs)
        elif w is None:
            return None
        elif isinstance(w, str):
            if w.lower().startswith('cls') or w.lower() == 'christoffel':
                w = self.christoffel_weight(x, **kwargs)
                return w
            elif w.lower()[:3] in ['mcs', 'lhs']:
                return None
            else:
                raise ValueError(' weight {} not defined'.format(w))
        else:
            return w

    def fit(self, method, x, y, w=None, **kwargs):
        """
        fit PCE model with data set(x,y) with specified method
        """

        self.fit_method = str(method).upper()

        x = np.array(x, copy=False, ndmin=2)
        y = np.array(y, copy=False, ndmin=2)
        X = self.orth_poly.vandermonde(x)
        y = np.squeeze(y)

        if method.lower().startswith('quad'):
            self._fit_quadrature(x,w,y)

        elif method.lower() == 'ols':
            fit_intercept = kwargs.get('fit_intercept'  , True )
            normalize     = kwargs.get('normalize'      , False)
            n_jobs        = kwargs.get('n_jobs'         , None ) 
            active_basis  = kwargs.get('active_basis'   , None )
            n_splits      = min(kwargs.get('n_splits'   , x.shape[1]), x.shape[1])  ## if not given, default is Leave-one-out
            if active_basis is None or len(active_basis) == 0:
                self.active_basis = self.orth_poly.basis_degree
                self.active_index = np.arange(self.orth_poly.num_basis).tolist()
            else:
                self.active_basis = active_basis
                self.active_index = [i for i, ibasis_degree in enumerate(self.orth_poly.basis_degree) if ibasis_degree in active_basis]
            if len(self.active_index) == 0:
                raise ValueError('UQRA.PCE.fit(OLS): active basis could not be empty ')
            X = X[:, self.active_index]
            ### build OLS model
            ols_reg = linear_model.LinearRegression(fit_intercept=fit_intercept, normalize=normalize, n_jobs=n_jobs)
            kfolder = model_selection.KFold(n_splits=n_splits,shuffle=True)
            ## calculate k-folder cross-validation error

            w = self.weight_func(x, w, active=self.active_index)
            WX, Wy = self._rescale_data(X, y, w) if w is not None else (X, y)
            s,v,d = np.linalg.svd(WX)


            neg_mse = model_selection.cross_val_score(ols_reg, WX, Wy, 
                    scoring='neg_mean_squared_error', cv=kfolder, n_jobs=n_jobs)
            self.model = ols_reg.fit(X, y, sample_weight=w)
            self.cv_error = -np.mean(neg_mse)
            self.coef    = np.array(copy.deepcopy(ols_reg.coef_), ndmin=2)
            self.coef[:,0] = self.coef[:,0] + ols_reg.intercept_
            self.coef    = np.squeeze(self.coef)
            self.score   = ols_reg.score(X,y,w)

        elif method.lower() == 'olslar':
            """
            (weighted) Ordinary Least Error on selected orth_poly (LARs)
            Reference: Blatman, Géraud, and Bruno Sudret. "Adaptive sparse polynomial chaos expansion based on least angle regression." Journal of Computational Physics 230.6 (2011): 2345-2367.
            Arguments:
                x: array-like of shape (ndim, nsamples) 
                    sample input values in zeta (selected Wiener-Askey distribution) space
                y: array-like of shape (nsamples [,n_output_dims/nQoI])
                    QoI observations

                w: array-like weights, optional
                n_splits: number of folders used in cross validation, default nsamples, i.e.: leave one out 
            Returns:

            """
            fit_intercept = kwargs.get('fit_intercept'  , False)
            normalize     = kwargs.get('normalize'      , False)
            n_jobs        = kwargs.get('n_jobs'         , None ) 
            shrinkage     = kwargs.get('shrinkage'      , 'CV' )
            n_splits      = kwargs.get('n_splits'  , x.shape[1]) 

            full_weight = self.weight_func(x, w, active=self.active_index)
            WX, Wy = self._rescale_data(X, y, full_weight) if full_weight is not None else (X, y)

            ### -------------------------------------------------------------------------------------------------------
            ### standardize data: zero mean, unit variance for each predictor
            ### need to be completed
            # scaler     = preprocessing.StandardScaler().fit(WX)
            # WX_scaled  = scaler.transform(WX) 
            # Wy_scaled  = Wy - np.mean(Wy)  ## just zero mean, not unit variance
            # model_lars = linear_model.Lars(fit_intercept=fit_intercept, normalize=normalize).fit(WX_scaled,Wy_scaled)
            ### -------------------------------------------------------------------------------------------------------

            model_lars = linear_model.Lars(fit_intercept=fit_intercept, normalize=normalize).fit(WX,Wy)
            if shrinkage.upper() in ['CV', 'CROSS VALIDATION', 'CROSS-VALIDATION']:
                ### if not given, default is Leave-one-out
                n_splits = min(n_splits, x.shape[1])  ## avoid number of samples less than # folders
                kfolder  = model_selection.KFold(n_splits=n_splits,shuffle=True)

                cv_err_path  = []
                cp_statistics = []
                ### OLS regression with frist k basis
                for k in range(1, min(len(model_lars.active_), X.shape[1])+1):
                    active_ = model_lars.active_[:k] #np.unique([0,] + model_lars.active_[:k]).tolist()
                    kfolder = model_selection.KFold(n_splits=n_splits,shuffle=True)
                    reg_ols = linear_model.LinearRegression(fit_intercept=False)
                    X_      = X[:,active_]
                    k_weight= self.weight_func(x, w, active=active_) 
                    WX_, Wy = self._rescale_data(X_, y, k_weight) if k_weight is not None else (X_, y) 
                    neg_mse = model_selection.cross_val_score(reg_ols, WX_, Wy, 
                            scoring='neg_mean_squared_error', cv=kfolder)
                    cv_err_path.append( -np.mean(neg_mse))
                k = np.argmin(cv_err_path) +1
                active_index  = model_lars.active_[:k] #if 0 in model_lars.active_[:k] else model_lars.active_[:k] + [0,]
                active_basis  = [self.orth_poly.basis_degree[i] for i in active_index] 
                self.cv_err_path   = cv_err_path
            elif shrinkage.upper() in ['CP', 'CP-STATISTICS', 'CPSTATISTICS']:
                ### OLS regression with all basis
                ols_reg0= linear_model.LinearRegression(fit_intercept=True).fit(X, y, full_weight)
                y_hat0  = ols_reg0.predict(X)
                std0    = np.sqrt(np.linalg.norm(y-y_hat0)**2/(X.shape[0]-X.shape[1]-1))
                reg_ols.fit(X_, y, k_weight)
                y_hat = reg_ols.predict(X_)
                cp_statistics.append(np.linalg.norm(y-y_hat)**2/std0**2 - X_.shape[0] + 2*k)
                # k = np.argmin(cp_statistics)+ 1
                # print(k)
                raise NotImplementedError
            self.fit('OLS', x,y,w=w, active_basis=active_basis, fit_intercept=fit_intercept, 
                    normalize=normalize, n_jobs=n_jobs, n_splits=n_splits)
            self.Lars = model_lars

        elif method.lower().startswith('lasso'):
            fit_intercept = kwargs.get('fit_intercept'  , True )
            normalize     = kwargs.get('normalize'      , False)
            n_jobs        = kwargs.get('n_jobs'         , None ) 
            max_iter      = kwargs.get('max_iter'       , 500  ) 
            n_splits      = min(kwargs.get('n_splits', x.shape[1]), x.shape[1])  ## if not given, default is Leave-one-out

            self._fit_lassolars(X,y,w=w, fit_intercept=fit_intercept, normalize=normalize, n_jobs=n_jobs, n_splits=n_splits)

        else:
            raise ValueError(' {:s} not defined for UQRA.PCE.fit method'.format(method)) 

    def _fit_quadrature(self, x, w, y):
        """
        fit with quadrature points
        """
        self.fit_method = 'GLK' 
        x = np.array(x, copy=False, ndmin=2)
        y = np.array(y, copy=False, ndmin=2)
        X = self.orth_poly.vandermonde(x, normed=False)
        y = np.squeeze(y) + 0.0
        w = np.asarray(w) + 0.0
        if w.ndim != 1:
            raise TypeError("expected 1D vector for w")
        if len(x.T) != len(w):
            raise TypeError("expected x and w to have same length")

        # norms = np.sum(X.T**2 * w, -1)
        norms       = self.orth_poly.basis_norms *self.orth_poly.basis_norms_const**self.orth_poly.ndim
        coef        = np.sum(X.T * y * w, -1) / norms 
        self.model  = self.orth_poly.set_coef(coef) 
        self.coef   = coef
        self.active_index = range(self.num_basis)
        self.active_basis = self.orth_poly.basis_degree

    def _fit_lassolars(self,x,y,w=None, **kwargs):
        """
        (weighted) Ordinary Least Error on selected basis (LARs)
        Reference: Blatman, Géraud, and Bruno Sudret. "Adaptive sparse polynomial chaos expansion based on least angle regression." Journal of Computational Physics 230.6 (2011): 2345-2367.
        Arguments:
            x: array-like of shape (ndim, nsamples) 
                sample input values in zeta (selected Wiener-Askey distribution) space
            y: array-like of shape (nsamples [,n_output_dims/nQoI])
                QoI observations

            w: array-like weights, optional
            n_splits: number of folders used in cross validation, default nsamples, i.e.: leave one out 
        Returns:

        """
        ## parameters for LassoLars 
        kf = model_selection.KFold(n_splits=n_splits,shuffle=True)

        if y.ndim == 1:
            if w is not None:
            # Sample weight can be implemented via a simple rescaling.
                X, y = self._rescale_data(X, y, w)
            try:    
                model  = linear_model.LassoLarsCV(max_iter=max_iter,cv=kf, n_jobs=cpu_count).fit(X,y)
            except ValueError as e:
                #### looks like a bug in KFold
                print(e)
                return
            self.model    = model 
            self.cv_error = np.min(np.mean(model.mse_path_, axis=1))
            self.coef     = copy.deepcopy(model.coef_)
            self.coef[0]  = model.intercept_
            self.active_index = np.unique([0,] + [i for i, icoef in enumerate(model.coef_) if abs(icoef) > epsilon])
            self.active_basis = [self.orth_poly.basis_degree[i] for i in self.active_index]
            self.sparsity = len(self.active_index)
            self.score    = model.score(X, y)
            # self._least_ns_ratio()
        elif y.ndim == 2:
            raise NotImplementedError
            model_ls      = []
            cv_error_ls   = []
            coef_ls       = []
            active_index_ls = []
            active_basis_ls = []
            sparsity_ls   = []
            score_ls      = []
            for iy in y.T:
                if w is not None:
                    # Sample weight can be implemented via a simple rescaling.
                    X, iy = self._rescale_data(X, iy, w)

                try:    
                    model  = linear_model.LassoLarsCV(max_iter=max_iter,cv=kf, n_jobs=cpu_count).fit(X,iy)
                except ValueError as e:
                    #### looks like a bug in KFold
                    print(e)
                    return
                except ConvergenceWarning:
                    print('ConvergenceWarning: ')
                    print(X.shape)
                    print(iy.shape)

                model_ls.append(model)
                cv_error_ls.append(np.min(np.mean(model.mse_path_, axis=1)))
                coef_ls.append(model.coef_)
                active_index_ls.append([i for i, icoef in enumerate(model.coef_) if abs(icoef) > epsilon])
                active_index_ls += [0,]
                active_basis_ls.append([self.orth_poly.basis_degree[i] for i in active_index_ls[-1]])
                sparsity_ls.append(len(active_index_ls[-1]))
                score_ls.append(model.score(X,iy))

            self.model      = model_ls
            self.cv_error   = np.array(cv_error_ls)
            self.coef       = np.array(coef_ls)
            self.active_index=active_index_ls
            self.active_basis=active_basis_ls
            self.sparsity   = np.array(sparsity_ls)
            self.score      = np.array(score_ls)

    # def _fit_olslars(self,x,y,w=None, **kwargs):
        # """
        # (weighted) Ordinary Least Error on selected orth_poly (LARs)
        # Reference: Blatman, Géraud, and Bruno Sudret. "Adaptive sparse polynomial chaos expansion based on least angle regression." Journal of Computational Physics 230.6 (2011): 2345-2367.
        # Arguments:
            # x: array-like of shape (ndim, nsamples) 
                # sample input values in zeta (selected Wiener-Askey distribution) space
            # y: array-like of shape (nsamples [,n_output_dims/nQoI])
                # QoI observations

            # w: array-like weights, optional
            # n_splits: number of folders used in cross validation, default nsamples, i.e.: leave one out 
        # Returns:

        # """
        # x = np.array(x, copy=False, ndmin=2)
        # y = np.array(y, copy=False, ndmin=2)
        # X = self.orth_poly.vandermonde(x)
        # y = np.squeeze(y)
        # ## parameters for LassoLars 
        # n_splits= kwargs.get('n_splits', X.shape[0])
        # n_splits= min(n_splits, X.shape[0])
        # kf      = model_selection.KFold(n_splits=n_splits,shuffle=True)
        # cpu_count = kwargs.get('cpu_count', mp.cpu_count())

        # ### 1. Perform variable selection first
        # model_lars = linear_model.Lars().fit(X,y)
        # ### 2. Perform linear regression on every set of first i basis 
        # n_active_basis = min(len(model_lars.active_), X.shape[0]-1)
        # for i in range(n_active_basis):
            # active_indices = model_lars.active_[:i+1]
            # # active_indices = np.unique(np.array([0, *active_indices])) ## always has column of ones
            # X_             = X[:, active_indices]
            # ### Calculate loo error for each basis set
            # model          = linear_model.LinearRegression()
            # neg_mse        = model_selection.cross_val_score(model, X_,y,scoring = 'neg_mean_squared_error', cv=kf, n_jobs=cpu_count)
            # error_loo      = -np.mean(neg_mse)
            # ### Fitting with all samples
            # model.fit(X_,y, w=w)

            # if error_loo < self.cv_error:
                # self.model    = model 
                # self.cv_error = error_loo
                # self.coef     = model.coef_
                # self.active_index = active_indices
                # self.active_basis = [self.orth_poly.basis_degree[i] for i in self.active_index]

    def mean(self):
        return self.coef[0]

    def var(self, pct=1):
        cum_var = -np.cumsum(np.sort(-self.coef[1:] **2))
        if cum_var[-1] == 0:
            self.active_index = [0]
            self.active_basis  = [self.orth_poly.basis_degree[0]]
        else:
            y_hat_var_pct = cum_var / cum_var[-1] 
            n_pct_var_term= np.argwhere(y_hat_var_pct > pct)[0][-1] + 1 ## return index +1 since cum_var starts with 1, not 0
            self.active_index= [0,] + list(np.argsort(-self.coef[1:])[:n_pct_var_term+1]+1) ## +1 to always count phi_0
            self.active_basis  = [self.orth_poly.basis_degree[i] for i in self.active_index]

    def predict(self, x, **kwargs):
        n_jobs = kwargs.get('n_jobs', 1)
        if n_jobs == 1:
            y = self._predict(x)
        else:
            parallel_batch_size = int(1e6)
            parallel_batch_x    = []
            idx0, idx1 = 0, min(parallel_batch_size, x.shape[1])
            while idx1 <= x.shape[1] and idx0 < idx1:
                parallel_batch_x.append(x[:, idx0:idx1])
                idx0 = idx1
                idx1 = min(idx1+parallel_batch_size, x.shape[1])
            with mp.Pool(processes=n_jobs) as p:
                y = list(p.imap(self._predict, parallel_batch_x))
            y = np.concatenate(y, axis=0)
        return y

    def _predict(self,x, **kwargs):
        """
        Predict using surrogate models 

        Arguments:	
        X : array-like, shape = (n_features/ndim, nsamples)
            Query points where the surrogate model are evaluated

        Returns:	
        y : list of array, array shape = (nsamples, )
            predicted value from surrogate models at query points
        """
        if x.shape[0] != self.ndim:
            raise ValueError('Expecting {:d}-D sampels, but {:d} given'.format(self.ndim, x.shape[0]))

        if self.fit_method == 'GLK':
            y = self.orth_poly(x)

        elif self.fit_method in ['OLS']:

            size_of_array_4gb = 1e8/2.0

            if x.shape[1] * self.num_basis < size_of_array_4gb:
                X = self.orth_poly.vandermonde(x)[:, self.active_index]
                y = self.model.predict(X)
            else:
                batch_size = math.floor(size_of_array_4gb/self.num_basis)  ## large memory is allocated as 8 GB
                y = []
                for i in range(math.ceil(x.shape[1]/batch_size)):
                    idx_beg = i*batch_size
                    idx_end = min((i+1) * batch_size, x.shape[1])
                    x_      = x[:,idx_beg:idx_end]
                    X_      = self.orth_poly.vandermonde(x_)[:, self.active_index]
                    y_      = self.model.predict(X_)
                    y      += list(y_)
                y = np.array(y) 

        elif self.fit_method in ['OLSLARS']:
            size_of_array_4gb = 1e8/2.0
            if x.shape[1] * self.num_basis < size_of_array_4gb:
                X = self.orth_poly.vandermonde(x)[:, self.active_index]
                y = self.model.predict(X)
            else:
                batch_size = math.floor(size_of_array_4gb/self.num_basis)  ## large memory is allocated as 8 GB
                y = []
                for i in range(math.ceil(x.shape[1]/batch_size)):
                    idx_beg = i*batch_size
                    idx_end = min((i+1) * batch_size, x.shape[1])
                    x_      = x[:,idx_beg:idx_end]
                    X_      = self.orth_poly.vandermonde(x_)[:, self.active_index]
                    y_      = self.model.predict(X_)
                    y      += list(y_)
                y = np.array(y) 


        elif self.fit_method in ['LASSOLARS']:
            size_of_array_4gb = 1e8/2.0
            if x.shape[1] * self.num_basis < size_of_array_4gb:
                X = self.orth_poly.vandermonde(x)
                try: 
                    y = self.model.predict(X)
                except AttributeError:
                    y = np.array([imodel.predict(X) for imodel in self.model])
            else:
                batch_size = math.floor(size_of_array_4gb/self.num_basis)  ## large memory is allocated as 8 GB
                try:
                    y = []
                    for i in range(math.ceil(x.shape[1]/batch_size)):
                        idx_beg = i*batch_size
                        idx_end = min((i+1) * batch_size, x.shape[1])
                        x_      = x[:,idx_beg:idx_end]
                        X_      = self.orth_poly.vandermonde(x_)
                        y_      = self.model.predict(X_)
                        y      += list(y_)
                except:
                    y = [[] for _ in self.model]
                    for i in range(math.ceil(x.shape[1]/batch_size)):
                        idx_beg = i*batch_size
                        idx_end = min((i+1) * batch_size, x.shape[1])
                        x_      = x[:,idx_beg:idx_end]
                        X_      = self.orth_poly.vandermonde(x_)
                        for j, imodel in enumerate(self.model):
                            y_      = imodel.predict(X_)
                            y[j]   += list(y_)

                y = np.array(y) 


        return y

    def sample_y(self,X, nsamples=1, random_state=0):
        """
        Draw samples from Surrogate model and evaluate at X.

        Parameters:	
        X : array-like, shape = (n_features,n_samples_X)
        Query points where the GP samples are evaluated

        nsamples : int, default: 1
        The number of samples drawn from the Gaussian process

        random_state : int, RandomState instance or None, optional (default=0)
        If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.

        Returns:	
        y_samples : list, list item shape = (n_samples_X, [n_output_dims], nsamples)
        Values of nsamples samples drawn from Gaussian process and evaluated at query points.

        """
        raise NotImplementedError

    def _rescale_data(self, X, y, w):
        """Rescale data so as to support w"""
        n_samples = X.shape[0]
        w = np.asarray(w)
        if w.ndim == 0:
            w = np.full(n_samples, w, dtype=w.dtype)
        w = np.sqrt(w)
        sw_matrix = sparse.dia_matrix((w, 0), shape=(n_samples, n_samples))
        X = sw_matrix @ X
        y = sw_matrix @ y
        return X, y

    def _least_ns_ratio(self):
        """
        """
        if self.orth_poly.nickname == 'Leg':
            ratio_sp = self.sparsity / self.num_basis
            if ratio_sp > 0.8:
                self.least_ns_ratio = 1.5
            elif ratio_sp > 0.6:
                self.least_ns_ratio = 1.8
            elif ratio_sp > 0.5:
                self.least_ns_ratio = 2.0
            elif ratio_sp > 0.4:
                self.least_ns_ratio = 2.2
            elif ratio_sp > 0.3:
                self.least_ns_ratio = 2.8
            elif ratio_sp > 0.2:
                self.least_ns_ratio = 3.5
            elif ratio_sp > 0.1:
                self.least_ns_ratio = 4.5
            else:
                self.least_ns_ratio = 6.0
        elif self.orth_poly.nickname.lower().startswith('hem'):
            ratio_sp = self.sparsity / self.num_basis
            if ratio_sp > 0.8:
                self.least_ns_ratio = 1.5
            elif ratio_sp > 0.6:
                self.least_ns_ratio = 1.8
            elif ratio_sp > 0.5:
                self.least_ns_ratio = 2.0
            elif ratio_sp > 0.4:
                self.least_ns_ratio = 2.2
            elif ratio_sp > 0.3:
                self.least_ns_ratio = 2.8
            elif ratio_sp > 0.2:
                self.least_ns_ratio = 3.5
            elif ratio_sp > 0.1:
                self.least_ns_ratio = 4.5
            else:
                self.least_ns_ratio = 6.0
        else:
            raise NotImplementedError

    def christoffel_weight(self, u, active=None):
        """
        return normalized Christoffel function P/sum(phi(x)**2)

        arguments:
            u: ndarray of shape (ndim, nsamples), random input variable
            basis: uqra.polynomial object, Space basis function
            active: list of active indices of basis, (columns of vander(u))
        """
        u = np.array(u, ndmin=2)
        U = self.orth_poly.vandermonde(u) ## default normalized vandermonde
        if active is not None:
            U = U[:, active]
        Kp= np.sum(U*U, axis=1)
        w = U.shape[1]/Kp
        return w

