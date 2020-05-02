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
import scipy.stats as stats
import multiprocessing as mp
from sklearn import linear_model
from sklearn import model_selection
from ._surrogatebase import SurrogateBase
import museuq, math
from scipy import sparse
class PolynomialChaosExpansion(SurrogateBase):
    """
    Class to build polynomial chaos expansion (PCE) model
    """

    def __init__(self, basis=None, random_seed=None):
        super().__init__(random_seed=random_seed)
        self.name           = 'Polynomial Chaos Expansion'
        self.nickname       = 'PCE'
        self.basis          = basis  ### a list of marginal basis

        if basis is None:
            self.ndim       = None 
            self.num_basis  = None 
            self.deg        = None
            self.cv_error   = np.inf
            self.active_index = None
            self.active_basis = None
        else:
            self.ndim       = basis.ndim 
            self.num_basis  = self.basis.num_basis
            self.deg        = self.basis.deg
            self.active_index = None if self.num_basis is None else range(self.num_basis)
            self.active_basis = None if self.basis.basis_degree is None else self.basis.basis_degree 
            self.cv_error   = np.inf
            if self.deg is None:
                self.tag        = '{:d}{:s}0'.format(self.ndim, self.basis.nickname)
            else:
                self.tag        = '{:d}{:s}{:d}'.format(self.ndim, self.basis.nickname,self.deg)

            # ### Now assuming same marginal basis
            # try:
                # dist_name = self.basis[0].name 
            # except AttributeError:
                # dist_name = self.basis[0].dist.name 

            # if dist_name == 'norm':
                # self.basis = museuq.Hermite(d=self.ndim, deg=self.deg)
            # elif dist_name == 'uniform':
                # self.basis = museuq.Legendre(d=self.ndim, deg=self.deg)
            # else:
                # raise ValueError('Polynomial for {} has not been defined yet'.format(basis[0].name))

    def info(self):
        print(r'   - {:<25s} : {:<20s}'.format('Surrogate Model Name', self.name))
        if self.deg is not None:
            print(r'     - {:<23s} : {}'.format('Askey-Wiener basis'   , self.basis.name))
            print(r'     - {:<23s} : {}'.format('Polynomial order (p)', self.deg ))
            print(r'     - {:<23s} : {:d}'.format('No. poly basis   (P)', self.basis.num_basis))
            print(r'     - {:<23s} : {:d}'.format('No. active basis (s)', len(self.active_index)))

    def set_degree(self, p):
        if self.basis is None:
            raise ValueError('Basis is not defined')
        else:
            self.basis.set_degree(p)
            self.num_basis = self.basis.num_basis
            self.deg       = self.basis.deg
            self.active_index = None if self.num_basis is None else range(self.num_basis)
            self.active_basis = None if self.basis.basis_degree is None else self.basis.basis_degree 
    
    def fit(self, method, x, y, w=None, **kwargs):
        if method.lower().startswith('quad'):
            self.fit_quadrature(x,w,y)
        elif method.lower() == 'ols':
            self.fit_ols(x,y,w=w, **kwargs)
        elif method.lower() == 'olslars':
            self.fit_olslars(x,y,w=w,**kwargs)
        elif method.lower() == 'lassolars':
            self.fit_lassolars(x,y,sample_weight=w,**kwargs)
        else:
            raise NotImplementedError

    def fit_quadrature(self, x, w, y):
        """
        fit with quadrature points
        """
        self.fit_method = 'GLK' 
        x = np.array(x, copy=False, ndmin=2)
        y = np.array(y, copy=False, ndmin=2)
        X = self.basis.vandermonde(x, normed=False)
        y = np.squeeze(y) + 0.0
        w = np.asarray(w) + 0.0
        if w.ndim != 1:
            raise TypeError("expected 1D vector for w")
        if len(x.T) != len(w):
            raise TypeError("expected x and w to have same length")

        # norms = np.sum(X.T**2 * w, -1)
        norms       = self.basis.basis_norms *self.basis.basis_norms_const**self.basis.ndim
        coef        = np.sum(X.T * y * w, -1) / norms 
        self.model  = self.basis.set_coef(coef) 
        self.coef   = coef
        self.active_index = range(self.num_basis)
        self.active_basis = self.basis.basis_degree

    def fit_ols(self,x,y,w=None,  **kwargs):
        """
        Fit PCE meta model with (weighted) Ordinary Least Error  

        Arguments:
            x: array-like of shape (ndim, nsamples) 
                sample input values in zeta (selected Wiener-Askey distribution) space
            y: array-like of shape (nsamples [,n_output_dims/nQoI])
                QoI observations

            w: array-like weights, optional
            n_splits: number of folders used in cross validation, default nsamples, i.e.: leave one out 
        Returns:

        """
        self.fit_method = 'OLS' 
        x = np.array(x, copy=False, ndmin=2)
        y = np.array(y, copy=False, ndmin=2)
        X = self.basis.vandermonde(x)
        y = np.squeeze(y)
        active_basis = kwargs.get('active_basis', None)
        if active_basis is None:
            active_index = np.arange(self.basis.num_basis).tolist()
        else:
            active_index = [i for i in range(self.basis.num_basis) if self.basis.basis_degree[i] in active_basis]
        X = X[:, active_index]

        n_splits= kwargs.get('n_splits', X.shape[0])
        n_splits= min(n_splits, X.shape[0])
        kf      = model_selection.KFold(n_splits=n_splits,shuffle=True)

        ## calculate k-folder cross-validation error
        model   = linear_model.LinearRegression(fit_intercept=False)
        neg_mse = model_selection.cross_val_score(model, X, y, scoring = 'neg_mean_squared_error', cv=kf, n_jobs=mp.cpu_count())
        ## fit the model with all data 
        ## X has a column with ones, and want return coefficients to incldue that column
        model.fit(X, y, sample_weight=w)
        self.cv_error= -np.mean(neg_mse)
        self.model   = model 
        self.coef    = model.coef_
        if active_basis is None:
            self.active_index = range(self.num_basis)
            self.active_basis = self.basis.basis_degree
        else:
            self.active_index = active_index
            self.active_basis = active_basis
        self.score   = model.score(X,y,w)

    def fit_olslars(self,x,y,w=None, **kwargs):
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
        self.fit_method = 'OLSLARS' 
        x = np.array(x, copy=False, ndmin=2)
        y = np.array(y, copy=False, ndmin=2)
        X = self.basis.vandermonde(x)
        y = np.squeeze(y)
        ## parameters for LassoLars 
        n_splits= kwargs.get('n_splits', X.shape[0])
        n_splits= min(n_splits, X.shape[0])
        kf      = model_selection.KFold(n_splits=n_splits,shuffle=True)

        ### 1. Perform variable selection first
        model_lars       = linear_model.Lars(fit_intercept=False).fit(X,y)
        ### 2. Perform linear regression on every set of first i basis 
        n_active_basis = min(len(model_lars.active_), X.shape[0]-1)
        for i in range(n_active_basis):
            active_indices = model_lars.active_[:i+1]
            # active_indices = np.unique(np.array([0, *active_indices])) ## always has column of ones
            X_             = X[:, active_indices]
            ### Calculate loo error for each basis set
            model          = linear_model.LinearRegression(fit_intercept=False)
            neg_mse        = model_selection.cross_val_score(model, X_,y,scoring = 'neg_mean_squared_error', cv=kf, n_jobs=mp.cpu_count())
            error_loo      = -np.mean(neg_mse)
            ### Fitting with all samples
            model.fit(X_,y, sample_weight=w)

            if error_loo < self.cv_error:
                self.model    = model 
                self.cv_error = error_loo
                self.coef     = model.coef_
                self.active_index = active_indices
                self.active_basis = [self.basis.basis_degree[i] for i in self.active_index]

    def fit_lassolars(self,x,y,sample_weight=None, **kwargs):
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
        self.fit_method = 'LASSOLARS' 
        x = np.array(x, copy=False, ndmin=2)
        y = np.array(y, copy=False, ndmin=1)
        X = self.basis.vandermonde(x)
        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            X, y = self._rescale_data(X, y, sample_weight)
        ## parameters for LassoLars 
        n_splits= kwargs.get('n_splits', X.shape[0])
        n_splits= min(n_splits, X.shape[0])
        max_iter= kwargs.get('max_iter', 500)
        epsilon = kwargs.get('epsilon', 1e-6)
        kf      = model_selection.KFold(n_splits=n_splits,shuffle=True)

        try:    
            model         = linear_model.LassoLarsCV(max_iter=max_iter,cv=kf, n_jobs=mp.cpu_count(),fit_intercept=False).fit(X,y)
        except ValueError as e:
            #### looks like a bug in KFold
            tqdm.write(e)
            return
        self.model    = model 
        self.cv_error = np.min(np.mean(model.mse_path_, axis=1))
        self.coef     = model.coef_
        self.active_index = [i for i, icoef in enumerate(model.coef_) if abs(icoef) > epsilon]
        self.active_basis = [self.basis.basis_degree[i] for i in self.active_index]
        self.sparsity = len(self.active_index)
        self.score    = model.score(X, y)

    def mean(self):
        return self.coef[0]

    def var(self, pct=1):
        cum_var = -np.cumsum(np.sort(-self.coef[1:] **2))
        if cum_var[-1] == 0:
            self.var_basis_index = [0]
            self.var_pct_basis  = [self.basis.basis_degree[0]]
        else:
            y_hat_var_pct = cum_var / cum_var[-1] 
            n_pct_var_term= np.argwhere(y_hat_var_pct > pct)[0][-1] + 1 ## return index +1 since cum_var starts with 1, not 0
            self.var_basis_index= [0,] + list(np.argsort(-self.coef[1:])[:n_pct_var_term+1]+1) ## +1 to always count phi_0
            self.var_pct_basis  = [self.basis.basis_degree[i] for i in self.var_basis_index]

    def predict(self,x, **kwargs):
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
            y = self.basis(x)

        elif self.fit_method in ['OLS']:
            size_of_array_4gb = 1e8/2.0
            if x.shape[1] * self.num_basis < size_of_array_4gb:
                X = self.basis.vandermonde(x)[:, self.active_index]
                y = self.model.predict(X)
            else:
                batch_size = math.floor(size_of_array_4gb/self.num_basis)  ## large memory is allocated as 8 GB
                y = []
                for i in range(math.ceil(x.shape[1]/batch_size)):
                    idx_beg = i*batch_size
                    idx_end = min((i+1) * batch_size, x.shape[1])
                    x_      = x[:,idx_beg:idx_end]
                    X_      = self.basis.vandermonde(x_)[:, self.active_index]
                    y_      = self.model.predict(X_)
                    y      += list(y_)
                y = np.array(y) 
        elif self.fit_method in ['OLSLARS','LASSOLARS']:
            size_of_array_4gb = 1e8/2.0
            if x.shape[1] * self.num_basis < size_of_array_4gb:
                X = self.basis.vandermonde(x)
                y = self.model.predict(X)
            else:
                batch_size = math.floor(size_of_array_4gb/self.num_basis)  ## large memory is allocated as 8 GB
                y = []
                for i in range(math.ceil(x.shape[1]/batch_size)):
                    idx_beg = i*batch_size
                    idx_end = min((i+1) * batch_size, x.shape[1])
                    x_      = x[:,idx_beg:idx_end]
                    X_      = self.basis.vandermonde(x_)
                    y_      = self.model.predict(X_)
                    y      += list(y_)
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

    def _rescale_data(self, X, y, sample_weight):
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
        y = sw_matrix @ y
        return X, y
