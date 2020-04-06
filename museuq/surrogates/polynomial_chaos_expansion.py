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
import museuq
class PolynomialChaosExpansion(SurrogateBase):
    """
    Class to build polynomial chaos expansion (PCE) model
    """

    def __init__(self, distributions=None, random_seed=None):
        super().__init__(random_seed=random_seed)
        self.name           = 'Polynomial Chaos Expansion'
        self.nickname       = 'PCE'
        self.basis          = distributions  ### a list of marginal distributions

        if distributions is None:
            self.ndim       = None 
            self.num_basis  = None 
            self.deg        = None
            self.active_    = None
            self.cv_error   = np.inf
        else:
            # if isinstance(self.distributions, (list, tuple))
            # if hasattr(stats, self.distributions.name):
                # self.distributions = [self.distributions,]
            self.ndim       = distributions.ndim 
            self.num_basis  = self.basis.num_basis
            self.deg = self.basis.deg
            self.active_    = range(self.num_basis) if self.num_basis is not None else None
            self.cv_error   = np.inf
            # ### Now assuming same marginal distributions
            # try:
                # dist_name = self.distributions[0].name 
            # except AttributeError:
                # dist_name = self.distributions[0].dist.name 

            # if dist_name == 'norm':
                # self.basis = museuq.Hermite(d=self.ndim, deg=self.deg)
            # elif dist_name == 'uniform':
                # self.basis = museuq.Legendre(d=self.ndim, deg=self.deg)
            # else:
                # raise ValueError('Polynomial for {} has not been defined yet'.format(distributions[0].name))

    def info():
        print(r'   * {:<25s} : {:<20s}'.format('Surrogate Model Name', self.name))
        if self.deg is not None:
            print(r'     - {:<23s} : {}'.format('Askey-Wiener distributions'   , self.basis.name))
            print(r'     - {:<23s} : {}'.format('Polynomial order (p)', self.deg ))
            print(r'     - {:<23s} : {:d}'.format('No. poly basis (P)', self.basis.num_basis))
            print(r'     - {:<23s} : {:d}'.format('No. active basis (P)', len(self.active_)))

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

        # print(r' > PCE surrogate models with {:s}'.format(self.fit_method))
        # print(r'   * {:<25s} : ndim={:d}, p={:d}'.format('Polynomial', self.ndim, self.deg))
        # print(r'   * {:<25s} : (X, Y, W) = {} x {} x {}'.format('Train data shape', x.shape, y.shape, w.shape))

        # norms = np.sum(X.T**2 * w, -1)
        norms = self.basis.basis_norms *self.basis.basis_norms_const**self.basis.ndim
        coef = np.sum(X.T * y * w, -1) / norms 
        self.model  = self.basis.set_coef(coef) 
        self.active_= range(self.num_basis)
        self.coef   = coef

    def fit_ols(self,x,y,w=None, *args, **kwargs):
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

        n_splits= kwargs.get('n_splits', X.shape[0])
        kf      = model_selection.KFold(n_splits=n_splits,shuffle=True)

        # print(r' > PCE surrogate models with {:s}'.format(self.fit_method))
        # print(r'   * {:<25s} : ndim={:d}, p={:d}'.format('Polynomial', self.ndim, self.deg))
        # print(r'   * {:<25s} : X = {}, Y = {}'.format('Train data shape', X.shape, y.shape))
        
        ## calculate k-folder cross-validation error
        model   = linear_model.LinearRegression(fit_intercept=False)
        neg_mse = model_selection.cross_val_score(model, X, y, scoring = 'neg_mean_squared_error', cv=kf, n_jobs=mp.cpu_count())
        ## fit the model with all data 
        ## X has a column with ones, and want return coefficients to incldue that column
        model.fit(X, y, sample_weight=w)
        self.cv_error= -np.mean(neg_mse)
        self.model   = model 
        self.active_ = range(self.num_basis)
        self.coef    = model.coef_

    def fit_olslars(self,x,y,w=None, *args, **kwargs):
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
        kf      = model_selection.KFold(n_splits=n_splits,shuffle=True)

        # print(r' > PCE surrogate models with {:s}'.format(self.fit_method))
        # print(r'   * {:<25s} : ndim={:d}, p={:d}'.format('Polynomial', self.ndim, self.deg))
        # print(r'   * {:<25s} : X = {}, Y = {}'.format('Train data shape', X.shape, y.shape))
        ### 1. Perform variable selection first
        model_lars       = linear_model.Lars(fit_intercept=False).fit(X,y)
        self.active_lars = model_lars.active_ ## Indices of active basis at the end of the path.
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
                self.active_  = active_indices
                self.cv_error = error_loo
                self.coef     = model.coef_
        # print(r'   * {:<25s} : {} ->#:{:d}'.format('Active basis', self.active_, len(self.active_)))

    def fit_lassolars(self,x,y,w=None, *args, **kwargs):
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
        y = np.array(y, copy=False, ndmin=2)
        X = self.basis.vandermonde(x)
        y = np.squeeze(y)
        if w is not None:
            X = (X.T * w).T
            y = y *w
        ## parameters for LassoLars 
        n_splits= kwargs.get('n_splits', X.shape[0])
        max_iter= kwargs.get('max_iter', 500)
        kf      = model_selection.KFold(n_splits=n_splits,shuffle=True)

        # print(r' > PCE surrogate models with {:s}'.format(self.fit_method))
        # print(r'   * {:<25s} : ndim={:d}, p={:d}'.format('Polynomial', self.ndim, self.deg))
        # print(r'   * {:<25s} : X = {}, Y = {}'.format('Train data shape', X.shape, y.shape))

        model         = linear_model.LassoLarsCV(max_iter=max_iter,cv=kf, n_jobs=mp.cpu_count(),fit_intercept=False).fit(X,y)
        self.active_  = list(*np.nonzero(model.coef_))
        self.model    = model 
        self.cv_error = np.min(np.mean(model.mse_path_, axis=1))
        self.coef     = model.coef_
        # print(r'   * {:<25s} : {} ->#:{:d}'.format('Active basis', self.active_, len(self.active_)))


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
        elif self.fit_method in ['OLS','OLSLARS','LASSOLARS']:
            w = kwargs.get('w', None)
            X = self.basis.vandermonde(x)
            X = (X.T * w).T if w else X
            y = self.model.predict(X)
            y = y/w if w else y

        # print(r'   * {:<25s} : {}'.format('Prediction output', y.shape))
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



