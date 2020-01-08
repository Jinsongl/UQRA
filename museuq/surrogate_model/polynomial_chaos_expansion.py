#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import numpy as np, chaospy as cp
from sklearn import linear_model
from museuq.surrogate_model.base import SurrogateModel
import museuq.utilities.metrics_collections as uq_metrics 

class PolynomialChaosExpansion(SurrogateModel):
    """
    Class to build polynomial chaos expansion (PCE) model
    """

    def __init__(self, poly_order, dist_zeta_J, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_zeta_J    = dist_zeta_J
        self.poly_order     = int(poly_order)#### basis_order is just one values (int) 

        poly, norm          = cp.orth_ttr(self.poly_order, self.dist_zeta_J, retall=True)
        self.basis_         = poly
        self.basis_norms_   = norm
        self.active_        = [] ## Indices of active variables (used for sparse model).

        # __________________________________________________________________________________________________________________
        #           |                           Parameters
        # metamodel |-------------------------------------------------------------------------------------------------------
        #           |           required            |            optional 
        # ------------------------------------------------------------------------------------------------------------------
        # PCE       | dist_zeta, method         | dist_zeta_J, dist_x, dist_x_J, poly_order
        # __________________________________________________________________________________________________________________
        # __________________________________________________________________________________________________________________
        # For PCE model, following parameters are required:
        # print(r'------------------------------------------------------------')
        # print(r'>>> Initialize SurrogateModel Object...')
        # print(r'------------------------------------------------------------')
        print(r'>>> Building Surrogate Models')
        print(r'   * {:<25s} : {:<20s}'.format('Model name', 'Polynomial Chaos Expansion'))
        # Following parameters are required
        print(r'   * Requried parameters')
        try:
            print(r'     - {:<23s} : {}'.format('Zeta Joint dist'       , self.dist_zeta_J))
            print(r'     - {:<23s} : {}'.format('Polynomial order (p)'  , self.poly_order ))
            print(r'     - {:<23s} : {:d}'.format('No. poly basis (P)'  , len(self.basis_)))
        except KeyError:
            print(r'     Error: dist_zeta, poly_order are required parameters for PCE model')

        # Following parameters are optional
        print(r'   * Optional parameters:')
        for key, value in kwargs.items():
            if key in ['dist_zeta_J','poly_order']:
                pass
            else:
                print(r'     - {:<20s} : {}'.format(key,value))
                
    def fit(self,x,y,w=None, *args, **kwargs):
        """
        Fit PCE meta model with given observations (x,y,w)

        Arguments:
            x: array-like of shape (ndim, nsamples) 
                sample input values in zeta (selected Wiener-Askey distribution) space
            y: array-like of shape (nsamples,[n_output_dims])
                QoI observations

            optional arguments:
            methods: str
                OLS: ordinary least square
                WLS: weighted least square
                GLK: Galerkin projection
            w: array-like weights
                OLS: None
                WLS: weight matrix of shape (nsamples, )
                GLK: quadrature points weights of shape (nsamples, )

        Returns:

            self.metamodels   : chaospy.Poly 
            self.coeffs_basis_: coefficients in terms of each orthogonal polynomial basis, Phi1, Phi2, ...
            self.poly_coeffs  : coefficients in terms of polynomial functions. x, x^2, x^3...
        """
        x = np.array(x)
        x = x.reshape(1,-1) if x.ndim == 1 else x
        y = np.array(y)
        y = y.reshape(-1,1) if y.ndim == 1 else y


        self.fit_method = kwargs.get('method', 'GLK')
        print(r' > Fitting PCE surrogate models with {:s}'.format(self.fit_method))
        print(r'   * {:<25s} : {}'.format('PCE polynomial order', self.poly_order))

        if self.fit_method.upper() in ['GALERKIN', 'GLK','PROJECTION']:
            if w is None:
                raise ValueError("Quadrature weights are needed for Galerkin method")
            w = np.squeeze(w)
            y = np.squeeze(y)
            print(r'   * {:<25s} : (X, Y, W) = {} x {} x {}'.format('Train data shape', x.shape, y.shape, w.shape))
            f_hat, orthpoly_coeffs = cp.fit_quadrature(self.basis_, x, w, y, retall=True)
            self.metamodels   = f_hat
            self.coeffs_basis_= orthpoly_coeffs
            self.active_      = range(len(self.basis_))

        elif self.fit_method.upper() in ['OLS']:
            print(r'   * {:<25s} : X = ({},{}), Y = {}'.format('Train data shape', x.shape[1], len(self.basis_), y.shape))
            f_hat, orthpoly_coeffs = cp.fit_regression(self.basis_, x, y, retall=True)
            self.metamodels   = f_hat
            self.coeffs_basis_= orthpoly_coeffs
            self.active_      = range(len(self.basis_))

        elif self.fit_method.upper() in ['WLS']:
            W = np.diag(np.squeeze(w))
            print(r'   * {:<25s} : X = ({},{}), Y = {}, W={}'.format('Train data shape', x.shape[1], len(self.basis_), y.shape, W.shape ))
            f_hat, orthpoly_coeffs= cp.fit_regression(self.basis_, np.dot(W, x.T).T, np.dot(W, y), retall=True)
            self.metamodels   = f_hat
            self.coeffs_basis_= orthpoly_coeffs
            self.active_      = range(len(self.basis_))

        elif self.fit_method.upper() in ['LASSOLARS']:
            ## parameters for LassoLarsCV
            max_iter= kwargs.get('max_iter', 500)
            X = self.basis_(*x)
            y = np.squeeze(y)
            print(r'   * {:<25s} : X = {}, Y = {}'.format('Train data shape', tuple(reversed(X.shape)), y.shape))
            model = linear_model.LassoLarsCV(max_iter=max_iter, cv=X.shape[1]).fit(X.T,y)
            self.metamodels     = model 
            self.coeffs_basis_  = model.coef_[np.nonzero(model.coef_)] 
            self.active_        = [0,] + list(*np.nonzero(model.coef_))
            print(r'   * {:<25s} : {}'.format('Active basis', self.active_))
            # f_hat = cp.polynomial(self.basis_[np.nonzero(model.coef_)]*active_coefs).sum() + model.intercept_


        elif self.fit_method.upper() in ['OLSLARS']:
            """
            Methods in 
                Blatman, Géraud, and Bruno Sudret. "Adaptive sparse polynomial chaos expansion based on least angle regression." Journal of Computational Physics 230.6 (2011): 2345-2367.

            """
            ## parameters for LassoLars 
            cv_error= kwargs.get('error', 'leave_one_out_error')
            X = self.basis_(*x)
            y = np.squeeze(y)
            print(r'   * {:<25s} : X = {}, Y = {}'.format('Train data shape', tuple(reversed(X.shape)), y.shape))
            model = linear_model.Lars().fit(X.T,y)
            model_errors  = np.inf
            for i in range(len(model.active_)+1):
                active_indices = np.nonzero(model.coef_)[0] #model.active_[:i+1]
                active_indices = np.array([0, *active_indices])
                active_basis   = cp.polynomial(self.basis_[active_indices[:i+1]])
                print(active_basis)
                f_hat, coeffs  = cp.fit_regression(active_basis, x, y, retall=True)
                X = active_basis(*x).T
                error = uq_metrics.leave_one_out_error(X, y)
                if error < model_errors:
                    self.metamodels     = f_hat
                    self.coeffs_basis_  = coeffs
                    self.active_        = active_indices
            print(r'   * {:<25s} : {}'.format('Active basis', self.active_))

        elif self.fit_method.upper() in ['LARS']:
            raise ValueError('Method to calculate PCE coefficients {:s} is not defined'.format(method))
        else:
            raise ValueError('Method to calculate PCE coefficients {:s} is not defined'.format(method))

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
        if self.fit_method.upper() in ['GALERKIN', 'GLK','PROJECTION','OLS', 'WLS','OLSLARS' ]:
            y_pred = self.metamodels(*x).T

        elif self.fit_method.upper() in ['LASSOLARS']:
            X = self.basis_(*x)
            y_pred = self.metamodels.predict(X.T)
        else:
            raise NotImplementedError

        print(r'   * {:<25s} : {}'.format('Prediction output', y_pred.shape))
        return y_pred


