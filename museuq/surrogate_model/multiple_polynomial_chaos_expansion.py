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
from museuq.surrogate_model.base import SurrogateModel
from tqdm import tqdm
import museuq.utilities.helpers as uqhelpers 

class PolynomialChaosExpansion(SurrogateModel):
    """
    Class to build multiple polynomial chaos expansion (mPCE) model
    """

    def __init__(self, basis_orders, dist_zeta_J, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dist_zeta_J = dist_zeta_J
        self.multiplicity= kwargs.get('m',1)

        #### make sure basis_orders is a list
        if np.isscalar(basis_orders):
            self.basis_orders = [int(basis_orders),]
        else:
            self.basis_orders= list(int(x) for x in basis_orders)

        #### orhtogonal polynomial basis dist_zeta_J for given orders
        self.basis          = []
        self.orthpoly_norms = []
        for p in self.basis_orders:
            poly, norm = cp.orth_ttr(p, self.dist_zeta_J, retall=True)
            self.basis.append(poly)
            self.orthpoly_norms.append(norm)

        #### collections of metamodels
        self.metamodels   = []
        self.basis_coeffs = [] ## coefficients in terms of each orthogonal polynomial basis, Phi1, Phi2, ...
        self.poly_coeffs  = [] ## coefficients in terms of polynomial functions. x, x^2, x^3...

        # __________________________________________________________________________________________________________________
        #           |                           Parameters
        # metamodel |-------------------------------------------------------------------------------------------------------
        #           |           required            |            optional 
        # ------------------------------------------------------------------------------------------------------------------
        # PCE       | dist_zeta, fit_method         | dist_zeta_J, dist_x, dist_x_J, basis_orders
        # __________________________________________________________________________________________________________________
        # __________________________________________________________________________________________________________________
        # For PCE model, following parameters are required:
        print(r'------------------------------------------------------------')
        print(r'>>> Initialize SurrogateModel Object...')
        print(r'------------------------------------------------------------')
        print(r' > Surrogate model properties')
        print(r'   * {:<17s} : {:<20s}'.format('Model name', 'Multiple Polynomial Chaos Expansion'))
        # Following parameters are required
        print(r'   * Requried parameters')
        # print(self.dist_zeta_J)
        # print(self.basis_orders)
        # print(len(self.basis[0]))
        try:
            print(r'     - {:<25s} : {}'.format('Zeta Joint dist'         , self.dist_zeta_J  ))
            print(r'     - {:<25s} : {}'.format('Basis orders (p)'        , self.basis_orders ))
            print(r'     - {:<25s} : {:d}'.format('Total order basis (P)'   , len(self.basis[0])))
        except KeyError:
            print(r'     Error: dist_zeta, basis_orders are required parameters for PCE model')

        # Following parameters are optional
        # self.dist_zeta_marginal = kwargs.get('dist_zeta_marginal'   , None)
        # self.dist_x             = kwargs.get('dist_x'               , None)
        # self.dist_x_J           = kwargs.get('dist_x_J'             , None)
        print(r'   * Optional parameters:')
        for key, value in kwargs.items():
            if key in ['dist_zeta_J','basis_orders']:
                pass
            else:
                print(r'     - {:<20s} : {}'.format(key,value))
                
    def fit(self,x,y,w=None,fit_method='GLK'):
        """
        Fit PCE meta model with given observations (x,y,w)

        Arguments:
            x: array-like of shape (ndim, nsamples) or (nsamples,) 
                sample input values in zeta (selected Wiener-Askey distribution) space
            y: array-like of shape (nsamples, m)
                QoI observations
                build one PCE model for each column of y

            optional arguments:
            methods: str
                OLS: ordinary least square
                WLS: weighted least square
                GLK: Galerkin projection
            w: array-like weights
                OLS: None
                WLS: weight matrix of shape (nsamples, )
                GLK: quadrature points weights of shape (nsamples, )
        """
        x = np.array(x)
        y = np.array(y)
        y = y.reshape(-1,1) if y.ndim ==1 else y ## reshape y if y is in (n,) format

        print(r' > Fitting PCE surrogate models with {:s}'.format(fit_method))
        print(r'   * {:<17s} : {}'.format('PCE model order', self.basis_orders))

        for iporder, iorthpoly_basis in zip(self.basis_orders, self.basis):
            print(r'   * {:<17s} : {:d}'.format('Polynomial basis order: ', iporder))
            if fit_method.upper() in ['GALERKIN', 'GLK']:
                print(r'   * {:<17s} : (X, Y, W) = {} x {} x {}'.format('Train data shape', x.shape, y.shape, w.shape))
            elif fit_method.upper() in ['OLS']:
                print(r'   * {:<17s} : X = ({},{}), Y = {}'.format('Train data shape', x.shape[1], len(iorthpoly_basis), y.shape))
            elif fit_method.upper() in ['WLS']:
                W = np.diag(np.squeeze(w))
                print(r'   * {:<17s} : X = ({},{}), Y = {}, W={}'.format('Train data shape', x.shape[1], len(iorthpoly_basis), y.shape, W.shape ))
            else:
                raise ValueError('Method to calculate PCE coefficients {:s} is not defined'.format(fit_method))

            # f_hat_iporder           = []
            # orthpoly_coeffs_iporder = []
            # f_hat_coeffs_iporder    = []
            
            for iy in tqdm(np.array(y).T, ascii=True, desc="   - ".format(iporder)):
                uqhelpers.blockPrint()
                f_hat, orthpoly_coeffs = self._single_fit(x,iy,w,fit_method=fit_method)
                uqhelpers.enablePrint()

                # f_hat_iporder.append(f_hat)
                # orthpoly_coeffs_iporder.append(orthpoly_coeffs)
                # f_hat_coeffs_iporder.append(f_hat.coefficients)

                self.metamodels.append(f_hat)
                self.basis_coeffs.append(orthpoly_coeffs)
                self.poly_coeffs.append(f_hat.coefficients)

    def _single_fit(self,x,y,w=None,fit_method='GLK'):
        """
        Fit PCE meta model with given observations (x,y,w)

        Arguments:
            x: array-like of shape (ndim, nsamples) or (nsamples,) 
                sample input values in zeta (selected Wiener-Askey distribution) space
            y: array-like of shape (nsamples,)
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
        """

        if fit_method.upper() in ['GALERKIN', 'GLK']:
            if w is None:
                raise ValueError("Quadrature weights are needed for Galerkin method")
            if iorthpoly_basis.dim != x.shape[0]:
                raise ValueError("Polynomial base functions and variables must have same dimensions, but have Poly.ndim={} and x.ndim={}".format(iorthpoly_basis.dim, x.shape[0]))
            w = np.squeeze(w)
            f_hat, orthpoly_coeffs = cp.fit_quadrature(iorthpoly_basis, x, w, y, retall=True)
            return f_hat, orthpoly_coeffs
        elif fit_method.upper() in ['OLS']:
            f_hat, orthpoly_coeffs= cp.fit_regression(iorthpoly_basis, x, y, retall=True)
            return f_hat, orthpoly_coeffs
        elif fit_method.upper() in ['WLS']:
            W = np.diag(np.squeeze(w))
            f_hat, orthpoly_coeffs= cp.fit_regression(iorthpoly_basis, np.dot(W, x.T).T, np.dot(W, y), retall=True)
            return f_hat, orthpoly_coeffs

        else:
            raise ValueError('Method to calculate PCE coefficients {:s} is not defined'.format(fit_method))

    def predict(self,X, y_true=None, **kwargs):
        """
        Predict using surrogate models 

        Arguments:	
        X : array-like, shape = (n_features/ndim, nsamples)
            Query points where the surrogate model are evaluated

        Returns:	
        y : list of array, array shape = (nsamples, )
            predicted value from surrogate models at query points
        """
        if not self.metamodels:
            raise ValueError('No surrogate models exist, need to fit first')
        y_pred = []
        y_pred_all_metamodels = []

        print(r' > Predicting with PCE surrogate models... ')
        ## See explainations about Poly above
        for i, imetamodel in enumerate(self.metamodels):
            iy_pred = imetamodel(*X)
            print(r'   * {:<17s} : {:d}/{:d}    -> Output: {}'.format('Surrogate model (PCE)', i, len(self.metamodels), iy_pred.shape))
            y_pred_all_metamodels.append(iy_pred)

        if self.multiplicity == 1:
            y_pred = y_pred_all_metamodels
            if y_true is not None:
                scores = self.score(y_pred, y_true, **kwargs)
                if len(y_pred) == 1:
                    y_pred = y_pred[0]
                return y_pred, scores
            else:
                if len(y_pred) == 1:
                    y_pred = y_pred[0]
                return y_pred
        else:
            raise NotImplementedError 

            # assert len(self.metamodels) == self.multiplicity * len(self.basis_orders)

            # y_pred_all_metamodels = np.array(y_pred_all_metamodels)
            # idx = np.random.randint(0, y_pred_all_metamodels.shape[0],size=y_pred_all_metamodels.shape[1])
            # y_pred = np.choose(idx, y_pred_all_metamodels)


