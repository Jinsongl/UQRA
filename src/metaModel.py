#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import chaospy as cp
import numpy as np
import sklearn.gaussian_process as skgp

# import utility.dataIO as dataIO

class metaModel(object):
    """
    Meta model class object 
    General options:
    """
    def __init__(self, model_class, meta_orders, fit_method, dist_zeta_list):
        """
        model_class: 
            surrogate model classes to be used, e.g. PCE, aPCE or Gaussian Process (GP) etc 
        meta_orders:
            Accept only one number for now. ///ndarray or list (something iterable), the highest order of plolynomal terms ???
        fit_method: 
            For PCE model, based on experimental design methods, proper projection or regression fitting will be implemented
            For aPCE model, 
            For GP model,

        dist_zeta_list: list, marginal distributions of underlying random variables from selected Wiener-Askey polynomial, mutually independent
        orthPoly, norms: orthogonal polynomial basis and their corresponding normalization constant. E[Phi_i, Phi_i] = nomrs[i]
        
        """
        self.model_class= model_class
        self.fit_method = fit_method
        self.meta_orders = []
        if not isinstance(meta_orders, list):
            self.meta_orders.append(int(meta_orders))
        else:
            self.meta_orders = list(int(x) for x in meta_orders)
            # self.meta_orders.append(int(x) for x in meta_orders)

        self.dist_zeta_list   = dist_zeta_list
        self.distJ      = dist_zeta_list if len(dist_zeta_list) == 1 else cp.J(*dist_zeta_list)
        self.orthPoly   = []
        self.norms      = [] 
        self.fit_l2error= []
        self.cv_l2errors= []
        self.f_hats     = []

    def get_orth_poly(self):
        for p_order in self.meta_orders:
            poly, norm = cp.orth_ttr(p_order, self.distJ, retall=True)
            self.orthPoly.append(poly)
            self.norms.append(norm)


    def fit_model(self,x,y,w=None):
        """
        Fit specified meta model with given observations (x,y, w for quadrature)


        x: list of length len(doe_order), number of sample sets
            each element contains input sample variables both in zeta and physical space of shape (ndim, nsamples), each column includes [zeta_0,...zeta_n]        
        y: list of length len(doe_order), number of observation sets
            each element contains (nsamples,)
        w: array-like of shape(nsamples), weight for quadrature 
        """
        print('************************************************************')
        print('Building {:d} models with {:d} sample sets and meta model orders {}'.format(len(self.meta_orders) * len(x), len(x), [x for x in self.meta_orders]))

        for i in range(len(x)):
            print('>>> Building surrogate model with sample sets: {:d}'.format(i+1))
            datax = np.array(x[i])
            datay = np.array(y[i])
            dataw = np.array(w[i]) if w else w
                
            # dataw = np.array(w[i]) if not w else w
            # datay = datay.T if datay.shape[1] == datax.shape[1] else datay

            if self.model_class.upper() == 'PCE':
               f_hat, l2_er = self.__build_pce_model(datax,datay,dataw) 
            elif self.model_class.upper() == "GP":
                print("Not implemented yet")
            elif self.model_class.upper() == 'APCE':
                print("Not implemented yet")
            else:
                pass

            self.f_hats.append(f_hat)
            self.fit_l2error.append(l2_er)

    def cross_validate(self, x, y):
        """
        Cross validation of the fitted metaModel with given data
        """
        if not self.f_hats:
            raise ValueError('No meta model exists')
        for _, f_hats in enumerate(self.f_hats):
            _cv_l2error = []
            print('\t\tCross validation {:s} metamodel of order \
                    {:d}'.format(self.model_class, max([sum(order) for order in f.keys])))
            for _, f in enumerate(f_hats):
                f_fit = [f(*val) for val in x.T]
                f_cv.append(f_fit)
                cv_l2error = np.sqrt(np.mean((np.asfarray(f_fit) - y)**2),axis=0)
                _cv_l2error.append(cv_l2error)
            self.cv_l2errors.append(_cv_l2error)



    def predict(self, sample_size, models_chosen=None, rule='R', R=1):
        """
        predict using metaModel

        return:
            predict value of shape(nRepeat, nMetaModels, nsample for one repeat[, predicted value] )
        Noting returned. Results are saved in csv files. One seperate file for each repeat.
            [zeta0, ..., zetaD, predicted value by f_hat]
        """
        if not self.f_hats:
            raise ValueError('No meta model exists')

        # sampling from joint distribution
        rand_samples = self.distJ.sample(sample_size,rule=rule)

        if models_chosen is not None:
            f_hats = [self.f_hats[i][j] for i, j in models_chosen]
        else:
            f_hats = [f for sublist in self.f_hats for f in sublist]

        # run one model to get the size of return y, could return one value or
        # a time series(response evolve with time)
        # f = self.f_hats[0][0]
        # y = f(*rand_samples.T[0])
        # if isinstance(y, float):
            # print('here')
            # f_pred = np.empty([R, len(self.f_hats), sample_size])
        # else:
            # f_pred = np.empty([R, len(self.f_hats), sample_size, len(y)])
        print('************************************************************')
        print('Predicting with {:d} {:s} surrogate models with {:d} samples'.format(len(f_hats), self.model_class, int(sample_size)))
        f_pred = []
        for i, f in enumerate(f_hats):
            print('>>> Predicting with {:}th order {:s} surrogate model'.format(self.meta_orders[i%len(self.f_hats[0])], self.model_class))
            _f_pred = []
            for r in range(int(R)):
                print('\r\tRepeated: {:d} / {:d}'.format(r+1, R),end='')
                y = [f(*val) for val in rand_samples.T]
                _f_pred.append(y)
            print('\n')
            f_pred.append(_f_pred)
        return np.array(f_pred)

    def __build_pce_model(self, x, y, w=None):
        """
        Build Polynomial chaos expansion surrogate model with x, y, optional w

        Arguments:
            x: sample input values in zeta (selected Wiener-Askey distribution) space
                array-like of shape(ndim, nsamples)
            y: array-like of shape(nsamples,ndim), each row is a full observation(e.g. time serie)
                easy to implement for [foo(i) for i in x]

            w: array-like of shape(nsamples), weight for quadrature 

        Returns:
            f_hats: list of surrogate models
            l2_ers: list of l2 error between observed Y responses and fitted value at input samples
        """
        f_hats = []
        l2_ers = []
        if not self.orthPoly:
            self.get_orth_poly()
        for i, poly in enumerate(self.orthPoly):
            if self.fit_method.upper() == 'SP' or self.fit_method.upper == 'GQ':
                assert w is not None
                assert poly.dim == x.shape[0]
                print('\tBuilding PCE surrogate model of order {:d} with quadrature'.format(self.meta_orders[i]))
                f_hat = cp.fit_quadrature(poly, x, w, y, norms=self.norms[i])
            elif self.fit_method.upper() == 'RG':
                print('\tBuilding PCE surrogate model of order {:d} with regression'.format(self.meta_orders[i]))
                f_hat = cp.fit_regression(poly, x, y)
            else:
                raise ValueError('fit method {:s} is not defined'.format(self.fit_method))
            f_fit = np.array([f_hat(*val) for val in x.T])
            fit_error = np.sqrt(((f_fit-y)**2).mean(axis=0))
            f_hats.append(f_hat)
            l2_ers.append(fit_error)

        return (f_hats, l2_ers)

    def __build_gp_model(self, x, y, kernel=None, alpha=1e-10, optimizer=’fmin_l_bfgs_b’, n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None):
        """

        Kernel functions:
            ConstantKernel, WhiteKernel, RBF, Matern, 
        """
        print('\tBuilding Gaussian Process Regression(Kriging) surrogate model')
        f_hats = []
        l2_ers = []


        return (f_hats, l2_ers)

    def __build_apce_model(self, x, y):
        print('\tBuilding aPCE surrogate model')
        f_hats = []
        l2_ers = []



        return (f_hats, l2_ers)

