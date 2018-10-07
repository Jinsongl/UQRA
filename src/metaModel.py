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
# import utility.dataIO as dataIO

class metaModel(object):
    """
    Meta model class object 
    General options:
        model_class: 
            surrogate model classes to be used, e.g. PCE, aPCE or Gaussian Process (GP) etc 
        meta_orders:
            Accept only one number for now. ///ndarray or list (something iterable), the highest order of plolynomal terms ???
        doe_method: 
            For PCE model, based on experimental design methods, proper projection or regression fitting will be implemented
            For aPCE model, 
            For GP model,

        dist: list, marginal distributions of underlying random variables from selected Wiener-Askey polynomial, mutually independent
        orthPoly, norms: orthogonal polynomial basis and their corresponding normalization constant. E[Phi_i, Phi_i] = nomrs[i]
    """
    def __init__(self, model_class, meta_orders, doe_method, dist):
        self.model_class     = model_class
        if isinstance(meta_orders, int):
            self.meta_orders = [meta_orders,]
        else:
            self.meta_orders = meta_orders
        self.dist       = dist
        self.distJ      = cp.J(*dist)
        self.orthPoly   = []
        self.norms      = [] 
        self.fit_l2error= []
        self.cv_l2error = []
        self.f_hats     = []

    def get_orth_poly(self):
        for p_order in self.meta_orders:
            poly, norm = cp.orth_ttr(p_order, self.distJ, retall=True)
            self.orthPoly.append(poly)
            self.norms.append(norm)


    def fit_model(self,x,y,w=None):
        """
        Fit specified meta model with given observations (x,y, w for quadrature)

        x: array-like of shape(ndim, nsamples)
        y: array-like of shape(nsamples,ndim), each row is a full observation(e.g. time serie)
            easy to implement for [foo(i) for i in x]
        w: array-like of shape(nsamples), weight for quadrature 
        """
        x = np.asfarray(x)
        y = np.asfarray(y)
        w = np.asfarray(w) if not w else w
        y = y.T if y.shape[1] == x.shape[1] else y

        if self.model_class.upper() == 'PCE':
           f_hats, l2_ers = self.__build_pce_model(x,y,w) 
        elif self.model_class.upper() == "GP":
            print("Not implemented yet")
        elif self.model_class.upper() == 'APCE':
            print("Not implemented yet")
        else:
            pass

        self.f_hats = f_hats
        self.fit_l2error = l2_ers

    def cross_validate(self, x, y):
        """
        Cross validation of the fitted metaModel with given data
        """
        if not self.f_hats:
            raise ValueError('No meta model exists')
        f_cv = []
        for i, f in enumerate(self.f_hats):
            print('\t\tCross validation {:s} metamodel of order \
                    {:d}'.format(self.model_class, max([sum(order) for order in f.keys])))
            f_fit = [f(*val) for val in x.T]
            f_cv.append(f_fit)
            cvError = np.sqrt(np.mean((np.asfarray(f_fit) - y)**2),axis=0)
            self.cv_l2error.append(cvError)
        return np.asfarray(f_cv)



    def predict(self, sample_size, rule='R', R=1):
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
        # run one model to get the size of return y, could return one value or
        # a time series(response evolve with time)
        f = self.f_hats[0]
        y = f(*rand_samples.T[0])
        if len(y) == 1:
            f_pred = np.empty([R, len(self.f_hats), sample_size])
        else:
            f_pred = np.empty([R, len(self.f_hats), sample_size, len(y)])

        for r in xrange(int(R)):
            for i, f in enumerate(self.f_hats):
                print('\t\tPredicting with {:s} of order {:d} for {:d} samples, \
                        repeated {:d} / {:d} times'.format(self.model_class,  \
                            self.meta_orders[i], sample_size,r, R))
                y = [f(*val) for val in rand_samples.T]
                f_pred[r,i,:] = np.asfarray(y) 
        return f_pred

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
        print('\tBuilding PCE surrogate model')
        f_hats = []
        l2_ers = []
        if not self.orthPoly:
            self.get_orth_poly()
        for i, poly in enumerate(self.orthPoly):
            if self.doe_method.upper() == 'QUAD' or self.doe_method.upper == 'GQ':
                assert w is not None
                assert poly.dim == x.shape[0]
                print('\t\tFitting PCE of order {:d} with quadrature'.format(self.meta_orders[i]))
                f_hat = cp.fit_quadrature(poly, x, w, y, norms=self.norms[i])
            else:
                print('\t\tFitting PCE of order {:d} with regression'.format(self.meta_orders[i]))
                f_hat = cp.fit_regression(poly, x, y)
            f_fit = np.array([f_hat(*val) for val in x.T])
            fit_error = np.sqrt(((f_fit-y)**2).mean(axis=0))
            f_hats.append(f_hat)
            l2_ers.append(fit_error)

        return (f_hats, l2_ers)

    def __build_gp_model(self, x, y):
        print('\tBuilding Gaussian Process surrogate model')
        f_hats = []
        l2_ers = []



        return (f_hats, l2_ers)

    def __build_apce_model(self, x, y):
        print('\tBuilding aPCE surrogate model')
        f_hats = []
        l2_ers = []



        return (f_hats, l2_ers)

