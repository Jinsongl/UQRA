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
import scipy.stats as scistats
from scipy.stats.kde import gaussian_kde
from statsmodels.distributions.empirical_distribution import ECDF
import warnings
warnings.filterwarnings(action="ignore",  message="^internal gelsd")

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
        orthPoly, orthPoly_norms: orthogonal polynomial basis and their corresponding normalization constant. E[Phi_i, Phi_i] = nomrs[i]
        
        """
        self.model_class    = model_class
        self.fit_method     = fit_method
        self.meta_orders    = []
        if np.isscalar(meta_orders):
            self.meta_orders.append(int(meta_orders))
        else:
            self.meta_orders = list(int(x) for x in meta_orders)
            # self.meta_orders.append(int(x) for x in meta_orders)

        self.dist_zeta_list   = dist_zeta_list
        self.distJ      = dist_zeta_list if len(dist_zeta_list) == 1 else cp.J(*dist_zeta_list)
        self.orthPoly   = []
        self.cv_l2errors= []
        self.f_hats     = []
        self.metric_names = ['value','moments','norms','upper fractile','ECDF','pdf']
        self.metrics    = [[] for _ in range(len(self.metric_names))] ## each element is of shape: [sample sets, metaorders, repeated samples, metric]
        self.f_orth_coeffs = []
        self.orthPoly_norms= [] 

    def get_orth_poly(self):
        for p_order in self.meta_orders:
            poly, norm = cp.orth_ttr(p_order, self.distJ, retall=True)
            self.orthPoly.append(poly)
            self.orthPoly_norms.append(norm)

    def fit_model(self,x,y,w=None):
        """
        Fit specified meta model with given observations (x,y, w for quadrature)


        x: list of length len(doe_order), number of sample sets
            each element contains input sample variables both in zeta and physical space of shape (ndim, nsamples), each column includes [zeta_0,...zeta_n]        
        y: list of length len(doe_order), number of observation sets
            each element contains (nsamples,)
        w: array-like of shape(nsamples), weight for quadrature 
        """
        print('------------------------------------------------------------')
        print('►►► Build Surrogate Models')
        print(' - Number of surrogate models to build: {:d}'.format(len(self.meta_orders)* len(x)))
        print(' - Number of samples sets             : {:d}'.format(len(x)))
        print(' - Type of surrogate models           : {:s}'.format(self.model_class))
        print(' - Method to calculate coefficients   : {:s}'.format(self.fit_method))
        print(' - Orders of surrogate models to build: {}'.format( [x for x in self.meta_orders]))
        # print('Building {:d} models with {:d} sample sets and meta model orders {}'.format(len(self.meta_orders) * len(x), len(x), [x for x in self.meta_orders]))

        for i in range(len(x)):
            print('>>> Sample sets: {:d}/{:d}'.format(i+1, len(x)))
            datax = np.array(x[i])
            datay = np.array(y[i])
            dataw = np.array(w[i]) if w else w
                
            # dataw = np.array(w[i]) if not w else w
            # datay = datay.T if datay.shape[1] == datax.shape[1] else datay

            if self.model_class.upper() == 'PCE':
                # print('\t Model Class: {:s};  method: {:s};  #orders: {:d}'.format(self.model_class, self.fit_method, len(self.meta_orders)))
                # print('\t'+'-'*50)
                self.__build_pce_model(datax,datay,w=dataw) 
            elif self.model_class.upper() == "GP":
                print("Not implemented yet")
            elif self.model_class.upper() == 'APCE':
                print("Not implemented yet")
            else:
                pass

        print('>>> ----  Done (build surrogate models)  ----')

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

    def predict(self, sampling=[1e5,10,'R'], retmetrics=[1,1,1,1,1,1]):
        """
        predict using metaModel
        return:
            predict value of shape(nRepeat, nMetaModels, nsamples for one repeat[, predicted value] )
        Noting returned. Results are saved in csv files. One seperate file for each repeat.
            [zeta0, ..., zetaD, predicted value by f_hat]
        """
        if not self.f_hats:
            raise ValueError('No meta model exists')

        nsamples, nrepeat, rule = sampling 
        nsamples = [nsamples,] * nrepeat

        # if models_chosen is not None:
            # f_hats = [self.f_hats[i][j] for i, j in models_chosen]
        # else:
            # f_hats = [f for sublist in self.f_hats for f in sublist]

        print('************************************************************')
        print('Make prediction with surrogate models...')
        print(' - Number of prediction samples: {:4.2e}'.format(int(nsamples[0])))
        print(' - Number of repeat            : {:d}'.format(nrepeat))
        # print('Predicting with {:d} {:s} surrogate models with {} samples'.format(len(self.f_hats), self.model_class, [x for x in nsamples]))
        self.metric_names = [iname for i, iname in zip(retmetrics, self.metric_names) if i]
        self.metrics = [[] for _ in range(sum(retmetrics))]

        ## f_hats of shape [samples sets, meta_orders]
        for i0, if_hats_sample_set in enumerate(self.f_hats):
            metrics_per_set = [[] for _ in range(sum(retmetrics))]
            for i1, if_hat in enumerate(if_hats_sample_set):
                metrics_per_f = [[] for _ in range(sum(retmetrics))]
                for r, isamples in enumerate(nsamples):
                    print('\r>>> Surrogate model: [{:d}, {:d}]/[{:>2d}, {:>2d}];   repeat: {:d}/{:d}'.\
                            format(i0+1, i1+1, len(self.f_hats),len(self.f_hats[0]), r+1, nrepeat), end='')
                    # print('\r\tRepeated: {:d} / {:d}'.format(r+1, nrepeat),end='')
                    # sampling from joint distribution
                    samples_zeta = self.distJ.sample(isamples,rule=rule)
                    y = if_hat(samples_zeta)
                    # if samples_zeta.ndim == 1:
                        # y = if_hat(samples_zeta)
                        # # y = [if_hat(val)[0] for val in samples_zeta.T]
                    # else:
                        # y = if_hat(samples_zeta)
                        # # y = [if_hat(*val)[0] for val in samples_zeta.T]
                    # print('predict shape: {}'.format(y.shape))
                    if np.any(retmetrics):
                        metrics_all = self.__cal_metrics(y,p=[0.01,0.001],retmetrics=retmetrics)
                        for i, imetric in enumerate(metrics_all):
                            metrics_per_f[i].append(imetric)

                if np.any(retmetrics):
                    for i, imetric in enumerate(metrics_per_f):
                        metrics_per_set[i].append(imetric)

            if np.any(retmetrics):
                for i, imetric in enumerate(metrics_per_set):
                    self.metrics[i].append(imetric)
        
        # for i, f in enumerate(self.f_hats):
            # # print('>>> Predicting with {:}th order {:s} surrogate model'.format(self.meta_orders[i%len(self.f_hats[0])], self.model_class))
            # _metric = [[],[],[],[],[],[]]
            # for r, isamples in enumerate(nsamples):
                # print('\r>>> Surrogate model: {:d}/{:d};   repeat: {:d}/{:d}'.format(i, len(self.f_hats), r+1, nrepeat), end='')
                # # print('\r\tRepeated: {:d} / {:d}'.format(r+1, nrepeat),end='')
                # # sampling from joint distribution
                # samples_zeta = self.distJ.sample(isamples,rule=rule)
                # if samples_zeta.ndim == 1:
                    # y = [f(val) for val in samples_zeta.T]
                # else:
                    # y = [f(*val) for val in samples_zeta.T]
                # if retmetrics:
                    # metrics = self.__cal_metrics(y,p=[0.01,0.001])
                    # for i, imetric in enumerate(metrics):
                        # _metric[i].append(imetric)
            # for i, _ in enumerate(_metric):
                # self.metrics[i].append(_metric[i]) 

# [pce_model.metrics[i][j][1] for i in np.arange(len(samples_train[0])*len(meta_orders) ) for j in np.arange(samples_test[1])]
        print('\n>>> ----  Done (prediction with surrogate models)  ----')

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
            optional:
            f_coeffs: coefficients for orthogo
        """

        if not self.orthPoly:
            self.get_orth_poly()
        complete_order = []
        f_hats = []
        f_orth_coeffs = []
        for i, poly in enumerate(self.orthPoly):
            if self.fit_method.upper() in ['SP','GQ','QUAD']:
                assert w is not None
                assert poly.dim == x.shape[0]
                print('\r\tCompleted meta model (order): {}'.format(complete_order), end='')
                # f_hat, coeffs = cp.fit_quadrature(poly, x, w, y, norms=self.orthPoly_norms[i], retall=True)
                f_hat, coeffs = cp.fit_quadrature(poly, x, w, y, retall=True)
                f_hats.append(f_hat)
                f_orth_coeffs.append(coeffs)
            elif self.fit_method.upper() == 'RG':
                # print('\tBuilding PCE surrogate model of order {:d} with regression'.format(self.meta_orders[i]))
                print('\r    Model order: {:d}/{:d}  '.format(i,len(self.meta_orders)), end='')
                f_hat, coeffs = cp.fit_regression(poly, x, y, retall=True)
                f_hats.append(f_hat)
                f_orth_coeffs.append(coeffs)
            else:
                raise ValueError('fit method {:s} is not defined'.format(self.fit_method))
            complete_order.append(self.meta_orders[i])
        self.f_hats.append(f_hats)
        self.f_orth_coeffs.append(f_orth_coeffs)
        print('\n')

    def __build_gp_model(self, x, y, kernel=None, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None):
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


    def __cal_metrics(self,x,p=[0.01,0.001],retmetrics=[1,1,1,1,1,1]):
        """
        Calculate a set of error metrics used to evaluate the accuracy of the approximation
        Reference: "On the accuracy of the polynomial chaos approximation R.V. Field Jr., M. Grigoriu"
        arguments: 
            x: estimate data of shape(n,)
            y: real data of shape(n,)
        [values, moments, norms, upper tails, ecdf, pdf] 
        -----------------------------------------------------------------------
        0 | values 
        1 | moments, e = [1 - pce_moment(i)/real_moment(i)], for i in range(n)
        2 | norms : numpy.linalg.norm(x, ord=None, axis=None, keepdims=False)
        3 | upper fractile 
        4 | ecdf 
        5 | kernel density estimator pdf
        """
        metrics = []
        self.metric_names=[]
        x = np.squeeze(np.array(x))
        if retmetrics[0]:
            metrics.append(x)
            self.metric_names.append['value']
        if retmetrics[1]:
            metrics.append(self.__error_moms(x))
            self.metric_names.append('moments')
        if retmetrics[2]:
            metrics.append(self.__error_norms(x))
            self.metric_names.append('norms')
        if retmetrics[3]:
            metrics.append(self.__error_tail(x,p=p))
            self.metric_names.append('upper_tail')
        if retmetrics[4]:
            metrics.append(ECDF(x))
            self.metric_names.append('ECDF')
        if retmetrics[5]:
            metrics.append(gaussian_kde(x))
            self.metric_names.append('kde')
        # else:
            # NotImplementedError('Metric not implemented')
        return metrics


    # def __error_values(self,x,y=None,e=1e-3):
        # if y:
            # if np.any(y<e):
                # return x-y
            # else:
                # return (x-y)/y
        # else:
            # return x

    def __error_moms(self,x):
        x = np.squeeze(x)
        assert x.ndim == 1
        xx = np.array([x**i for i in np.arange(7)])
        xx_moms = np.mean(xx, axis=1)
        return xx_moms
        

        # y = np.squeeze(y)
        # yy = np.array([y**i for i in np.arange(7)])
        # yy_moms = np.mean(yy, axis=1)


        # if retvalue:
            # e = np.array([xx_moms, yy_moms])
        # else:
            # e = abs(1 - xx_moms/yy_moms)
        # return e

    def __error_norms(self, x):
        """
        ord	norm for matrices	        norm for vectors
        None	Frobenius norm	                2-norm
        ‘fro’	Frobenius norm	                –
        ‘nuc’	nuclear norm	                –
        inf	max(sum(abs(x), axis=1))	max(abs(x))
        -inf	min(sum(abs(x), axis=1))	min(abs(x))
        0	–	                        sum(x != 0)
        1	max(sum(abs(x), axis=0))	as below
        -1	min(sum(abs(x), axis=0))	as below
        2	2-norm (largest sing. value)	as below
        -2	smallest singular value	i       as below
        other	–	                        sum(abs(x)**ord)**(1./ord)
        """
        x = np.squeeze(x)
        # y = np.squeeze(y)
        assert x.ndim == 1
        e = []

        e.append(np.linalg.norm(x,ord=0))
        e.append(np.linalg.norm(x,ord=1))
        e.append(np.linalg.norm(x,ord=2))
        e.append(np.linalg.norm(x,ord=np.inf))
        e.append(np.linalg.norm(x,ord=-np.inf))
        return e

    def __error_tail(self,x, p=[0.01,]):
        """
        """
        e = []
        for ip in p:
            p_invx = scistats.mstats.mquantiles(x,1-ip)
            # p_invy = scistats.mstats.mquantiles(y,1-ip)
            e.append(p_invx[0])
        return e 



        



        

