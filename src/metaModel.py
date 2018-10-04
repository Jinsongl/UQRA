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
        metaOrder:
            Accept only one number for now. ///ndarray or list (something iterable), the highest order of plolynomal terms ???
        doe_scheme: 
            For PCE model, based on experimental design methods, proper projection or regression fitting will be implemented
            For aPCE model, 
            For GP model,

        dist: list, marginal distributions of underlying random variables from selected Wiener-Askey polynomial, mutually independent
        orthPoly, norms: orthogonal polynomial basis and their corresponding normalization constant. E[Phi_i, Phi_i] = nomrs[i]
    """
    def __init__(self, model_class, metaOrder, doe_scheme, dist):
        self.model_class     = model_class
        if isinstance(metaOrder, int):
            self.metaOrder = [metaOrder,]
        else:
            self.metaOrder  = metaOrder
        self.dist       = dist
        self.distJ      = cp.J(*dist)
        self.orthPoly   = []
        self.norms      = [] 
        self.fitError   = []
        self.CVError    = []
        self.f_hats     = []

    def getOrthPoly(self):
        for pOrder in self.metaOrder:
            poly, norm = cp.orth_ttr(pOrder, self.distJ, retall=True)
            self.orthPoly.append(poly)
            self.norms.append(norm)


    def fitModel(self,x,y,w=None):
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
            print('\tBuilding PCE surrogate model')
            if not self.orthPoly:
                self.getOrthPoly()
            for i, poly in enumerate(self.orthPoly):
                if self.doe_scheme.upper() == 'QUAD' or self.doe_scheme.upper == 'GQ':
                    assert w is not None
                    assert poly.dim == x.shape[0]
                    print('\t\tFitting PCE of order {:d} with quadrature'.format(self.metaOrder[i]))
                    f_hat = cp.fit_quadrature(poly, x, w, y, norms=self.norms[i])
                else:
                    print('\t\tFitting PCE of order {:d} with regression'.format(self.metaOrder[i]))
                    f_hat = cp.fit_regression(poly, x, y)
                f_fit = np.array([f_hat(*val) for val in x.T])
                fitError = np.sqrt(((f_fit-y)**2).mean(axis=0))
                self.f_hats.append(f_hat)
                self.fitError.append(fitError)
        elif self.model_class.upper() == "GP":
            print("Not implemented yet")
        elif self.model_class.upper() == 'APCE':
            print("Not implemented yet")
        else:
            pass


            

    def crossValidate(self, x, y):
        """
        Cross validation of the fitted metaModel with given data
        """
        if not self.f_hats:
            raise ValueError('No meta model exists')
        f_cv = []
        for i, f in enumerate(self.f_hats):
            print('\t\tCross validation {:s} metamodel of order {:d}'.format(self.model_class, max([sum(order) for order in f.keys])))
            f_fit = [f(*val) for val in x.T]
            f_cv.append(f_fit)
            cvError = np.sqrt(np.mean((np.asfarray(f_fit) - y)**2),axis=0)
            self.CVError.append(cvError)
        return np.asfarray(f_cv)



    def predict(self, sample_size, rule='R', R=1):
        """
        predict using metaModel
        Noting returned. Results are saved in csv files. One seperate file for each repeat.
            [zeta0, ..., zetaD, predicted value by f_hat]
        """
        if not self.f_hats:
            raise ValueError('No meta model exists')

        # simParams.updateFilename(simParams.scheme + 'Pred')


        rand_samples = self.distJ.sample(sample_size,rule=rule)
        f = self.f_hats[0]
        y = f(*rand_samples.T[0])
        if len(y) == 1:
            f_pred = np.empty([R, len(self.f_hats), sample_size])
        else:
            f_pred = np.empty([R, len(self.f_hats), sample_size, len(y)])

        for r in xrange(int(R)):
            for i, f in enumerate(self.f_hats):
                print('\t\tPredicting with {:s} of order {:d} for {:d} samples, repeated {:d} / {:d} times'.format(self.model_class, self.metaOrder[i], sample_size,r, R))
                # filenames = dataIO.setfilename(simParams)
                # for j in xrange(int(sample_size)):
                    # if j % int(sample_size/10) == 0:
                        # print "     > generating:   " + '{:2.1E}'.format(j)
                y = [f(*val) for val in rand_samples.T]
                f_pred[r,i,:] = np.asfarray(y) 
                # f_pred.append([ float(x) for x in arg] + [float(y),])
                # dataIO.saveData(np.array(f_pred), filenames[0], simParams.outdirName)
        return f_pred
        # return np.array(f_pred)


    # def predict(self,sample_size, R=1):
        # """
        # predict using metaModel
        # Return:[N,D] array
            # N: sample_size * R * numf_hats. e.g. sample_size=10,R =2, numf_hats=2. Then first 20 corresponds to the first f_hat, last 20 to the second.
            # D: [zeta0, ..., zetaD, predicted value by f_hat]
        # """
        # start = time.time()
        # if not self.f_hats:
            # raise ValueError('No meta model exists')

        # def iter_func():
            # for i, f in enumerate(self.f_hats):
                # print '... Predicting ', self.scheme, ' meta model of order:', self.metaOrder[i]
                # for j in xrange(int(sample_size*R)):
                    # if j % int(sample_size/10) == 0:
                        # print "     > generating:   " + '{:2.1E}'.format(j)
                    # arg = self.distJ.sample(1,rule='R')
                    # y  = f(*arg)
                    # for x in arg:
                        # yield float(x)
                    # # f_pred.append([ float(x) for x in arg] + [float(y),])
        # data = np.fromiter(iter_func(),dtype=float)
        # data = data.reshape((-1,int(R)))
        # end = time.time()
        # print (end - start)
        # Austin Hibbetts# return data 
