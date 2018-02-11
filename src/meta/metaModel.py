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
import utility.dataIO as dataIO

class metaModel(object):
    """
    Meta model class object 
    General options:
        scheme: Spectral projection (projection) or linear regression (regression)
        metaOrder: Accept only one number for now. ///ndarray or list (something iterable), the highest order of plolynomal terms ???
        dist: list, marginal distributions of underlying random variables for selected Wiener-Askey polynomial, mutually independent
        orthPoly, norms: orthogonal polynomial basis and their corresponding normalization constant. E[Phi_i, Phi_i] = nomrs[i]
    """
    def __init__(self,site, scheme, metaOrder, dist):
        self.site       = site
        self.scheme     = scheme
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


    def fitModel(self,data):
        if not self.orthPoly:
            self.getOrthPoly()
        for i, poly in enumerate(self.orthPoly):
            print "... Fitting ", self.scheme," meta model of order: ", self.metaOrder[i]
            if self.scheme == 'QUAD':
                f_hat = cp.fit_quadrature(poly, data.X[:,:-1].T, data.X[:,-1], data.Y, norms=self.norms[i])
                f_fit = np.array([f_hat(*val) for val in data.X[:,:-1]])
            else:
                f_hat = cp.fit_regression(poly, data.X.T, data.Y)
                f_fit = np.array([f_hat(*val) for val in data.X])
            fitError = np.sqrt(((f_fit-data.Y)**2).mean())
            self.f_hats.append(f_hat)
            self.fitError.append(float(fitError))

    def crossValidate(self, data):
        """
        Cross validation of the metaModel with given data
        """
        if not self.f_hats:
            raise ValueError('No meta model exists')
        f_cv = []
        for i, f in enumerate(self.f_hats):
            print '... Cross validation ',self.scheme, ' meta model of order: ', max([sum(order) for order in f.keys])
            f_fit = np.array([f(*val) for val in data.X])
            f_cv.append(f_fit)
            cvError = np.sqrt(np.mean((f_fit - data.Y)**2))
            self.CVError.append(float(cvError))
        return np.array(f_cv).T
    def predict(self,sampSize,simParams, R=1):
        """
        predict using metaModel
        Noting returned. Results are saved in csv files. One seperate file for each repeat.
            [zeta0, ..., zetaD, predicted value by f_hat]
        """
        if not self.f_hats:
            raise ValueError('No meta model exists')

        simParams.updateFilename(simParams.scheme + 'Pred')
        for i, f in enumerate(self.f_hats):
            print '... Predicting ', self.scheme, ' meta model of order:', self.metaOrder[i]
            for r in xrange(int(R)):
                print "     > Repeat: " + '{:2d}'.format(r) + ' Out of ' + '{:2d}'.format(R) 
                f_pred = []
                filenames = dataIO.setfilename(simParams)
                for j in xrange(int(sampSize)):
                    if j % int(sampSize/10) == 0:
                        print "     > generating:   " + '{:2.1E}'.format(j)
                    arg = self.distJ.sample(1,rule='R')
                    y  = f(*arg)
                    # v = [ float(x) for x in arg] + [float(y),]
                    f_pred.append([ float(x) for x in arg] + [float(y),])
                dataIO.saveData(np.array(f_pred), filenames[0], simParams.outdirName)

        # return np.array(f_pred)


    # def predict(self,sampSize, R=1):
        # """
        # predict using metaModel
        # Return:[N,D] array
            # N: sampSize * R * numf_hats. e.g. sampSize=10,R =2, numf_hats=2. Then first 20 corresponds to the first f_hat, last 20 to the second.
            # D: [zeta0, ..., zetaD, predicted value by f_hat]
        # """
        # start = time.time()
        # if not self.f_hats:
            # raise ValueError('No meta model exists')

        # def iter_func():
            # for i, f in enumerate(self.f_hats):
                # print '... Predicting ', self.scheme, ' meta model of order:', self.metaOrder[i]
                # for j in xrange(int(sampSize*R)):
                    # if j % int(sampSize/10) == 0:
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
        # return data 
