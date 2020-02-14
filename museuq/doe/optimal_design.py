#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
from museuq.doe.base import ExperimentalDesign
from museuq.utilities.decorators import random_state
from museuq.utilities.helpers import num2print
import numpy as np, scipy as sp
import numpy.linalg as LA
import copy
import itertools
from tqdm import tqdm
import time
import math

class OptimalDesign(ExperimentalDesign):
    """ Quasi-Optimal Experimental Design and Optimal Design"""

    def __init__(self, opt_criteria, random_seed=None):
        """
        Optimal/Quasi Optimal Experimental design:
        Arguments:
        n: int, number of samples 
        opt_criteria: optimal design criteria
        """
        super().__init__(random_seed=random_seed)
        self.criteria   = opt_criteria 
        self.filename   = '_'.join(['DoE', 'Opt'+self.criteria.capitalize()])
        self.selected_row = None 

    def __str__(self):
        return('Optimal Criteria: {:<15s}, num. samples: {:d} '.format(self.criteria, self.n_samples))

    def cal_svalue(self,R,X):
        """
        Calculate the S-values of new matrix [X;r.T], where r is each row of R
        
        Arguments:
        R: Matrix where each row vector will be added to matrix X to calculate the its Svalue
        X: Matrix composed of selected vectors from all candidates, (number of selected, number of polynomial functions)

        Return:
        ndarray (n,) containing s-values from each row of R
        
        """
        k,p = X.shape

        if k < p :
            svalues = self._cal_svalue_under(R,X)
        else:
            svalues = self._cal_svalue_over(R,X)
        return svalues

    @random_state
    def samples(self,X,n_samples, *args, **kwargs):
        """
        Xb = Y
        X: Design matrix X(u) of shape(num_samples, num_features)

        Return:
            Experiment samples of shape(ndim, n_samples)
        """
        self.filename= self.filename+num2print(n_samples)
        m, p        = X.shape
        assert m > p, 'Number of candidate samples are expected to be larger than number of features'

        if self.criteria.upper() == 'S':
            """ Xb = Y """
            selected_row   = kwargs.get('rows', None)
            is_basis_orth  = kwargs.get('is_orth', False)
            selected_row   = self._get_quasi_optimal(n_samples, X, selected_row, is_basis_orth)
        elif self.criteria.upper() == 'D':
            """ D optimality based on rank revealing QR factorization  """
            _, _, P = sp.linalg.qr(X.T, pivoting=True)
            selected_row = P[:n_samples]
        else:
            pass

        # self.X = X[selected_row,:]
        # if u is not None:
            # self.u = u[selected_row] if u.ndim == 1 else u[:, selected_row]
        # else:
            # pass
        # self.rows = selected_row
        return selected_row


    def adaptive(self,X,n_samples, selected_col=[]):
        """
        Xb = Y
        X: Design matrix X(u) of shape(num_samples, num_features)

        S: selected columns
        K: sparsity
        Return:
            Experiment samples of shape(ndim, n_samples)
        """

        selected_col = selected_col if selected_col else range(X.shape[1])
        if self.selected_row is None:
            X_selected = X[:,selected_col]
            selected_row = self.samples(X_selected, n_samples=n_samples)
            self.selected_row = selected_row
            return selected_row 
        else:
            X_selected = X[self.selected_row,:][:,selected_col]
            X_candidate= X[:, selected_col]
            ### X_selected.T * B = X_candidate.T -> solve for B
            X_curr_inv = np.dot(np.linalg.inv(np.dot(X_selected, X_selected.T)), X_selected)
            B = np.dot(X_curr_inv, X_candidate.T)
            X_residual = X_candidate.T - np.dot(X_selected.T, B)
            Q, R, P = sp.linalg.qr(X_residual, pivoting=True)
            selected_row = np.concatenate((self.selected_row, P))
            _, i = np.unique(selected_row, return_index=True)
            selected_row = selected_row[np.sort(i)]
            selected_row = selected_row[:len(self.selected_row) + n_samples]
            self.selected_row = selected_row
            return selected_row[len(self.selected_row):]

    def _cal_svalue_over(self, X0, X1):
        """
        Calculate the S value (without determinant) of candidate vectors w.r.t selected subsets
        when the current selection k >= p (eqn. 3.16) for each pair of (X[i,:], X1)


        Arguments:
        X0 -- candidate matrix of shape (number of candidates, p), 
        X1 -- selected subsets matrix of shape (k,p)

        Return:
        S value without determinant (eqn. 3.16)

        """
        AAinv   = LA.inv(np.dot(X1.T,X1))
        # start   = time.time()
        # A_l2    = LA.norm(X1, axis=0).reshape(1,-1) ## l2 norm for each column in X1, row vector
        # svalues = [] 
        # for r in X0:
            # r = r.reshape(1,-1) ## row vector
            # with np.errstate(invalid='ignore'):
                # d1 = 1.0 + np.asscalar(np.dot(r, np.dot(AAinv, r.T)))
                # d2 = np.prod(A_l2**2 + r**2)
            # svalues.append(d1/d2)
        # end = time.time()
        # print('for loop time elapse  : {}'.format(end-start))
        # print(np.around(svalues, 2))
        # start = time.time()
        X1_norms = LA.norm(X1, axis=0)
        # d1 = 1.0 + np.diagonal(X0.dot(AAinv).dot(X0.T))
        d1 = np.log(1.0 + (X0.dot(AAinv) * X0).sum(-1))
        d2 = np.sum(np.log(X1_norms**2 + X0**2), axis=1) 
        svalues = d1 - d2
        # end = time.time()
        # print(np.around(delta, 2))
        # print(max(abs(delta - svalues)))
        # print('matrix time elapse : {}'.format(end-start))

        return svalues

    def _cal_svalue_under(self, X0, X1):
        """
        Calculate the S-value (without determinant) of a candidate vector w.r.t selected subsets
        when the current selection k < p (eqn. 3.18)

        Arguments:
        X0 -- candidate matrix of shape (number of candidates, p), 
        X1 -- selected subsets matrix of shape (k,p)

        Return:
        S value without determinant (eqn. 3.18)

        """
        k,p = X1.shape
        # start = time.time()
        A = copy.copy(X1[0:k, 0:k]) ## shape (k, k)
        AAinv = LA.inv(A.T.dot(A))  ## shape (k, k)
        R = copy.copy(X0[:, 0:k])  ## shape (n-k, k)
        B = AAinv.dot(R.T)          ## shape (k, n-k)
        c = copy.copy(X1[0:k, k]).reshape((k,1))  ## shape(k, 1)  column vector
        g = AAinv.dot(A.T).dot(c)   ## shape (k, 1)
        gamma = X0[:,k]            ## shape (n-k,) 
        Alpha1= R.T * gamma         ## R[:,i] * gamma[i] , shape (k, n-k)
        Alpha1= c.T.dot(A) + Alpha1.T  ## shape (n-k, k), add c.T.dot(A) to each row of Alpha1.T
        Alpha3= g + B * gamma  ## shape (k, n-k)
        Alpha = []
        for ia, r, b, ic in zip(Alpha1, R, B.T, Alpha3.T):
            ia = ia.reshape(k,1) ## (1, k)
            r  =  r.reshape(k,1)  ## ()
            b  =  b.reshape(k,1)
            ic = ic.reshape(k,1)
            ib = np.identity(k) - b.dot(r.T)/(1.0 + r.T.dot(b))
            alpha2 = np.asscalar(ia.T.dot(ib).dot(ic))
            Alpha.append(alpha2)

        # d1 = 1.0 + (R * B.T).sum(-1)  ## shape (n-k, )
        # A_norms = LA.norm(A, axis=0)
        # d2 = np.prod(A_norms**2 + R**2, axis=1) ## shape (n-k, )
        # d4 = np.squeeze(c.T.dot(c) + gamma**2)  ## shape(n-k, )
        # d3 =  d4 - Alpha 
        # print('d1: {}'.format(d1))
        # print('d2: {}'.format(d2))
        # print('d3: {}'.format(d3))
        # print('d4: {}'.format(d4))
        # delta = d1 * d3 / d2 / d4


        d1 = np.log(1.0 + (R * B.T).sum(-1))  ## shape (n-k, )
        A_norms = LA.norm(A, axis=0)
        d2 = np.sum(np.log(A_norms**2 + R**2), axis=1) ## shape (n-k, )
        d4 = np.squeeze(c.T.dot(c) + gamma**2)  ## shape(n-k, )
        d3 =  d4 - Alpha 
        d4 = np.log(d4)
        if np.any(d3 > 0):
            ## d1, d2, d4 > 0. If there exist at least one d3 > 0, set negative d3 to -inf
            d3 = np.log(d3)
            d3 = np.nan_to_num(d3, nan=-np.inf)
            delta = d1 + d3 - d2 - d4
        else:
            ## all d3 < 0. then take the negative of all d3 and return the smallest s value
            d3 = np.log(abs(d3))
            delta = -(d1 + d3 - d2 - d4)

        # print('d1: {}'.format(d1))
        # print('d2: {}'.format(d2))
        # print('d3: {}'.format(d3))
        # print('d4: {}'.format(d4))

        # end   = time.time()
        # print('matrix loop time elaspe: {}'.format(end - start))

        # start = time.time()
        # A = X1[0:k, 0:k] ## shape (k, k)
        # AAinv = LA.inv(A.T.dot(A))  ## shape (k, k)
        # R = X0[:, 0:k]  ## shape (n-k, k)
        # B = AAinv.dot(R.T)          ## shape (k, n-k)
        # c = X1[0:k, k].reshape((k,1))  ## shape(k, 1)  column vector
        # g = AAinv.dot(A.T).dot(c)   ## shape (k, 1)
        # gamma = X0[:,k]            ## shape (n-k,) 
        # Alpha1= R.T * gamma         ## R[:,i] * gamma[i] , shape (k, n-k)
        # Alpha1= c.T.dot(A) + Alpha1.T  ## shape (n-k, k), add c.T.dot(A) to each row of Alpha1.T
        # Alpha3= g + B * gamma  ## shape (k, n-k)
        # Alpha = []
        # # Alpha2= [ np.identity(k) - np.tensordot(b, r, axes=0)/(1.0 + r.dot(b)) for r, b in zip (R, B.T)]
        # # Alpha = [ np.asscalar(ia.reshape(k,1).T.dot(ib).dot(ic.reshape(k,1))) for ia, ib, ic in zip(Alpha1, Alpha2, Alpha3.T)]
        # for ia, r, b, ic in zip(Alpha1, R, B.T, Alpha3.T):
            # ia = ia.reshape(k,1) ## (1, k)
            # r  =  r.reshape(k,1)  ## ()
            # b  =  b.reshape(k,1)
            # ic = ic.reshape(k,1)
            # ib = np.identity(k) - b.dot(r.T)/(1.0 + r.T.dot(b))
            # alpha2 = np.asscalar(ia.T.dot(ib).dot(ic))
            # Alpha.append(alpha2)


        # d1 = 1.0 + (R * B.T).sum(-1)  ## shape (n-k, )
        # A_norms = LA.norm(A, axis=0)
        # d2 = np.prod(A_norms**2 + R**2, axis=1) ## shape (n-k, )
        # d4 = np.squeeze(c.T.dot(c) + gamma**2)  ## shape(n-k, )
        # d3 =  d4 - Alpha 
        # delta = d1 * d3 / d2 / d4
        # end   = time.time()
        # print('list comprehension time elaspe: {}'.format(end - start))

        # print(np.around(svalues, 4))
        # print(np.around(delta, 4))
        # print(max(abs(delta - svalues)))
        return delta

    def _greedy_find_next_point(self, I, Q):
        """
        find the next quasi optimal sample

        Arguments:
        I -- list containing selected row indices from candidate design matrix X
        Q -- QR factorization of candidate design matrix X if basis is not orthogonal, otherwise is X

        Return:
        i -- integer, index with maximum svalue
        """
        ##  Find the index candidate set to chose from (remove those in I from all (0-M))
        I_left   = list(set(range(Q.shape[0])).difference(set(I)))
        Q_left   = Q[np.array(I_left, dtype=np.int32),:]
        Q_select = Q[np.array(I,      dtype=np.int32),:]
        svalues  = self.cal_svalue(Q_left,Q_select)
        assert(len(svalues) == len(I_left))
        i = I_left[np.argmax(svalues)] ## return the index with largest s-value
        return i

    def _get_quasi_optimal(self,m,X,I=None,is_orth=False):
        """
        return row selection matrix S containing indices for quasi optimal experimental design
        based on fast greedy algorithm 

        Arguments:
        m -- size of quasi optimal subset
        X -- design matrix with candidates samples of shape (M,p)
             M: number of samples, p: number of features
        I -- indices, nt ndarray of shape (N,) corresponding row selection matrix of length m
            if I is None, an empty list will be created first and m items will be appended 
            Otherwise, additional (m-m0) items (row index in design matrix X) will be appended 
        is_orth -- Boolean indicating if the basis space is orthogonal

        Returns:
        row selection matrix I of shape (m, M)
        """
        m = int(m)
        assert m > 0, "At least one sample in the designed experiemnts"
        M,p = X.shape
        assert M >= p, "quasi optimal sebset are design for overdetermined problem only"
        (Q,R) = (X, None ) if is_orth else LA.qr(X)
        I = [np.random.randint(0,M)] if I is None else I ## random initialize for the first point
        m1 = len(I) 

        pbar_x  = tqdm(range(m), ascii=True, desc="   - ")
        for _ in pbar_x:
            i = self._greedy_find_next_point(I,Q)
            I.append(i)
        I = sorted(I)
        return np.array(I) 
