#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
from museuq.experiment._experimentbase import ExperimentBase
from museuq.utilities.decorators import random_state
from museuq.utilities.helpers import num2print
import numpy as np, scipy as sp
import numpy.linalg as LA
import copy
import itertools
from tqdm import tqdm
import time
import multiprocessing as mp

def cal_alpha2(x):
    return x[0].dot(x[1]).dot(x[2]).item()

def append_list(list1, list2, n):
    len1 = len(list1)
    for i in list2:
        if i not in list1:
            list1.append(i)
        if len(list1) == len1 + n:
            return list1

class OptimalDesign(ExperimentBase):
    """ Quasi-Optimal Experimental Design and Optimal Design"""

    def __init__(self, optimality, curr_set=[], random_seed=None):
        """
        Optimal/Quasi Optimal Experimental design:
        Arguments:
        n: int, number of samples 
        optimality: optimal design optimality
        """
        super().__init__(random_seed=random_seed)
        self.optimality = optimality 
        self.filename   = '_'.join(['DoE', self.optimality.capitalize()])
        self.curr_set   = curr_set

    def __str__(self):
        return('Optimal Criteria: {:<15s}, num. samples: {:d} '.format(self.optimality, self.n_samples))


    @random_state
    def samples(self,X,n_samples, *args, **kwargs):
        """
        Xb = Y
        X: Design matrix X(u) of shape(num_samples, num_features)
        Arguments:
            X: design matrix to sample from. (feature to be added: distributions + n_candidates)

        Optional: 
            curr_set: list of selected row indices currently
            orth_basis: boolean, True if columns of design matrix is orthogonal to each other asymptotically
        Return:
            list of row indices selected 
        """
        # self.filename= self.filename + num2print(n_samples)

        if self.optimality.upper() == 'S':
            """ Xb = Y """
            orth_basis = kwargs.get('orth_basis', True)
            curr_set   = kwargs.get('curr_set', self.curr_set)
            curr_set   = self._get_quasi_optimal(n_samples, X, curr_set, orth_basis)
        elif self.optimality.upper() == 'D':
            """ D optimality based on rank revealing QR factorization  """
            curr_set   = kwargs.get('curr_set', self.curr_set)
            curr_set   = self._get_rrqr_optimal(n_samples, X, curr_set)
        else:
            raise NotImplementedError
        self.curr_set = curr_set
        return curr_set 

    def _get_rrqr_optimal(self, m, X, curr_set):
        """
        Return indices of m D-optimal samples based on RRQR 
        """
        X = np.array(X, copy=False, ndmin=2)
        n_begin = len(curr_set)
        new_set = []

        while len(new_set) < m:
            ## remove selected rows first
            idx = list(set(range(X.shape[0])).difference(set(curr_set)).difference(set(new_set)))
            if not idx:
                break
            else:
                X_      = X[idx,:]
                n, p    = X_.shape
                _,_,P   = sp.linalg.qr(X.T, pivoting=True)
                new_set+= list(P[:min(m,n,p)])
        new_set = new_set[:m] if len(new_set) > m else new_set ## break case
        curr_set += new_set
        return curr_set
        
    def _get_quasi_optimal(self,m,X,I=None,orth_basis=False):
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
        orth_basis -- Boolean indicating if the basis space is orthogonal

        Returns:
        row selection matrix I of shape (m, M)
        """
        m = int(m)
        assert m > 0, "At least one sample in the designed experiemnts"
        M,p = X.shape
        assert M >= p, "quasi optimal sebset are design for overdetermined problem only"
        (Q,R) = (X, None ) if orth_basis else LA.qr(X)
        I = [] if I is None else I
        pbar_x  = tqdm(range(m), ascii=True, desc="   - ")
        for _ in pbar_x:
            i = self._greedy_find_next_point(I,Q)
            I.append(i)
        return I

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
        
        if not I:
            i = np.random.randint(0,Q.shape[0], size=1).item()
        else:
            I_left   = list(set(range(Q.shape[0])).difference(set(I)))
            Q_left   = Q[np.array(I_left, dtype=np.int32),:]
            Q_select = Q[np.array(I,      dtype=np.int32),:]
            svalues  = self._cal_svalue(Q_left,Q_select)
            if len(svalues) != len(I_left):
                raise ValueError('len(I_left) = {}, however len(svalues) = {}'.format(len(I_left), len(svalues)))
            i = I_left[np.argmax(svalues)] ## return the index with largest s-value
        return i

    def _cal_svalue(self,R,X):
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

    def adaptive(self,X,n_samples, selected_col=[]):
        """
        Coherence Compressinve D optimality
        Xb = Y
        X: Design matrix X(u) of shape(num_samples, num_features)

        S: selected columns
        K: sparsity
        Return:
            Experiment samples of shape(ndim, n_samples)
        """

        selected_col = selected_col if selected_col else range(X.shape[1])
        if self.selected_rows is None:
            X_selected = X[:,selected_col]
            selected_rows = self.samples(X_selected, n_samples=n_samples)
            self.selected_rows = selected_rows
            return selected_rows 
        else:
            X_selected = X[self.selected_rows,:][:,selected_col]
            X_candidate= X[:, selected_col]
            ### X_selected.T * B = X_candidate.T -> solve for B
            X_curr_inv = np.dot(np.linalg.inv(np.dot(X_selected, X_selected.T)), X_selected)
            B = np.dot(X_curr_inv, X_candidate.T)
            X_residual = X_candidate.T - np.dot(X_selected.T, B)
            Q, R, P = sp.linalg.qr(X_residual, pivoting=True)
            selected_rows = np.concatenate((self.selected_rows, P))
            _, i = np.unique(selected_rows, return_index=True)
            selected_rows = selected_rows[np.sort(i)]
            selected_rows = selected_rows[:len(self.selected_rows) + n_samples]
            self.selected_rows = selected_rows
            return selected_rows[len(self.selected_rows):]

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
        X1_norms = LA.norm(X1, axis=0)
        d1 = np.log(1.0 + (X0.dot(AAinv) * X0).sum(-1))
        d2 = np.sum(np.log(X1_norms**2 + X0**2), axis=1) 
        svalues = d1 - d2
        return np.squeeze(svalues)

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
        try:
            AAinv = LA.inv(A.T.dot(A))  ## shape (k, k)
        except np.linalg.LinAlgError:
            u,s,v = np.linalg.svd(A.T.dot(A))
            print('singular value of A.T *A: {}'.format(s))

        R = copy.copy(X0[:, 0:k])  ## shape (n-k, k)
        B = AAinv.dot(R.T)          ## shape (k, n-k)
        c = copy.copy(X1[0:k, k]).reshape((k,1))  ## shape(k, 1)  column vector
        g = AAinv.dot(A.T).dot(c)   ## shape (k, 1)
        gamma = X0[:,k]            ## shape (n-k,) 
        ### calculating alpha with broadcasting
        ### eqn: 3.14-> alpha = Alpha1 * Alpha2 * Alph3
        ### Alpha1 = c.T A + gamma * r.T
        ### Alpha2 = I - b * r.T / (1 + r.T * b)
        ### Alpha3 = g + gamma * b
        Alpha1= R.T * gamma         ## R[:,i] * gamma[i] , shape (k, n-k)
        Alpha1= c.T.dot(A) + Alpha1.T  ## shape (n-k, k), add c.T.dot(A) to each row of Alpha1.T
        Alpha3= g + B * gamma  ## shape (k, n-k)



        # time0 = time.time()
        d1 = 1.0 + (R * B.T).sum(-1)                    ### shape (n-k, )
        Alpha2 = B.T[:,:,np.newaxis] * R[:,np.newaxis] ### shape (n-k, k ,k)
        Alpha2 = np.moveaxis(Alpha2,0,-1)   ## shape(k, k, n-k)
        Alpha2 = Alpha2/d1
        Alpha2 = np.moveaxis(Alpha2,-1, 0)   ## shape(n-k, k ,k)
        I = np.identity(Alpha2.shape[-1])
        Alpha2 = I - Alpha2   ## shape(n-k, k, k)
        if k <= 80:
            Alpha  = np.array([ia.dot(ib).dot(ic).item() for ia, ib, ic in zip(Alpha1[:,np.newaxis], Alpha2, Alpha3.T[:,:,np.newaxis])])
        else:
            pool = mp.Pool(processes=mp.cpu_count())
            results = [pool.map_async(cal_alpha2, zip(Alpha1[:,np.newaxis], Alpha2, Alpha3.T[:,:,np.newaxis]))]
            Alpha = np.squeeze([p.get() for p in results])
            pool.close()

        d1 = np.log(d1)  ## shape (n-k, )
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
        return delta


