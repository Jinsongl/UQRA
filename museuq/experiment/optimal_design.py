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
import time, math
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
            orth_basis  = kwargs.get('orth_basis', True)
            curr_set    = kwargs.get('curr_set', self.curr_set)
            new_set     = self._get_quasi_optimal(n_samples, X, curr_set, orth_basis)
            if len(np.unique(new_set)) != n_samples:
                print('Duplicate samples detected:')
                print('len(curr_set) = {}'.format(len(curr_set)))
                print('len(new_set) = {}'.format(len(new_set)))
                print('len(np.unique(new_set)) = {}'.format(len(np.unique(new_set))))
        elif self.optimality.upper() == 'D':
            """ D optimality based on rank revealing QR factorization  """
            curr_set = kwargs.get('curr_set', self.curr_set)
            new_set  = self._get_rrqr_optimal(n_samples, X, curr_set)
            if len(np.unique(new_set)) != n_samples:
                print('Duplicate samples detected:')
                print('len(curr_set) = {}'.format(len(curr_set)))
                print('len(new_set) = {}'.format(len(new_set)))
                print('len(np.unique(new_set)) = {}'.format(len(np.unique(new_set))))
        else:
            raise NotImplementedError

        print(set(new_set) & set(curr_set))
        self.curr_set = curr_set + new_set
        return new_set 

    def _get_rrqr_optimal(self, m, X, curr_set=[]):
        """
        Return indices of m D-optimal samples based on RRQR 
        """
        X       = np.array(X, copy=False, ndmin=2)
        new_set = []
        idx_cand= list(set(np.arange(X.shape[0])).difference(set(curr_set)))
        X_      = X[idx_cand, :]
        for _ in tqdm(range(math.ceil(m/min(X_.shape))), ascii=True, desc='    -'):
            idx_cand = list(set(idx_cand).difference(set(new_set)))
            if not idx_cand:
                break
            else:
                X_      = X[idx_cand,:]
                n, p    = X_.shape
                _,_,P   = sp.linalg.qr(X_.T, pivoting=True)
                new_set_= [idx_cand[i] for i in P[:min(m,n,p)]]
                if set(new_set) & set(new_set_):
                    raise ValueError
                new_set = new_set + new_set_ 
        new_set = new_set[:m] if len(new_set) > m else new_set ## break case
        return new_set 
        
    def _get_quasi_optimal(self,m,X,curr_set=[],orth_basis=False):
        """
        return row selection matrix S containing indices for quasi optimal experimental design
        based on fast greedy algorithm 

        Arguments:
        m -- size of quasi optimal subset
        X -- design matrix with candidates samples of shape (M,p)
             M: number of samples, p: number of features
        curr_set -- indices, nt ndarray of shape (N,) corresponding row selection matrix of length m
            if curr_set is None, an empty list will be created first and m items will be appended 
            Otherwise, additional (m-m0) items (row index in design matrix X) will be appended 
        orth_basis -- Boolean indicating if the basis space is orthogonal

        Returns:
        row selection matrix curr_set of shape (m, M)
        """
        int_m = int(m)
        if int_m != m:
            raise ValueError("deg must be integer")
        if int_m < 0:
            raise ValueError("deg must be non-negative")
        (Q,R)   = (X, None ) if orth_basis else LA.qr(X)
        X       = np.array(X, copy=False, ndmin=2)
        M,p     = X.shape
        new_set = []
        idx_cand= list(set(np.arange(X.shape[0], dtype=np.int32)).difference(set(curr_set)))
        assert M >= p, "quasi optimal sebset are design for overdetermined problem only"

        pbar_x  = tqdm(range(int_m), ascii=True, desc="   - ")
        for _ in pbar_x:
            i = self._greedy_find_next_point(curr_set,Q)
            curr_set.append(i)
        return curr_set

    def _greedy_find_next_point(self, curr_set, Q):
        """
        find the next quasi optimal sample

        Arguments:
        curr_set -- list containing selected row indices from candidate design matrix X
        Q -- QR factorization of candidate design matrix X if basis is not orthogonal, otherwise is X

        Return:
        i -- integer, index with maximum svalue
        """
        ##  Find the index candidate set to chose from (remove those in curr_set from all (0-M))
        
        if not curr_set:
            i = np.random.randint(0,Q.shape[0], size=1).item()
        else:
            cand_set = list(set(range(Q.shape[0])).difference(set(curr_set)))
            Q_cand   = Q[np.array(cand_set, dtype=np.int32),:]
            Q_sltd   = Q[np.array(curr_set, dtype=np.int32),:]
            svalues  = self._cal_svalue(Q_cand,Q_sltd)
            if len(svalues) != len(cand_set):
                raise ValueError('len(cand_set) = {}, however len(svalues) = {}'.format(len(cand_set), len(svalues)))
            i = cand_set[np.argmax(svalues)] ## return the index with largest s-value
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
        n_k, p  = X0.shape
        k,p     = X1.shape
        # start = time.time()
        A = copy.copy(X1[0:k, 0:k]) ## shape (k, k)
        try:
            AAinv = LA.inv(A.T.dot(A))  ## shape (k, k)
        except np.linalg.LinAlgError:
            u,s,v = np.linalg.svd(A.T.dot(A))
            print('singular value of A.T *A: {}'.format(s))

        R = copy.copy(X0[:, 0:k])  ## shape (n-k, k)
        B = AAinv.dot(R.T)         ## shape (k, n-k)
        c = copy.copy(X1[0:k, k]).reshape((k,1))  ## shape(k, 1)  column vector
        g = AAinv.dot(A.T).dot(c)  ## shape (k, 1)
        gamma = X0[:,k]            ## shape (n-k,) 
        ### calculating alpha with broadcasting
        ### eqn: 3.14-> alpha = Alpha1 * Alpha2 * Alph3
        ### Alpha1 = c.T A + gamma * r.T
        ### Alpha2 = I - b * r.T / (1 + r.T * b)
        ### Alpha3 = g + gamma * b
        Alpha1= R.T * gamma         ## R[:,i] * gamma[i] , shape (k, n-k)
        Alpha1= c.T.dot(A) + Alpha1.T  ## shape (n-k, k), add c.T.dot(A) to each row of Alpha1.T
        Alpha3= g + B * gamma  ## shape (k, n-k)

        ## size of largest array is of shape (n-k, k, k)
        if n_k * k * k < 1e9:
            d1 = 1.0 + (R * B.T).sum(-1)                    ### shape (n-k, )
            Alpha2 = B.T[:,:,np.newaxis] * R[:,np.newaxis] ### shape (n-k, k ,k)
            Alpha2 = np.moveaxis(Alpha2,0,-1)   ## shape(k, k, n-k)
            Alpha2 = Alpha2/d1
            Alpha2 = np.moveaxis(Alpha2,-1, 0)   ## shape(n-k, k ,k)
            I = np.identity(Alpha2.shape[-1])
            Alpha2 = I - Alpha2   ## shape(n-k, k, k)
            if k <= 80:
                Alpha  = [ia.dot(ib).dot(ic).item() for ia, ib, ic in zip(Alpha1[:,np.newaxis], Alpha2, Alpha3.T[:,:,np.newaxis])]
            else:
                pool = mp.Pool(processes=mp.cpu_count())
                results = [pool.map_async(cal_alpha2, zip(Alpha1[:,np.newaxis], Alpha2, Alpha3.T[:,:,np.newaxis]))]
                Alpha = [p.get() for p in results]
                pool.close()
        else:
            batch_size = math.floor(1e9/k/k)  ## large memory is allocated as 8 GB
            Alpha = []
            for i in range(math.ceil(n_k/batch_size)):
                idx_start = i*batch_size
                idx_end   = min((i+1) * batch_size, n_k)
                R_ = R[idx_start:idx_end, :]
                B_ = B[:, idx_start:idx_end]

                # time0 = time.time()
                d1 = 1.0 + (R_ * B_.T).sum(-1)                  ### shape (n-k, )
                Alpha2 = B_.T[:,:,np.newaxis] * R_[:,np.newaxis]### shape (n-k, k ,k)
                Alpha2 = np.moveaxis(Alpha2,0,-1)               ### shape (k, k, n-k)
                Alpha2 = Alpha2/d1
                Alpha2 = np.moveaxis(Alpha2,-1, 0)              ### shape (n-k, k ,k)
                I = np.identity(Alpha2.shape[-1])
                Alpha2 = I - Alpha2                             ### shape (n-k, k, k)
                if k <= 80:
                    Alpha_ =[ia.dot(ib).dot(ic).item() for ia, ib, ic in zip(Alpha1[idx_start:idx_end,np.newaxis], Alpha2, Alpha3.T[idx_start:idx_end,:,np.newaxis])]
                else:
                    pool = mp.Pool(processes=mp.cpu_count())
                    results = [pool.map_async(cal_alpha2, zip(Alpha1[idx_start:idx_end,np.newaxis], Alpha2, Alpha3.T[idx_start:idx_end,:,np.newaxis]))]
                    Alpha_ = [p.get() for p in results]
                    pool.close()
                Alpha += Alpha_

        Alpha = np.array(Alpha)
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
        return delta




