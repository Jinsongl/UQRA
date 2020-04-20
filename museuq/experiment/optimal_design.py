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
import museuq.utilities.helpers as helpers 
import numpy as np, scipy as sp
import copy
import itertools
from tqdm import tqdm
import time, math
import multiprocessing as mp

class OptimalDesign(ExperimentBase):
    """ Quasi-Optimal Experimental Design and Optimal Design"""

    def __init__(self, optimality, curr_set=[]):
        """
        Optimal/Quasi Optimal Experimental design:
        Arguments:
        n: int, number of samples 
        optimality: optimal design optimality
        """
        super().__init__()
        self.optimality = optimality 
        self.filename   = '_'.join(['DoE', self.optimality.capitalize()])
        self.curr_set   = curr_set

    def __str__(self):
        return('Optimal Criteria: {:<15s}, num. samples: {:d} '.format(self.optimality, self.n_samples))

    def samples(self,X,n, *args, **kwargs):
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
            try:
                ### return a new set of indices, curr_set will be updated
                row_adding = self._get_quasi_optimal(n, X, curr_set, orth_basis)
                self.curr_set = curr_set
            except:
                print(curr_set)
                raise ValueError('_get_quasi_optimal failed')
            if len(np.unique(row_adding)) != n:
                print('Duplicate samples detected:')
                print(' -> len(curr_set) = {}'.format(len(curr_set)))
                print(' -> len(row_adding) = {}'.format(len(row_adding)))
                print(' -> len(np.unique(row_adding)) = {}'.format(len(np.unique(row_adding))))
        elif self.optimality.upper() == 'D':
            """ D optimality based on rank revealing QR factorization  """
            curr_set = kwargs.get('curr_set', self.curr_set)
            row_adding  = self._get_rrqr_optimal(n, X, curr_set)
            self.curr_set = curr_set + row_adding
            if len(np.unique(row_adding)) != n:
                print('Duplicate samples detected:')
                print(' -> len(curr_set) = {}'.format(len(curr_set)))
                print(' -> len(row_adding) = {}'.format(len(row_adding)))
                print(' -> len(np.unique(row_adding)) = {}'.format(len(np.unique(row_adding))))
        else:
            raise NotImplementedError

        # print('current set: \n{}'.format(curr_set))
        # print('new set: \n{}'.format(row_adding))
        # print('intersection: \n{}'.format(set(row_adding) & set(curr_set)))
        return row_adding 

    def _get_rrqr_optimal(self, m, X, row_selected):
        """
        Return selected rows from design matrix X based on D-optimality implemented by RRQR 

        Arguments:
            row_selected: set of selected indices 
        """
        m       = helpers.check_int(m)
        X       = np.array(X, copy=False, ndmin=2)
        row_adding = [] ## list containing new selected indices in this run
        ## list of candidate indices, note that these indices corresponding to the rows in original design matrix X
        row_candidate= list(set(np.arange(X.shape[0])).difference(set(row_selected)))
        X_candidate  = X[row_candidate, :]  ## return the candidate design matrix after removing selected rows

        ## each QR iteration returns rank(X_candidate) samples, which is min(X_candidate.shape)
        ## to have m samples, need to run RRQR ceil(m/rnak(X_candidate)) times
        for _ in tqdm(range(math.ceil(m/min(X_candidate.shape))), ascii=True, desc='    - [D-Optimal]'):
            ## remove the new selected indices from candidate 
            row_candidate = list(set(row_candidate).difference(set(row_adding)))
            if not row_candidate:
                ## if candidate indices are empty, stop
                break
            else:
                ## update candidate design matrix, note that X row_candidate is the indices in the original design matrix X
                X_candidate  = X[row_candidate,:]
                n, p    = X_candidate.shape
                _,_,P   = sp.linalg.qr(X_candidate.T, pivoting=True)
                ## P[i] is the index in X_candidate corresponding the largest |singular value|
                ## need to find its corresponding index in the original matrix X
                new_set_= [row_candidate[i] for i in P[:min(m,n,p)]]
                ## check if there is any duplicate indices returned
                if set(row_adding) & set(new_set_):
                    raise ValueError('Duplicate samples returned')
                row_adding = row_adding + new_set_ 
        row_adding = row_adding[:m] if len(row_adding) > m else row_adding ## break case
        if set(row_adding) & set(row_selected):
            raise ValueError('Duplicate samples returned')
        row_selected += row_adding
        return row_adding 
        
    def _get_quasi_optimal(self,m,X,row_selected=[],orth_basis=False):
        """
        return row indices for quasi optimal experimental design based on fast greedy algorithm 

        Arguments:
        m -- size of 'new' quasi optimal subset
        X -- design matrix with candidates samples of shape (M,p)
             M: number of samples, p: number of features
        row_selected -- indices, ndarray of shape (N,) corresponding row selection matrix of length m
            if row_selected is None, an empty list will be created first and m items will be appended 
            Otherwise, additional (m-m0) items (row index in design matrix X) will be appended 
        orth_basis -- Boolean indicating if the basis space is orthogonal

        Returns:
        row selection matrix row_selected of shape (m, M)
        """
        m = helpers.check_int(m)
        X = np.array(X, copy=False, ndmin=2)
        if X.shape[0] < X.shape[1]:
            raise ValueError('Quasi optimal sebset are designed for overdetermined problem only')
        (Q, R)  = (X, None) if orth_basis else np.linalg.qr(X)
        row_adding = []
        for _ in tqdm(range(m), ascii=True, desc="   - [S-Optimal]",ncols=80):
            ## find the next optimal index from Q which is not currently selected
            i = self._greedy_find_next_point(row_selected,Q)
            ## check if this index is already selected
            if i in row_selected:
                print('Row {:d} already selected'.format(i))
                raise ValueError('Duplicate sample {:d} already exists'.format(i))
            row_selected.append(i)
            row_adding.append(i)
        return row_adding 

    def _greedy_find_next_point(self, row_selected, Q):
        """
        find the next quasi optimal sample

        Arguments:
        row_selected -- list containing selected row indices from original design matrix Q
        Q -- QR factorization of candidate design matrix X if basis is not orthogonal, otherwise is X

        Return:
        i -- integer, index with maximum svalue
        """
        ##  Find the index candidate set to chose from (remove those in row_selected from all (0-M))
        
        ## if row_selected is empty, choose one randomly
        if not row_selected:
            i = np.random.randint(0,Q.shape[0], size=1).item()
        else:
            row_candidate = list(set(range(Q.shape[0])).difference(set(row_selected)))
            ## split original design matrix Q into candidate matrix Q_cand, and selected Q_sltd
            Q_cand   = Q[np.array(row_candidate, dtype=np.int32),:]
            Q_sltd   = Q[np.array(row_selected , dtype=np.int32),:]
            ## calculate (log)S values for each row in Q_cand together with Q_sltd
            svalues  = self._cal_svalue(Q_cand,Q_sltd)
            if len(svalues) != len(row_candidate):
                raise ValueError('Expecting {:d} S values, but {:d} given'.format(len(row_candidate), len(svalues)))
            i = row_candidate[np.argmax(svalues)] ## return the indices with largest s-value in original matrix Q
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
        Calculate the log(S) value (without determinant) of candidate vectors w.r.t selected subsets
        when the current selection k >= p (eqn. 3.16) for each pair of (X[i,:], X1)


        Arguments:
        X0 -- candidate matrix of shape (n-k, p), 
        X1 -- selected subsets matrix of shape (k,p)

        Return:
        S value without determinant (eqn. 3.16)

        """
        try:
            AAinv = np.linalg.inv(X1.T.dot(X1))  ## shape (k, k)
        except np.linalg.LinAlgError:
            u,s,v = np.linalg.svd(X1.T.dot(X1))
            print('singular value of A.T *A: {}'.format(s))

        X1_norms = np.linalg.norm(X1, axis=0)  ## (p,)
        d1 = np.log(1.0 + (X0.dot(AAinv) * X0).sum(-1)) ## (n-k,)
        d2 = np.sum(np.log(X1_norms**2 + X0**2), axis=1) ## (n-k,)
        svalues = d1 - d2
        return np.squeeze(svalues)

    def _cal_svalue_under(self, X0, X1):
        """
        Calculate the S-value (without determinant) of a candidate vector w.r.t selected subsets
        when the current selection k < p (eqn. 3.18)

        Arguments:
        X0 -- candidate matrix of shape (n-k, p), 
        X1 -- selected submatrix of shape (k,p)

        Return:
        S value without determinant (eqn. 3.18)

        """
        n_k, p  = X0.shape
        k,p     = X1.shape
        # start = time.time()
        A = copy.copy(X1[0:k, 0:k])                         ## shape (k, k)
        try:
            AAinv = np.linalg.inv(A.T.dot(A))               ## shape (k, k)
        except np.linalg.LinAlgError:
            u,s,v = np.linalg.svd(A.T.dot(A))
            print('singular value of A.T *A: {}'.format(s))

        R = copy.copy(X0[:, 0:k])                           ## shape (n-k, k)
        B = AAinv.dot(R.T)                                  ## shape (k, n-k)
        c = copy.copy(X1[0:k, k]).reshape((k,1))            ## shape(k, 1)  column vector
        g = AAinv.dot(A.T).dot(c)                           ## shape (k, 1)
        gamma = X0[:,k]                                     ## shape (n-k,) 
        ### calculating alpha with broadcasting
        ### eqn: 3.14-> alpha = Alpha1 * Alpha2 * Alph3
        ### Alpha1 = c.T A + gamma * r.T
        ### Alpha2 = I - b * r.T / (1 + r.T * b)
        ### Alpha3 = g + gamma * b
        Alpha1 = R.T * gamma                                ## R[:,i] * gamma[i] , shape (k, n-k)
        Alpha1 = c.T.dot(A) + Alpha1.T                      ## shape (n-k, k), add c.T.dot(A) to each row of Alpha1.T
        Alpha3 = g + B * gamma                              ## shape (k, n-k)

        size_of_array_8gb = 1e8
        multiprocessing_threshold= 100
        ## size of largest array is of shape (n-k, k, k)
        if n_k * k * k < size_of_array_8gb:
            d1     = 1.0 + (R * B.T).sum(-1)                ### shape (n-k, )
            Alpha2 = B.T[:,:,np.newaxis] * R[:,np.newaxis]  ### shape (n-k, k ,k)
            Alpha2 = np.moveaxis(Alpha2,0,-1)               ### shape (k, k, n-k)
            Alpha2 = Alpha2/d1
            Alpha2 = np.moveaxis(Alpha2,-1, 0)              ### shape (n-k, k ,k)
            I      = np.identity(Alpha2.shape[-1])
            Alpha2 = I - Alpha2                             ### shape (n-k, k, k)
            Alpha  = [ia.dot(ib).dot(ic).item() for ia, ib, ic in zip(Alpha1[:,np.newaxis], Alpha2, Alpha3.T[:,:,np.newaxis])]
        else:
            batch_size = math.floor(size_of_array_8gb/k/k)  ## large memory is allocated as 8 GB
            Alpha = []
            for i in tqdm(range(math.ceil(n_k/batch_size)), ascii=True, desc='   Batch (n={:d}): -'.format(batch_size)):
                idx_start = i*batch_size
                idx_end   = min((i+1) * batch_size, n_k)
                R_        = R[idx_start:idx_end, :]
                B_        = B[:, idx_start:idx_end]

                # time0 = time.time()
                d1     = 1.0 + (R_ * B_.T).sum(-1)              ### shape (n-k, )
                Alpha2 = B_.T[:,:,np.newaxis] * R_[:,np.newaxis]### shape (n-k, k ,k)
                Alpha2 = np.moveaxis(Alpha2,0,-1)               ### shape (k, k, n-k)
                Alpha2 = Alpha2/d1
                Alpha2 = np.moveaxis(Alpha2,-1, 0)              ### shape (n-k, k ,k)
                I      = np.identity(Alpha2.shape[-1])
                Alpha2 = I - Alpha2                             ### shape (n-k, k, k)
                Alpha_ =[ia.dot(ib).dot(ic).item() for ia, ib, ic in zip(Alpha1[idx_start:idx_end,np.newaxis], Alpha2, Alpha3.T[idx_start:idx_end,:,np.newaxis])]
                Alpha.extend(Alpha_)

        Alpha = np.array(Alpha)
        if Alpha.shape != (n_k,):
            print(Alpha)
            raise ValueError('Expecting Alpha shape to be ({},), but {} given'.format(n_k, Alpha.shape))
        d1 = np.log(1.0 + (R * B.T).sum(-1))  ## shape (n-k, )
        A_norms = np.linalg.norm(A, axis=0)
        d2 = np.sum(np.log(A_norms**2 + R**2), axis=1) ## shape (n-k, )
        d4 = np.squeeze(c.T.dot(c) + gamma**2)  ## shape(n-k, )
        d3 =  d4 - Alpha 
        d4 = np.log(d4)

        if np.any(d3 > 0):
            ## d1, d2, d4 > 0. If there exist at least one d3 > 0, set negative d3 to -inf
            with np.errstate(divide='ignore'):
                d3 = np.log(d3)
                d3 = np.nan_to_num(d3, nan=-np.inf)
                delta = d1 + d3 - d2 - d4
        else:
            ## all d3 < 0. then take the negative of all d3 and return the smallest s value
            d3 = np.log(abs(d3))
            delta = -(d1 + d3 - d2 - d4)
        return delta




    # def _cal_svalue_over(self, X0, X1):
        # """
        # Calculate the S value (without determinant) of candidate vectors w.r.t selected subsets
        # when the current selection k >= p (eqn. 3.16) for each pair of (X[i,:], X1)


        # Arguments:
        # X0 -- candidate matrix of shape (number of candidates, p), 
        # X1 -- selected subsets matrix of shape (k,p)

        # Return:
        # log S value without determinant (eqn. 3.16)

        # """
        # XXinv   = np.linalg.inv(np.dot(X1.T,X1))
        # start   = time.time()
        # A_l2    = np.linalg.norm(X1, axis=0).reshape(1,-1) ## l2 norm for each column in X1, row vector
        # svalues_log = [] 
        # for r in X0:
            # r = r.reshape(1,-1) ## row vector
            # with np.errstate(invalid='ignore'):
                # d1 = np.log(1 + np.dot(r, np.dot(XXinv, r.T)))
                # d2 = np.log(np.prod(A_l2**2 + r**2))
            # svalues_log.append(d1 - d2)
        # end = time.time()
        # print('for loop time elapse  : {}'.format(end-start))
        # # print(np.around(np.exp(svalues_log), 2))
        # start = time.time()
        # X1_norms = np.linalg.norm(X1, axis=0)
        # # d1 = 1.0 + np.diagonal(X0.dot(XXinv).dot(X0.T))
        # d1 = 1.0 + (X0.dot(XXinv) * X0).sum(-1)
        # d2 = np.prod(X1_norms**2 + X0**2, axis=1) 
        # delta = d1/d2
        # end = time.time()
        # # print(np.around(delta, 2))
        # print('matrix time elapse : {}'.format(end-start))

        # return svalues_log

    # def _cal_svalue_under(self, X0, X1):
        # """
        # Calculate the log S-value (without determinant) of a candidate vector w.r.t selected subsets
        # when the current selection k < p (eqn. 3.18)

        # Arguments:
        # X0 -- candidate matrix of shape (number of candidates, p), 
        # X1 -- selected subsets matrix of shape (k,p)

        # Return:
        # log S value without determinant (eqn. 3.18)

        # """
        # k,p = X1.shape
        # assert k < p
        # X1 = copy.copy(X1[:,0:k])
        # X0 = copy.copy(X0[:,0:k+1])
        # svalues_log = [] 
        # XXinv = np.linalg.inv(np.dot(X1.T,X1))
        # A_l2 = np.linalg.norm(X1, axis=0).reshape(1,-1)


        # for r in X0:
            # c = r[0:k].reshape((k,1)) ## column vector
            # gamma = r[k]
            # r = copy.copy(c)

            # b = np.dot(XXinv,r)
            # g = np.dot(XXinv,np.dot(X1.T,c))

            # a1 = np.dot(c.T,X1) + gamma * r.T
            # a2 = np.identity(k) - np.dot(b,r.T)/(1 + np.dot(r.T,b))
            # a3 = g + gamma *b
            # a = np.squeeze(np.dot(a1,np.dot(a2,a3)))

            # with np.errstate(invalid='ignore'):
                # d1 = np.log(np.squeeze(1 + np.dot(r.T, b)))
                # d2 = np.sum(np.log(A_l2**2 + r.T**2))
                # d3 = np.log(np.squeeze(np.dot(c.T,c) + gamma**2 - a))
                # d4 = np.log(np.squeeze(np.dot(c.T,c) + gamma**2))
            # svalues_log.append(d1 + d3 - d2 - d4)
        # return svalues_log

