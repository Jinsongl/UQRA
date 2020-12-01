#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
from uqra.experiment._experimentbase import ExperimentBase
import uqra.utilities.helpers as helpers 
import numpy as np, scipy as sp
import copy
import itertools
from tqdm import tqdm
import time, math
import multiprocessing as mp
import warnings

class OptimalDesign(ExperimentBase):
    """ Quasi-Optimal Experimental Design and Optimal Design"""

    def __init__(self, X, optimal_samples=[]):
        """
        Optimal/Quasi Optimal Experimental design:
        Arguments:
            X: {array-like, sparse matrix} of shape (n_samples, n_features)
            optimal_samples: list of indices for optimal samples 
        """
        super().__init__()
        ## return the candidate design matrix after removing selected rows
        self.X  = np.array(X, copy=False, ndmin=2, dtype=np.float64)
        self.optimal_samples   = optimal_samples
        self.candidate_samples = self._list_diff(np.arange(X.shape[0]), self.optimal_samples)
        self._check_complement(self.candidate_samples, self.optimal_samples)

    def __str__(self):
        return('UQRA.Experiment.OptimalDesign')

    def get_samples(self, optimality, n, **kwargs):
        """
        Return n new samples from X exclude those already selected
        Xb = Y
        X: Design matrix X(u) of shape(n_samples, n_features)
        Arguments:
            n: int, number of new samples to be added

        Optional: 
            optimal_samples: list of selected row indices currently
            is_orth_col: boolean, True if columns of design matrix is orthogonal to each other asymptotically
        Return:
            list of row indices selected 
        """
        n = helpers.check_int(n)
        optimality      = str(optimality).upper()
        self.filename   = '_'.join(['DoE', optimality])
        optimal_samples = kwargs.get('optimal_samples', [])
        self.optimal_samples   = self._list_union(self.optimal_samples, optimal_samples)
        self.candidate_samples = self._list_diff(self.candidate_samples, self.optimal_samples)
        assert self._check_complement(self.optimal_samples, self.candidate_samples)

        try:
            algorithm = kwargs['algorithm']
        except KeyError:
            if optimality == 'S':
                algorithm = 'AFP' ## approximated Fekete points with QR with column pivoting
            elif optimality == 'D':
                algorithm = 'RRQR' ## QR with column pivoting  
            else:
                raise NotImplementedError

        if optimality == 'D':
            print('   > UQRA D-Optimality Design with {:s}'.format(algorithm))
            optimal_samples = self.D_Optimality(n, algorithm=algorithm)

        if optimality.upper() == 'S':
            print('   > UQRA S-Optimality Design with {:s}'.format(algorithm))
            optimal_samples = self.S_Optimality(n, algorithm=algorithm)

        self.optimal_samples   = self._list_union(self.optimal_samples, optimal_samples)
        self.candidate_samples = self._list_diff(self.candidate_samples, optimal_samples)
        return self.optimal_samples

    def D_Optimality(self, n, algorithm='RRQR'):
        """
        Optimal design with D-optimality
        Arguments:
            n: int, number of samples to be returned
            Algorithm: str, algorithms used to generated D-optimal samples
                Available in reference:
                    1 General Exchange Procedure 
                    2 DETMAX Algorithm 
                    3 Fedorov Algorithm
                    4 Modified Fedorov Algorithm 
                    5 k-Exchange Algorithm
                    6 kl-Exchange Algorithm 
                    7 Modified kl-Exchange Algorithm 
                Available in MATLAB:
                     cordexch
                     rowexch
                
        Algorithms reference:  
            Triefenbach, F. (2008). Design of experiments: the D-optimal approach and its implementation as a computer algorithm. Bachelor's Thesis in Information and Communication Technology.
        """
        if algorithm.upper() == 'RRQR':
            try:
                optimal_samples = self._get_samples_rrqr(n)
            except:
                print(self.optimal_samples)
                raise ValueError('Error: UQRA.Experiment.OptimalDesign._get_samples_rrqr failed')
        else:
            print('NotImplementedError: UQRA.OptimalDesign.D_Optimality: algorithm {:s} not implemented...'.format(algorithm))
            raise NotImplementedError
        return optimal_samples

    def S_Optimality(self, n, algorithm='AFP'):
        """
        Optimal design with S-optimality
        Arguments:
            n: int, number of samples to be returned
            Algorithm: str, algorithms used to initialize S-optimal samples
                1. 'TSM': truncated square matrices randomly choose first one and sequentially choose next sample with largest S-value
                2. 'AFP': first n_feature samples are chosen as the Fekete points, calculated by Approximated Fekete Point algorithm
                
        Algorithms reference:  
           Bos, L., De Marchi, S., Sommariva, A., & Vianello, M. (2010). Computing multivariate Fekete and Leja points by numerical linear algebra. SIAM Journal on Numerical Analysis, 48(5), 1984-1999.
        """

        if len(self.optimal_samples)!=0:
            ## if selected optimal samples are not empty, choose following samples that maximize S-values based on current selected samples
            optimal_samples = self._get_samples_svalue(n, self.candidate_samples, self.optimal_samples)
        else:
            ## Two algorithms to initialize optimal samples
            X_rank = min(self.X.shape)
            if algorithm.upper() in ['AFP', 'RRQR']:
                try:
                    optimal_samples = self._get_samples_rrqr(min(X_rank, n))
                except:
                    print(self.optimal_samples)
                    raise ValueError('Error: UQRA.Experiment.OptimalDesign._get_samples_rrqr failed')
            elif algorithm.upper() == 'TSM': ##
                try:
                    ### return a new set of indices, optimal_samples will be updated
                    optimal_samples = self._get_samples_tsm(min(X_rank, n))
                except:
                    print(self.optimal_samples)
                    raise ValueError('Error: UQRA.Experiment.OptimalDesign._get_samples_tsm failed')
            else:
                print('UQRA.OptimalDesign.S_Optimality: algorithm {:s} not implemented...'.format(algorithm))
                raise NotImplementedError

            candidate_samples = self._list_diff(self.candidate_samples, optimal_samples)
            if min(X_rank, n) < n:
                optimal_samples1 = self._get_samples_svalue(n-min(X_rank, n), candidate_samples, optimal_samples)
                optimal_samples = self._list_union(optimal_samples, optimal_samples1)
        return optimal_samples

    def _get_samples_rrqr(self, n):
        """
        Return selected rows from design matrix X based on D-optimality implemented by RRQR 

        Arguments:
            row_selected: set of selected indices 
        """
        n = helpers.check_int(n)
        candidate_samples = self.candidate_samples
        optimal_samples   = []
        ## each QR iteration returns rank(X) samples, which is min(candidate samples, n_features)
        X_rank = min(len(candidate_samples), self.X.shape[1])
        n_iterations = math.ceil(n/X_rank)
        for i_iteration in tqdm(range(n_iterations), ascii=True, desc='   - [RRQR]',ncols=80):
            ## remove the new selected indices from candidate 
            X = self.X[candidate_samples, :]
            ## number of samples to be generated in this iteration, X_rank except for the last iteration
            n_samples = X_rank if i_iteration < n_iterations-1 else int(n-X_rank * i_iteration)
            if X.shape[0] <= n_samples:
                optimal_samples_ = candidate_samples
                if X.shape[0] < n_samples:
                    print('\n'+'Warning: UQRA.Experiment.OptimalDesign._get_samples_rrqr: requested {:d} optimal samples but only {:d} candidate samples available, returning all candiate samples'.format(n_samples, X.shape[0]))
            else:
                ## update candidate design matrix, note that X row_candidate is the indices in the original design matrix X
                _,_,Pivot = sp.linalg.qr(X.T, pivoting=True)
                ## Pivot[i] is the index in X corresponding the largest |singular value|
                ## need to find its corresponding index in the original matrix X
                optimal_samples_ = [candidate_samples[i] for i in Pivot[:n_samples]]
            optimal_samples  = self._list_union(optimal_samples,  optimal_samples_)
            candidate_samples= self._list_diff(candidate_samples, optimal_samples_)
        if len(optimal_samples)!= n:
            raise ValueError('Requesting {:d} new samples but {:d} returned'.format(n, n0))
        return optimal_samples
        
    def _get_samples_tsm(self,n):
        """
        return initial points for S-optimality based on Truncated Square matrices
        """
        n = helpers.check_int(n) 
        optimal_samples = [np.random.randint(0, self.X.shape[0], size=1).item(),]
        candidate_samples = self._list_diff(self.candidate_samples, optimal_samples)
        optimal_samples0= self._get_samples_svalue(n-1, candidate_samples, optimal_samples) 
        optimal_samples = self._list_union(optimal_samples, optimal_samples0)
        return optimal_samples

    def _get_samples_svalue(self,n, candidate_samples, optimal_samples):
        """
        return row indices for quasi optimal experimental design based on fast greedy algorithm 

        Arguments:
        n -- size of 'new' quasi optimal subset
        X -- design matrix with candidates samples of shape (M,p)
             M: number of samples, p: number of features
        row_selected -- indices, ndarray of shape (N,) corresponding row selection matrix of length n
            if row_selected is None, an empty list will be created first and n items will be appended 
            Otherwise, additional (n-m0) items (row index in design matrix X) will be appended 
        is_orth_col -- Boolean indicating if the basis space is orthogonal

        Returns:
        row selection matrix row_selected of shape (n, M)
        """
        candidate_samples = copy.deepcopy(candidate_samples)
        optimal_samples   = copy.deepcopy(optimal_samples)
        optimal_samples_  = [] ## new samples to be added in this step 
        for _ in tqdm(range(n), ascii=True, desc="   - [S-values]",ncols=80):
        # for _ in range(n):
            ## find the next optimal index from Q which is not currently selected
            optimal_samples_all= self._list_union(optimal_samples, optimal_samples_)
            candidate_samples_ = self._list_diff(candidate_samples, optimal_samples_)
            X_cand  = self.X[candidate_samples_ ,:]
            X_sltd  = self.X[optimal_samples_all,:]
            ## calculate (log)S values for each row in X_cand together with X_sltd
            svalues = self._cal_svalue(X_cand,X_sltd)
            if len(svalues) != len(candidate_samples_):
                raise ValueError('Expecting {:d} S values, but {:d} given'.format(len(candidate_samples_), len(svalues)))
            i = candidate_samples_[np.argmax(svalues)] ## return the indices with largest s-value in original matrix Q
            ## check if this index is already selected
            if i in optimal_samples_all:
                print('Row {:d} already selected'.format(i))
                raise ValueError('Duplicate sample {:d} already exists'.format(i))
            optimal_samples_.append(i)
        return optimal_samples_

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
            # Alpha  = [ia.dot(ib).dot(ic).item() for ia, ib, ic in zip(Alpha1[:,np.newaxis], Alpha2, Alpha3.T[:,:,np.newaxis])]
            Alpha_ = np.einsum('ijk,ikl->ijl', Alpha1[:,np.newaxis], Alpha2, optimize='greedy')
            Alpha  = np.einsum('ijl,ilj->i', Alpha_, Alpha3.T[:,:,np.newaxis], optimize='greedy')
        else:
            batch_size = math.floor(size_of_array_8gb/k/k)  ## large memory is allocated as 8 GB
            Alpha = []
            # for i in tqdm(range(math.ceil(n_k/batch_size)), ascii=True, desc='   Batch (n={:d}): -'.format(batch_size),ncols=80):
            for i in range(math.ceil(n_k/batch_size)):
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

                Alpha_ = np.einsum('ijk,ikl->ijl', Alpha1[idx_start:idx_end,np.newaxis], Alpha2, optimize='greedy')
                Alpha_ = np.einsum('ijl,ilj->i', Alpha_, Alpha3.T[idx_start:idx_end,:,np.newaxis], optimize='greedy')

                # Alpha_ =[ia.dot(ib).dot(ic).item() for ia, ib, ic in zip(Alpha1[idx_start:idx_end,np.newaxis], Alpha2, Alpha3.T[idx_start:idx_end,:,np.newaxis])]
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

    def _remove_rows_from_matrix(self, A, rows):
        """
        remove rows from matrix A
        Arguments:
            A: ndarray of shape(m,n)
            rows: row index to be removed, len=k
        Return:
            matrix of shape(m-k,n)
        """

        A = np.array(A, copy=False, ndmin=2)
        A = A[list(set(np.arange(A.shape[0])).difference(set(rows))), :]  
        return A

    def _argmax_svalues_greedy(self, optimal_samples, Q):
        """
        find the next quasi optimal sample

        Arguments:
        optimal_samples -- list containing selected row indices from original design matrix Q
        Q -- QR factorization of candidate design matrix X if basis is not orthogonal, otherwise is X

        Return:
        i -- integer, index with maximum svalue
        """
        ##  Find the index candidate set to chose from (remove those in optimal_samples from all (0-M))
        
        ## if optimal_samples is empty, choose one randomly
        row_candidate = list(set(range(Q.shape[0])).difference(set(optimal_samples)))
        ## split original design matrix Q into candidate matrix Q_cand, and selected Q_sltd
        Q_cand   = Q[np.array(row_candidate, dtype=np.int32),:]
        Q_sltd   = Q[np.array(optimal_samples , dtype=np.int32),:]
        ## calculate (log)S values for each row in Q_cand together with Q_sltd
        svalues  = self._cal_svalue(Q_cand,Q_sltd)
        if len(svalues) != len(row_candidate):
            raise ValueError('Expecting {:d} S values, but {:d} given'.format(len(row_candidate), len(svalues)))
        i = row_candidate[np.argmax(svalues)] ## return the indices with largest s-value in original matrix Q
        return i


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

    def _check_complement(self, A, B, U=None):
        """
        check if A.union(B) = U and A.intersection(B) = 0
        """
        A = set(A)
        B = set(B)
        U = set(np.arange(self.X.shape[0])) if U is None else U
        if A.union(B) != U:
            raise ValueError(' Union of sets A and B are not the universe U')
        if len(A.intersection(B)) != 0:
            raise ValueError(' Sets A and B have common elements: {}'.format(A.intersection(B)))
        return True

    def _list_union(self, ls1, ls2):
        """
        append ls2 to ls1 and check if there exist duplicates
        return the union of two lists and remove duplicates
        """
        ls = list(copy.deepcopy(ls1)) + list(copy.deepcopy(ls2))
        if len(ls) != len(set(ls1).union(set(ls2))):
            raise ValueError('Duplicate elements found in list when append to each other')
        return ls

    def _list_diff(self, ls1, ls2):
        """
        returns a list that is the difference between two list, elements present in ls1 but not in ls2
        """
        ls1 = list(copy.deepcopy(ls1))
        ls2 = list(copy.deepcopy(ls2))
        for element in ls2:
            ls1.remove(element)
        return ls1

    def _list_inter(self, ls1, ls2):
        """
        return common elements between ls1 and ls2 
        """
        ls = list(set(ls1).intersection(set(ls2)))
        return ls
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

