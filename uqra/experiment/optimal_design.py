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

    def __init__(self, X):
        """
        Optimal/Quasi Optimal Experimental design:
        Arguments:
            X: {array-like, sparse matrix} of shape (n_samples, n_features)
            optimal_samples: list of indices for optimal samples 
        """
        super().__init__()
        ## return the candidate design matrix after removing selected rows
        self.X  = np.array(X, copy=False, ndmin=2, dtype=np.float64)
        self.optimal_samples   = [] 
        self.candidate_samples = list(np.arange(X.shape[0]))

    def __str__(self):
        return('UQRA.Experiment.OptimalDesign')

    def samples(self, optimality, n, initialization='AFP', algorithm='GREEDY', **kwargs):
        """
        Perform Optimal design with specified optimality, return n samples
        Return n new samples from X : Design matrix X(u) of shape(n_samples, n_features)
        Arguments:
            optimality: str, alphabetic optimal design
            n: int, number of new samples to be added
            initialization: method to initialize optimal sample sets
                1. 'TSM': truncated square matrices 
                2. 'AFP': Approximated Fekete Point 
                3. a list of indices represeting selected samples from candidate
            algorithm: algorithm employed to perform each optimality

        Optional: 
            
            is_orth_col: boolean, True if columns of design matrix is orthogonal to each other asymptotically
        Return:
            list of row indices selected 
        """
        n          = helpers.check_int(n)
        optimality = str(optimality).upper()
        # self.filename = '_'.join(['DoE', optimality])
        print('   > UQRA {:s}-Optimality Design: n={:d} ...'.format(optimality, n))

        if isinstance(initialization, str):
            ## selected optimal samples at this step must be empty
            assert len(self.optimal_samples) == 0
            n_initialization = min(self.X.shape[0], self.X.shape[1], n)
            if initialization.upper() in ['QR', 'RRQR', 'AFP', 'FEKETE']:
                optimal_samples = self._initial_samples_rrqr(n_initialization)
                print('    -> 1: Initialization ({:s}), n={:d} ...'.format(initialization, len(optimal_samples)))

            elif initialization.upper() in ['TSM', 'TRUNCATED', 'SQUARE']:
                optimal_samples = self._initial_samples_greedy_tsm(n_initialization, optimality)
                print('    -> 1: Initialization ({:s}), n={:d} ...'.format(initialization, len(optimal_samples)))

            else:
                print('    -> UQRA {:s}-Optimality Design: Initialization {:s} NOT implemented'.format(initialization))
                raise NotImplementedError
            n = n - len(optimal_samples)

        elif isinstance(initialization, (list, tuple, np.ndarray, np.generic)):
            n_min = min(self.X.shape)
            optimal_samples = list(np.array(initialization).flatten())
            print('    -> 1: Initialization with selected samples: n={:d} ...'.format(len(optimal_samples)))
            if len(optimal_samples) < n_min:
                optimal_samples0 = self._initial_samples_greedy_tsm(min(n_min-len(optimal_samples),n), optimality,
                    optimal_samples=optimal_samples)
                n = n - len(optimal_samples0) + len(optimal_samples)
                optimal_samples = optimal_samples0
        else:
            print('   > {} not implemented for UQRA.OptiamlDesign'.format(initialization))
            raise NotImplementedError

        self.optimal_samples   = optimal_samples
        self.candidate_samples = self._list_diff(self.candidate_samples, optimal_samples)
        assert self._check_complement(self.optimal_samples, self.candidate_samples)

        if n>0:
            print('    -> 2: Continue Optimality Design, n={:d} ...'.format(n))
            if optimality == 'D':
                optimal_samples = self.get_D_Optimality_samples(n, algorithm=algorithm)

            if optimality.upper() == 'S':
                optimal_samples = self.get_S_Optimality_samples(n, algorithm=algorithm)
        else:
            optimal_samples = []

        self.optimal_samples   = self._list_union(self.optimal_samples, optimal_samples)
        self.candidate_samples = self._list_diff(self.candidate_samples, optimal_samples)
        assert self._check_complement(self.optimal_samples, self.candidate_samples)
        return self.optimal_samples

    def get_D_Optimality_samples(self, n, algorithm):
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
            Triefenbach, F. (2008). Design of experiments: the D-optimal approach and its implementation as a computer initialization. Bachelor's Thesis in Information and Communication Technology.
        """
        if algorithm.upper() == 'GREEDY':

            candidate_samples = copy.deepcopy(self.candidate_samples)
            optimal_samples   = copy.deepcopy(self.optimal_samples)
            for _ in tqdm(range(n), ascii=True, desc="          [Greedy D]",ncols=80):
                ## find the next optimal index from Q which is not currently selected
                candidate_samples = self._list_diff(candidate_samples, optimal_samples)
                assert self._check_complement(optimal_samples, candidate_samples)
                X_cand  = self.X[candidate_samples ,:]
                X_sltd  = self.X[optimal_samples,:]
                ## calculate (log)S values for each row in X_cand together with X_sltd
                optimality_values = self._greedy_update_D_Optimality_full(X_sltd, X_cand)
                if len(optimality_values) != len(candidate_samples):
                    raise ValueError('Expecting {:d} S values, but {:d} given'.format(len(candidate_samples), len(optimality_values)))
                i = candidate_samples[np.argmax(optimality_values)] ## return the indices with largest s-value in original matrix Q
                ## check if this index is already selected
                if i in optimal_samples:
                    print('Row {:d} already selected'.format(i))
                    raise ValueError('Duplicate sample {:d} already exists'.format(i))
                optimal_samples.append(i)
        else:
            print('NotImplementedError: UQRA.OptimalDesign.get_D_Optimality_samples: algorithm {:s} not implemented...'.format(algorithm))
            raise NotImplementedError
        optimal_samples = self._list_diff(optimal_samples, self.optimal_samples)
        return optimal_samples

    def get_S_Optimality_samples(self, n, algorithm):
        """
        Optimal design with S-optimality
        Arguments:
            n: int, number of samples to be returned
            Algorithm: str, algorithms used to initialize S-optimal samples
        Algorithms reference:  
           Bos, L., De Marchi, S., Sommariva, A., & Vianello, M. (2010). Computing multivariate Fekete and Leja points by numerical linear algebra. SIAM Journal on Numerical Analysis, 48(5), 1984-1999.
        """
        if algorithm.upper() == 'GREEDY':

            candidate_samples = copy.deepcopy(self.candidate_samples)
            optimal_samples   = copy.deepcopy(self.optimal_samples)
            for _ in tqdm(range(n), ascii=True, desc="          [Greedy S]",ncols=80):
                ## find the next optimal index from Q which is not currently selected
                candidate_samples = self._list_diff(candidate_samples, optimal_samples)
                assert self._check_complement(optimal_samples, candidate_samples)
                X_cand  = self.X[candidate_samples ,:]
                X_sltd  = self.X[optimal_samples,:]
                ## calculate (log)S values for each row in X_cand together with X_sltd
                optimality_values = self._greedy_update_S_Optimality_full(X_sltd, X_cand)
                if len(optimality_values) != len(candidate_samples):
                    raise ValueError('Expecting {:d} S values, but {:d} given'.format(len(candidate_samples), len(optimality_values)))
                i = candidate_samples[np.argmax(optimality_values)] ## return the indices with largest s-value in original matrix Q
                ## check if this index is already selected
                if i in optimal_samples:
                    print('Row {:d} already selected'.format(i))
                    raise ValueError('Duplicate sample {:d} already exists'.format(i))
                optimal_samples.append(i)
        else:
            print('NotImplementedError: UQRA.OptimalDesign.get_D_Optimality_samples: algorithm {:s} not implemented...'.format(algorithm))
            raise NotImplementedError
        optimal_samples = self._list_diff(optimal_samples, self.optimal_samples)
        return optimal_samples

    def _initial_samples_rrqr(self, n):
        """
        Return rows corresponding to largest absolute singular values in design matrix X based on RRQR 

        Arguments:
            n: set of selected indices 
        """
        n = helpers.check_int(n)
        X = self.X[self.candidate_samples, :]
        if n > min(X.shape):
            raise ValueError('Can only return at most rank(X) samples')
        # print('   - [Initialization (RRQR)]'.ljust(80, '#'))
        _,_,Pivot = sp.linalg.qr(X.T, pivoting=True)
        optimal_samples = [self.candidate_samples[i] for i in Pivot[:n]]

        # ## each QR iteration returns rank(X) samples, which is min(candidate samples, n_features)
        # X_rank = min(len(candidate_samples), self.X.shape[1])
        # n_iterations = math.ceil(n/X_rank)
        # for i_iteration in tqdm(range(n_iterations), ascii=True, desc='   - [RRQR]',ncols=80):
            # ## remove the new selected indices from candidate 
            # X = self.X[candidate_samples, :]
            # ## number of samples to be generated in this iteration, X_rank except for the last iteration
            # n_samples = X_rank if i_iteration < n_iterations-1 else int(n-X_rank * i_iteration)
            # if X.shape[0] <= n_samples:
                # optimal_samples_ = candidate_samples
                # if X.shape[0] < n_samples:
                    # print('\n'+'Warning: UQRA.Experiment.OptimalDesign._initial_samples_rrqr: requested {:d} optimal samples but only {:d} candidate samples available, returning all candiate samples'.format(n_samples, X.shape[0]))
            # else:
                # ## update candidate design matrix, note that X row_candidate is the indices in the original design matrix X
                # _,_,Pivot = sp.linalg.qr(X.T, pivoting=True)
                # ## Pivot[i] is the index in X corresponding the largest |singular value|
                # ## need to find its corresponding index in the original matrix X
                # optimal_samples_ = [candidate_samples[i] for i in Pivot[:n_samples]]
            # optimal_samples  = self._list_union(optimal_samples,  optimal_samples_)
            # candidate_samples= self._list_diff(candidate_samples, optimal_samples_)
        # if len(optimal_samples)!= n:
            # raise ValueError('Requesting {:d} new samples but {:d} returned'.format(n, n0))
        return optimal_samples
        
    def _initial_samples_greedy_tsm(self, n, optimality, optimal_samples=None):
        """
        return initial samples selected based with truncated square matrices
        """
        n = helpers.check_int(n) 
        if n > min(len(self.candidate_samples), self.X.shape[1]):
            raise ValueError('Can only return at most rank(X) samples')
        ## if no samples are selected currectly, then first sample is drawn randomly
        ## the rest n-1 samples are draw with greedy algorithm successfully
        if optimal_samples is None:
            optimal_samples = [np.random.randint(0, self.X.shape[0], size=1).item(),]
            n = n-1

        candidate_samples = copy.deepcopy(self.candidate_samples)
        optimal_samples   = copy.deepcopy(optimal_samples)

        for _ in tqdm(range(n), ascii=True, desc="   - [Initialization (TSM)-{:s}]".format(optimality),ncols=80):
        # for _ in range(n):
            ## find the next optimal index from Q which is not currently selected
            candidate_samples = self._list_diff(candidate_samples, optimal_samples)
            assert self._check_complement(optimal_samples, candidate_samples)
            X_cand  = self.X[candidate_samples ,:]
            X_sltd  = self.X[optimal_samples,:]
            ## calculate (log)S values for each row in X_cand together with X_sltd
            if optimality == 'S':
                optimality_values = self._greedy_update_S_Optimality_truncate(X_sltd, X_cand)
            elif optimality == 'D':
                optimality_values = self._greedy_update_D_Optimality_truncate(X_sltd, X_cand)
            else:
                print('   > UQRA {:s}-Optimal Design for TSM{:s} NOT implemented'.format(optimality))
                raise NotImplementedError

            if len(optimality_values) != len(candidate_samples):
                raise ValueError('Expecting {:d} S values, but {:d} given'.format(len(candidate_samples), len(optimality_values)))
            i = candidate_samples[np.argmax(optimality_values)] ## return the indices with largest s-value in original matrix Q
            ## check if this index is already selected
            if i in optimal_samples:
                print('Row {:d} already selected'.format(i))
                raise ValueError('Duplicate sample {:d} already exists'.format(i))
            optimal_samples.append(i)
        return optimal_samples

    def _greedy_update_S_Optimality_full(self, A, B):
        """
        Calculate S-value with matrix determinant update formula for each row element in B 
        Only for overdetermined system, i.e. A.T * A is not singular, i.e. n0 > p
        
        Arguments: 
            A: ndarray of shape(n0, p), selected
            B: ndarray of shape(n1, p), candidate

        Return: 
            log(S-values): S([A; r.T]), r is row in B
        """

        A = np.array(A, copy=False, ndmin=2)
        B = np.array(B, copy=False, ndmin=2)

        if A.shape[0] < A.shape[1]:
            raise ValueError('S-value updating formula only works for overdetermined system, however given {}'.format(A.shape))
        if A.shape[1] != B.shape[1]:
            raise ValueError('matrix A, B must have same number of columns')

        n0, p = A.shape
        n1, p = B.shape
        
        try:
            AAinv = np.linalg.inv(A.T.dot(A))  ## shape (p, p)
        except np.linalg.LinAlgError:
            u,s,v = np.linalg.svd(A.T.dot(A))
            print('singular value of A.T *A: {}'.format(s))

        d1 = np.log(1.0 + (B.dot(AAinv) * B).sum(-1)) ## (n1,)
        A_col_norm = np.linalg.norm(A, axis=0)
        d2 = np.sum(np.log(A_col_norm**2 + B**2), axis=1) ## (n1)

        res = d1 - d2
        return np.squeeze(res)

    def _greedy_update_D_Optimality_full(self, A, B):
        """
        Calculate S-value with matrix determinant update formula for each row element in B 
        Only for overdetermined system, i.e. A.T * A is not singular, i.e. n0 > p
        
        Arguments: 
            A: ndarray of shape(n0, p), selected
            B: ndarray of shape(n1, p), candidate

        Return: 
            log(S-values): S([A; r.T]), r is row in B
        """

        A = np.array(A, copy=False, ndmin=2)
        B = np.array(B, copy=False, ndmin=2)

        if A.shape[0] < A.shape[1]:
            raise ValueError('Updating formula for S-value only works for overdetermined system')
        if A.shape[1] != B.shape[1]:
            raise ValueError('matrix A, B must have same number of columns')

        n0, p = A.shape
        n1, p = B.shape
        
        try:
            AAinv = np.linalg.inv(A.T.dot(A))  ## shape (p, p)
        except np.linalg.LinAlgError:
            u,s,v = np.linalg.svd(A.T.dot(A))
            print('singular value of A.T *A: {}'.format(s))

        res = 1.0 + (B.dot(AAinv) * B).sum(-1) ## (n1,)
        return np.squeeze(res)

    def _greedy_update_S_Optimality_truncate(self, X0, X1):
        """
        Calculate the S-value (without determinant) of a candidate vector w.r.t selected subsets
        when the current selection k < p (eqn. 3.18)

        Arguments:
        X0 -- selected submatrix of shape (k,p), A
        X1 -- candidate matrix of shape (n-k, p), B

        Return:
        S value without determinant (eqn. 3.18)

        """
        X0 = np.array(X0, copy=False, ndmin=2)
        X1 = np.array(X1, copy=False, ndmin=2)

        if X0.shape[0] > X0.shape[1]:
            raise ValueError('Updating formula for S-value only works for underdetermined system')
        if X0.shape[1] != X1.shape[1]:
            raise ValueError('matrix A, B must have same number of columns')
        k,p     = X0.shape
        n_k, p  = X1.shape
        # start = time.time()
        A = copy.copy(X0[0:k, 0:k])                         ## shape (k, k)
        try:
            AAinv = np.linalg.inv(A.T.dot(A))               ## shape (k, k)
        except np.linalg.LinAlgError:
            u,s,v = np.linalg.svd(A.T.dot(A))
            print('singular value of A.T *A: {}'.format(s))

        R = copy.copy(X1[:, 0:k])                           ## shape (n-k, k)
        B = AAinv.dot(R.T)                                  ## shape (k, n-k)
        c = copy.copy(X0[0:k, k]).reshape((k,1))            ## shape(k, 1)  column vector
        g = AAinv.dot(A.T).dot(c)                           ## shape (k, 1)
        gamma = X1[:,k]                                     ## shape (n-k,) 
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
            with np.errstate(divide='ignore', invalid='ignore'):
                d3 = np.log(d3)
                d3 = np.nan_to_num(d3, nan=-np.inf)
                delta = d1 + d3 - d2 - d4
        else:
            ## all d3 < 0. then take the negative of all d3 and return the smallest s value
            d3 = np.log(abs(d3))
            delta = -(d1 + d3 - d2 - d4)
        return delta

    def _greedy_update_D_Optimality_truncate(self, X0, X1):
        """
        Calculate the S-value (without determinant) of a candidate vector w.r.t selected subsets
        when the current selection k < p (eqn. 3.18)

        Arguments:
        X0 -- selected submatrix of shape (k,p), A
        X1 -- candidate matrix of shape (n-k, p), B

        Return:
        S value without determinant (eqn. 3.18)

        """
        X0 = np.array(X0, copy=False, ndmin=2)
        X1 = np.array(X1, copy=False, ndmin=2)

        if X0.shape[0] > X0.shape[1]:
            raise ValueError('Updating formula for S-value only works for underdetermined system')
        if X0.shape[1] != X1.shape[1]:
            raise ValueError('matrix A, B must have same number of columns')
        k,p     = X0.shape
        n_k, p  = X1.shape
        # start = time.time()
        A = copy.copy(X0[0:k, 0:k])                         ## shape (k, k)
        try:
            AAinv = np.linalg.inv(A.T.dot(A))               ## shape (k, k)
        except np.linalg.LinAlgError:
            u,s,v = np.linalg.svd(A.T.dot(A))
            print('singular value of A.T *A: {}'.format(s))

        R = copy.copy(X1[:, 0:k])                           ## shape (n-k, k)
        B = AAinv.dot(R.T)                                  ## shape (k, n-k)
        c = copy.copy(X0[0:k, k]).reshape((k,1))            ## shape(k, 1)  column vector
        g = AAinv.dot(A.T).dot(c)                           ## shape (k, 1)
        gamma = X1[:,k]                                     ## shape (n-k,) 
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
                Alpha.extend(Alpha_)
        Alpha = np.array(Alpha)
        if Alpha.shape != (n_k,):
            print(Alpha)
            raise ValueError('Expecting Alpha shape to be ({},), but {} given'.format(n_k, Alpha.shape))
        d1 = np.log(1.0 + (R * B.T).sum(-1))  ## shape (n-k, )
        A_norms = np.linalg.norm(A, axis=0)
        d4 = np.squeeze(c.T.dot(c) + gamma**2)  ## shape(n-k, )
        d3 =  d4 - Alpha 
        d4 = np.log(d4)

        if np.any(d3 > 0):
            ## d1, d4 > 0. If there exist at least one d3 > 0, set negative d3 to -inf
            with np.errstate(divide='ignore', invalid='ignore'):
                d3 = np.log(d3)
                d3 = np.nan_to_num(d3, nan=-np.inf)
                delta = d1 + d3 - d4
        else:
            ## all d3 < 0. then take the negative of all d3 and return the smallest s value
            d3 = np.log(abs(d3))
            delta = -(d1 + d3 - d4)
        return delta

    def _check_complement(self, A, B, U=None):
        """
        check if A.union(B) = U and A.intersection(B) = 0
        """
        A = set(A)
        B = set(B)
        U = set(np.arange(self.X.shape[0])) if U is None else set(U)
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
            try:
                ls1.remove(element)
            except ValueError:
                pass
        return ls1

    def _list_inter(self, ls1, ls2):
        """
        return common elements between ls1 and ls2 
        """
        ls = list(set(ls1).intersection(set(ls2)))
        return ls


### ------------------- outdated functions -----------------------
    def _get_samples_svalue(self,n, candidate_samples, optimal_samples):
        """
        return row indices for quasi optimal experimental design based on fast greedy initialization 

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
            assert self._check_complement(optimal_samples, candidate_samples)
            X_cand  = self.X[candidate_samples_ ,:]
            X_sltd  = self.X[optimal_samples_all,:]
            ## calculate (log)S values for each row in X_cand together with X_sltd
            optimality_values = self._cal_svalue(X_cand,X_sltd)
            if len(optimality_values) != len(candidate_samples_):
                raise ValueError('Expecting {:d} D-Optimality values, but {:d} given'.format(len(candidate_samples_), len(optimality_values)))
            i = candidate_samples_[np.argmax(optimality_values)] ## return the indices with largest s-value in original matrix Q
            ## check if this index is already selected
            if i in optimal_samples_all:
                print('Row {:d} already selected'.format(i))
                raise ValueError('Duplicate sample {:d} already exists'.format(i))
            optimal_samples_.append(i)
        return optimal_samples_

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
        optimality_values  = self._cal_svalue(Q_cand,Q_sltd)
        if len(optimality_values) != len(row_candidate):
            raise ValueError('Expecting {:d} S values, but {:d} given'.format(len(row_candidate), len(optimality_values)))
        i = row_candidate[np.argmax(optimality_values)] ## return the indices with largest s-value in original matrix Q
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
        optimality_values = d1 - d2
        return np.squeeze(optimality_values)

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

