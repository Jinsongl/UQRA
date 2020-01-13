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

class OptimalDesign(ExperimentalDesign):
    """ Quasi-Optimal Experimental Design and Optimal Design"""

    def __init__(self, opt_criteria,*args, **kwargs):
        """
        Optimal/Quasi Optimal Experimental design:
        Arguments:
        n: int, number of samples 
        opt_criteria: optimal design criteria
        """
        super().__init__(*args, **kwargs)
        self.criteria   = opt_criteria 
        self.filename   = '_'.join(['DoE', 'Opt'+self.criteria.capitalize() + num2print(self.n_samples)])

    def __str__(self):
        return('Optimal Criteria: {:<15s}, num. samples: {:d} '.format(self.criteria, self.n_samples))

    def S(self,R,A):
        """
        Calculate the S-values of new matrix [A;r.T], where r is each row of R
        
        Arguments:
        R: Matrix where each row vector will be added to matrix A to calculate the its Svalue
        A: Matrix composed of selected vectors from all candidates, (number of selected, number of polynomial functions)

        Return:
        ndarray (n,) containing s-values from each row of R
        
        """
        k,p = A.shape

        if k < p :
            svalues = self._cal_logsvalue_under(R,A)
        else:
            svalues = self._cal_logsvalue_over(R,A)

        return svalues

    @random_state
    def samples(self, A, *args, **kwargs):
        """
        Ax = Y
        A: Design matrix A(u) of shape(num_samples, num_features)

        Return:
            Experiment samples of shape(ndim, n_samples)
        """
        u    = kwargs.get('u', None) ## samples input of shape(ndim, num_samples)
        m, p = A.shape
        assert m > p, 'Number of candidate samples are expected to be larger than number of features'

        if self.criteria.upper() == 'S':
            """ Ax = Y """
            selected_indices   = kwargs.get('I', None)
            is_basis_orth      = kwargs.get('is_orth', False)
            selected_indices   = self._get_quasi_optimal(self.n_samples, A, selected_indices, is_basis_orth)
        elif self.criteria.upper() == 'D':
            """ D optimality based on rank revealing QR factorization  """
            Q, R, P = sp.linalg.qr(A.T, pivoting=True)
            selected_indices = P[:self.n_samples]
        else:
            pass

        self.A = A[selected_indices,:]
        if u is not None:
            self.u = u[selected_indices] if u.ndim == 1 else u[:, selected_indices]
        else:
            pass
        self.I = selected_indices

    def _cal_logsvalue_over(self, R, A):
        """
        Calculate the S value (without determinant) of candidate vectors w.r.t selected subsets
        when the current selection k >= p (eqn. 3.16) for each pair of (r, A)


        Arguments:
        R -- candidate matrix of shape (number of candidates, p), 
        A -- selected subsets matrix of shape (k,p)

        Return:
        log S value without determinant (eqn. 3.16)

        """
        AAinv   = LA.inv(np.dot(A.T,A))
        A_l2    = LA.norm(A, axis=0).reshape(1,-1) ## row vector
        svalues_log = [] 
        for r in R:
            r = r.reshape(1,-1) ## row vector
            with np.errstate(invalid='ignore'):
                d1 = np.log(1 + np.dot(r, np.dot(AAinv, r.T)))
                d2 = np.log(np.prod(A_l2**2 + r**2))
            svalues_log.append(d1 - d2)
        return svalues_log

    def _cal_logsvalue_under(self, R, A):
        """
        Calculate the log S-value (without determinant) of a candidate vector w.r.t selected subsets
        when the current selection k < p (eqn. 3.18)

        Arguments:
        R -- candidate matrix of shape (number of candidates, p), 
        A -- selected subsets matrix of shape (k,p)

        Return:
        log S value without determinant (eqn. 3.18)

        """
        k,p = A.shape
        assert k < p
        A = copy.copy(A[:,0:k])
        R = copy.copy(R[:,0:k+1])
        svalues_log = [] 
        AAinv = LA.inv(np.dot(A.T,A))
        A_l2 = LA.norm(A, axis=0).reshape(1,-1)


        for r in R:
            c = r[0:k].reshape((k,1)) ## column vector
            gamma = r[k]
            r = copy.copy(c)

            b = np.dot(AAinv,r)
            g = np.dot(AAinv,np.dot(A.T,c))

            a1 = np.dot(c.T,A) + gamma * r.T
            a2 = np.identity(k) - np.dot(b,r.T)/(1 + np.dot(r.T,b))
            a3 = g + gamma *b
            a = np.squeeze(np.dot(a1,np.dot(a2,a3)))

            with np.errstate(invalid='ignore'):
                d1 = np.log(np.squeeze(1 + np.dot(r.T, b)))
                d2 = np.sum(np.log(A_l2**2 + r.T**2))
                d3 = np.log(np.squeeze(np.dot(c.T,c) + gamma**2 - a))
                d4 = np.log(np.squeeze(np.dot(c.T,c) + gamma**2))
            svalues_log.append(d1 + d3 - d2 - d4)
        return svalues_log

    def _greedy_find_next_point(self, I, Q):
        """
        find the next quasi optimal sample

        Arguments:
        I -- list containing selected row indices from candidate design matrix A
        Q -- QR factorization of candidate design matrix A if basis is not orthogonal, otherwise is A

        Return:
        i -- integer, index with maximum svalue
        """
        ##  Find the index candidate set to chose from (remove those in I from all (0-M))
        I_left   = list(set(range(Q.shape[0])).difference(set(I)))
        Q_left   = Q[np.array(I_left, dtype=np.int32),:]
        Q_select = Q[np.array(I,      dtype=np.int32),:]
        svalues  = self.S(Q_left,Q_select)
        assert(len(svalues) == len(I_left))
        # I_candi_sorted = list(map(lambda i: I_left[i], np.argsort(svalues)))
        # print(u'\tSorted S-value indices (increasing)', I_candi_sorted)
        i = I_left[np.argmax(svalues)]
        return i

    def _get_quasi_optimal(self,m,A,I=None,is_orth=False):
        """
        return row selection matrix S containing indices for quasi optimal experimental design
        based on fast greedy algorithm 

        Arguments:
        m -- size of quasi optimal subset
        A -- design matrix with candidates samples of shape (M,p)
             M: number of samples, p: number of features
        I -- indices, nt ndarray of shape (N,) corresponding row selection matrix of length m
            if I is None, an empty list will be created first and m items will be appended 
            Otherwise, additional (m-m0) items (row index in design matrix A) will be appended 
        is_orth -- Boolean indicating if the basis space is orthogonal

        Returns:
        row selection matrix I of shape (m, M)
        """
        m = int(m)
        assert m > 0, "At least one sample in the designed experiemnts"
        M,p = A.shape
        assert M >= p, "quasi optimal sebset are design for overdetermined problem only"
        # assert m < 2*p, 'Quasi optimal are disigned to choose ~p design points, too many asking'
        # print(u'>>'*20)
        # print(u'\tQuasi-Optimal Experiment Design')
        # print(u'\t>>>','*'*40)
        # print(u"\tNumber of design point:\t{:2d} \n\tNumber of samples:\t{:2d} \n\tNumber of features:\t{:2d}".format(m,A.shape[0],A.shape[1]))
        # print(u'\t>>>','*'*40)
        # (Q,R) = (A, _ )if is_orth else LA.qr(A, mode='complete')
        (Q,R) = (A, None ) if is_orth else LA.qr(A)
        # print(u'\tComplete QR factorization of Design matrix A. \n\t  Q.shape = {0}, R.shape={1}'.format(Q.shape, R.shape))
        # print(u'\t>>>','*'*40)
        # print(u'\tSearching for design points based on S-value')

        I = [np.random.randint(0,M)] if I is None else I ## random initialize for the first point
        m1 = len(I) 
        # print(u'\tRandom Initialize...')

        # for ipbar in pbar((i for i in range(m-m1))):
        # print(u'\tProcessed #:{:3d} out of {:3d}'.format(len(I), m), ';\tSelected: {:8d}'.format(I[-1]))

        pbar_x  = tqdm(range(m), ascii=True, desc="   - ")
        for _ in pbar_x:
        # while len(I) < m:
            i = self._greedy_find_next_point(I,Q)
            I.append(i)
            # print(u'\tProcessed #:{:3d} out of {:3d}'.format(len(I), m), ';\tSelected: {:8d}'.format(I[-1]))
        I = sorted(I)
        # print(u'\tQuasi-Optimal Experiment design done!')
        # print(u'\tSelected subset indice (first 10): ', I[:10] if len(I) > 10 else I)
        return np.array(I) 
