#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import numpy as np
import numpy.linalg as nla
import numpy.random as nrand 
import copy
import progressbar

def optimality_A():
    pass

def optimality_BA():
    pass

def optimality_BD():
    pass
def optimality_C():
    pass
def optimality_D():
    pass
def optimality_E():
    pass
def optimality_G():
    pass
def optimality_S():
    pass
def quasi_optimal():
    pass
def trace():
    pass

def get_quasi_optimal(m,A,I=None,is_orth=False):
    """
    return a list of containing indices of quasi optimal row selection matrix S 
    based on fast greedy algorithm and candidates design matrix A.

    Arguments:
    m -- size of quasi optimal subset
    A -- design matrix with candidates samples of shape (M,p)
         M: number of samples, p: number of features
    I -- indes list of corresponding row selection matrix of length m
        if I is None, an empty list will be created first and m items will be appended 
        Otherwise, additional (m-m0) items (row index in design matrix A) will be appended 
    is_orth -- Boolean indicating if the basis space is orthogonal


    Returns:
    row selection matrix I of shape (m, M)

    """
    m = int(m)
    assert m > 0, "At least one sample in the designed experiemnts"
    M,p = A.shape
    assert m >= p, "quasi optimal sebset are design for overdetermined problem only"
    # assert m < 2*p, 'Quasi optimal are disigned to choose ~p design points, too many asking'
    # print('>>'*20)
    print('\tQuasi-Optimal Experiment Design')
    print('\t>>>','*'*40)
    print("\tNumber of design point:\t{:2d} \n\tNumber of samples:\t{:2d} \n\tNumber of features:\t{:2d}".format(m,A.shape[0],A.shape[1]))
    print('\t>>>','*'*40)
    # (Q,R) = (A, _ )if is_orth else nla.qr(A, mode='complete')
    (Q,R) = (A, _ )if is_orth else nla.qr(A)
    print('\tComplete QR factorization of Design matrix A. \n\t  Q.shape = {0}, R.shape={1}'.format(Q.shape, R.shape))
    print('\t>>>','*'*40)
    print('\tSearching for design points based on S-value')

    I = [nrand.randint(0,M)] if I is None else I
    m1 = len(I) 
    print('\tRandom Initialize...')
    # widgets = ['\tProcessed: ', progressbar.Counter(), ' \tSelected: {:5d}'.format(I[-1]), ' (', progressbar.Timer(), ')']
    # pbar = progressbar.ProgressBar(widgets=widgets)

    # for ipbar in pbar((i for i in range(m-m1))):
    print('\tProcessed #:{:3d} out of {:3d}'.format(len(I), m), ';\tSelected: {:8d}'.format(I[-1]))
    while m1 < m:
        i = find_next(I,Q)
        I.append(i)
        print('\tProcessed #:{:3d} out of {:3d}'.format(len(I), m), ';\tSelected: {:8d}'.format(I[-1]))
        m1 = m1 + 1
    I = sorted(I)
    print('\tQuasi-Optimal Experiment design done!')
    print('\tSelected subset indice (first 10): ', I[:10] if len(I) > 10 else I)
    return I 



def find_next(I,Q):
    """
    find the next quasi optimal sample

    Arguments:
    I -- index list containing selected rows from design matrix 
    Q -- QR factorization of design matrix of M samples ()

    Return:
    i -- integer, index with maximum svalue
    """
##  Find the index candidate set to chose from (remove those in I from all (0-M))
    I_candi = list(set(range(Q.shape[0])).difference(set(I)))
    Q_candi = Q[np.array(I_candi, dtype=np.int32),:]
    Q_selec = Q[np.array(I,       dtype=np.int32),:]
    
    svalues = cal_svalue(Q_candi,Q_selec)
    assert(svalues.shape[0] == len(I_candi))
    # I_candi_sorted = list(map(lambda i: I_candi[i], np.argsort(svalues)))
    # print('\tSorted S-value indices (increasing)', I_candi_sorted)
    i = I_candi[np.argmax(svalues)]
    return i


def cal_svalue(R,A):
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
        svalues = cal_logsvalue_under(R,A)
    else:
        svalues = cal_logsvalue_over(R,A)

    return svalues
    

def cal_logsvalue_over(R,A):
    """
    Calculate the S value (without determinant) of candidate vectors w.r.t selected subsets
    when the current selection k >= p (eqn. 3.16)

    Arguments:
    R -- candidate row vector of shape (number of candidates,p)
    A -- selected subsets matrix of shape (k,p)

    Return:
    d -- log svalue without determinant (eqn. 3.16)

    """
    AAinv = nla.inv(np.dot(A.T,A))
    A_l2 = nla.norm(A, axis=0).reshape(-1,1)
    svalues_log = np.zeros(R.shape[0])
    for i, r in enumerate(R):
        r = r.reshape((len(r),1))
        assert(len(r) == A.shape[1])
        with np.errstate(invalid='ignore'):
            d1 = np.log(1 + np.dot(r.T, np.dot(AAinv, r)))
            d2 = np.log(np.prod(A_l2**2 + r.T**2))
        svalues_log[i] = d1 - d2
    return svalues_log

    
def cal_logsvalue_under(R,A):
    """
    Calculate the log S-value (without determinant) of a candidate vector w.r.t selected subsets
    when the current selection k < p (eqn. 3.18)

    Arguments:
    R -- candidate row vector of shape(number of candidates,p)
    A -- selected subsets matrix of shape (k,p)

    Return:
    log(d) -- svalue without determinant (eqn. 3.18)

    """
    k = A.shape[0]
    A = copy.copy(A[:,0:k])
    R = copy.copy(R[:,0:k+1])
    svalues_log = np.zeros(R.shape[0])
    AAinv = nla.inv(np.dot(A.T,A))
    A_l2 = nla.norm(A, axis=0).reshape(-1,1)


    for i, r in enumerate(R):
        c = r[0:k].reshape((k,1))
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

        d = d1 + d3 - d2 - d4
        svalues_log[i] = d 
    return svalues_log



def main():
    x = nrand.randn(100,1)
    A = [np.ones(x.shape),x,x**2,x**3]
    A = np.squeeze(np.asarray(A)).T
    m = 6
     
    I = get_quasi_optimal(m,A)
    print(I)



