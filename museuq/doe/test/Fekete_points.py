#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""
From theorem 4.1, the n-point set determined by Svalue, where the matrix A is 
contructed using orthonormal basis of degree up to (n-1) with measure u(x) in x[-1,1]. 

Asymptotic distribution of the empirical measure vn of the quasi-optimal set also follows
the arcsin distribution in [-1,1]
"""
import os
import sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chaospy as cp
from doe_quasi_optimal import get_quasi_optimal
from test_func import *
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt

def arcsin_cdf(x):
    return 1.0/np.pi * (np.arcsin(x) + 0.5 * np.pi)

x = np.linspace(-1,1,1000)
n_candidates = 100000
dist = cp.Uniform(-1,1)
x_candidates = dist.sample(n_candidates, rule='R')

print('Normed=True')
poly_orth, norm = cp.orth_ttr(30,dist,normed=True,retall=True)
print(poly_orth[:2])
X_candidates = poly_orth(x_candidates).T
XX = np.dot(X_candidates.T, X_candidates)
print(XX[:4,:4]/n_candidates)
print(norm)


poly_orth, norm = cp.orth_ttr(30,dist,retall=True)
print(poly_orth[:2])
X_candidates = poly_orth(x_candidates).T
XX = np.dot(X_candidates.T, X_candidates)
print(XX[:4,:4]/n_candidates)
print(norm)

# indice_quasi_opt = get_quasi_optimal(100, X_candidates)
# subset_quasi_opt = np.sort(x_candidates[indice_quasi_opt])
# ecdf = ECDF(subset_quasi_opt)
# fig = plt.figure()
# plt.plot(ecdf.x, ecdf.y, label='Legendre')
# plt.plot(x,arcsin_cdf(x) , label='Asymptotic Distribution')
# plt.legend()
# plt.show()
