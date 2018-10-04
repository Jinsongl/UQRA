#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import chaospy as cp
from doe_generator import *
import numpy as np
import numpy.random as nrand
import numpy.linalg as nla
import matplotlib.pyplot as plt
from doe_quasi_optimal import get_quasi_optimal
from test_func import gaussian_peak 
from test_func import plot_func
from statsmodels.distributions.empirical_distribution import ECDF
def l(u):

    c0,c1,c2 = 2.515517, 0.802853, 0.010328
    d1,d2,d3 = 1.432788, 0.189269, 0.001308
    # print(u.shape)
    b = np.minimum(u,1-u) **2
    t = np.sqrt(-np.log(b))
    l1 = np.sign(u-0.5)
    l2 = t - (c0+c1*t+c2*t**2) / (1 + d1*t + d2*t**2 + d3*t**3)

    return l1*l2
def h(u):
    y = -np.log(1-u)
    return y.reshape(u.shape)


dist_k        = cp.Exponential(1) 
dist_zeta   = cp.Normal()
dist_u      = cp.Uniform(0,1)
orthPoly,norms = cp.orth_ttr(5,dist_zeta,retall=True)

coord_zeta, w_zeta = cp.generate_quadrature(10,dist_zeta,rule="G")
coord_u, w_u = cp.generate_quadrature(5,dist_u,rule="e")
y_zeta = dist_k.inv(dist_zeta.cdf(coord_zeta))[0]
y_u = h(coord_u).T
print(y_zeta.shape)
print(y_u.shape)

f_hat_zeta,coefs = cp.fit_quadrature(orthPoly,coord_zeta,w_zeta,y_zeta,norms=norms,retall=True)
# print(f_hat_zeta)
print(cp.around(f_hat_zeta, 5))
# print(f_hat_zeta(2))
f_hat_u,coefs = cp.fit_quadrature(orthPoly,l(coord_u),w_u,y_u,norms=norms,retall=True)
f_hat_u = f_hat_u[0]
print(cp.around(f_hat_u, 5))
print(f_hat_u(2))



# # Validation

samples_k = dist_k.sample(1000,'R')
samples_zeta = dist_zeta.sample(1000,'R')
samples_zeta = samples_zeta.reshape(1, len(samples_zeta))
samples_u = dist_u.sample(1000,'R')
samples_u = samples_u.reshape(1, len(samples_u))
samples_k_zeta = f_hat_zeta(*samples_zeta)
samples_k_u = f_hat_u(*samples_u)

ecdfk = ECDF(samples_k)
ecdfk_zeta = ECDF(samples_k_zeta)
ecdfk_u = ECDF(samples_k_u)
plt.plot(ecdfk.x, ecdfk.y)
plt.plot(ecdfk_zeta.x, ecdfk_zeta.y)
plt.plot(ecdfk_u.x, ecdfk_u.y)
plt.show()

