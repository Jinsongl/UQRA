#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
dist_zeta = cp.Iid(cp.Normal(0,1),2)
orthPoly = cp.orth_ttr(5,dist_zeta)

samp_zeta = np.genfromtxt("MC200.csv", delimiter=',')
samp_zeta = samp_zeta[:,3:5].T

feval = np.genfromtxt("col1.csv",delimiter=',')[:,4]
f_hat = cp.fit_regression(orthPoly,samp_zeta,feval)

# print f_hat(0,0.2)
## Prediction
numRepeat = 2
numSamp = int(1e4)
# u =[np.zeros((numSamp,2))] * numRepeat
# fu=[np.zeros((numSamp,1))] * numRepeat
plt.figure()
for i in range(numRepeat):
    u = dist_zeta.sample(numSamp).T
    fu=[]
    for j in range(numSamp):
        fu.append(f_hat(u[j][0],u[j][1]))
    
    ecdf = ECDF(fu)
    plt.plot(ecdf.x,ecdf.y)
plt.yscale('log')
plt.show()

