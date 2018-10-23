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
import numpy as np
import matplotlib.pyplot as plt

def get_mu(x):
    """
    samples to calculate mu
    Arguments:
    x of shape(ndim,) or (ndim,nsamples), ndim == 2 here
    Return:
    mu: a number or of shape(nsamples,)
    """
    x = np.array(x)
    if x.ndim == 1:
        mu = 2*np.log(x[0]) + np.sqrt(x[1]) + 0.05*x[0]*x[1] 
    else:
        mu = 2*np.log(x[0,:]) + np.sqrt(x[1,:]) + 0.05*x[0,:]*x[1,:] 
    return mu

def get_sigma(x):
    """
    samples to calculate sigma
    Arguments:
    x of shape(ndim,) or (ndim,nsamples), ndim == 2 here
    Return:
    mu: a number or of shape(nsamples,)
    """
    x = np.array(x)
    if x.ndim == 1:
        sigma = 0.2*np.log(x[0] + 0.5*x[1] + 1)
    else:
        sigma = 0.2*np.log(x[0,:] + 0.5*x[1,:] + 1)
    return sigma

dist_x1 = cp.Weibull(2,9.5) 
dist_x2 = cp.Lognormal(0.9,0.5)

nRepeat =  100
nShortterm = int(1e2)
dist_J = cp.J(dist_x1, dist_x2)

# mu_x = get_mu(samples_x)
# sigma_x = get_sigma(samples_x)

long_term_max_y = []
short_term_xy = []
for irepeat in range(nRepeat):
    ilong_term_max_y = -np.inf
    ilong_term_max_x = 0
    short_term_max_y = []
    samples_x = dist_J.sample(nShortterm)
    print('\r Repeat {:d} / {:d}'.format(irepeat, nRepeat), end='')
    for i, isample_x in enumerate(samples_x.T):
        # print(get_mu(isample_x))
        # print(get_sigma(isample_x))
        iresponse = np.random.normal(get_mu(isample_x),get_sigma(isample_x),1000)
        imax = np.max(iresponse)
        short_term_max_y.append(imax)
        # print(imax)
        if imax > ilong_term_max_y:
            ilong_term_max_y = imax
            ilong_term_max_x = i
            # print(ilong_term_max_x)
            # long_term_max_y.append(isample_x.tolist().append(imax))

    response_max_ = samples_x[:,ilong_term_max_x].tolist()
    response_max_.append(ilong_term_max_y)
    long_term_max_y.append(response_max_)
    short_term_max_y = np.array(short_term_max_y).reshape(1,nShortterm)
    short_term_xy.append(np.vstack((samples_x,short_term_max_y)))
print('\n')

long_term_max_y = np.array(long_term_max_y)
print(long_term_max_y.shape)
# max_idx = np.argmax(short_term_xy[2,:])
fig, axes = plt.subplots(3,3)
print(axes.shape)
for i in range(3):
    for j in range(3):
        if not (i == 2 and j == 2):
            data2plot = short_term_xy[i*2 +j]
            max_idx = np.argmax(data2plot[2,:])
            axes[i,j].scatter(data2plot[0,:],data2plot[1,:])
            axes[i,j].plot(data2plot[0,max_idx], data2plot[1,max_idx],'ro')
            axes[i,j].set_xlim(0., 30)
            axes[i,j].set_ylim(0., 15)
            axes[i,j].set_xlabel('$x_1$')
            axes[i,j].set_ylabel('$x_2$')
        
axes[2,2].scatter(long_term_max_y[:,0],long_term_max_y[:,1],c=long_term_max_y[:,2],cmap='jet')
axes[2,2].set_xlim(0., 30)
axes[2,2].set_ylim(0., 15)

plt.show()




