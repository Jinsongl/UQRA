#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""

import numpy as np
import numpy.random as nrand
import numpy.linalg as nla
import matplotlib.pyplot as plt
import sys
import progressbar
from mpl_toolkits.mplot3d import Axes3D
import chaospy as cp
from doe_quasi_optimal import get_quasi_optimal
from test_func import *


# # define parameters
train_size  = 10000
test_size   = 20000
subset_size_= np.arange(230,500,50)
x_dim       = 2
func_objs   = {}
func_objs['gaussian'] = (gaussian_peak,{'c':[1,1], 'w':[1,0.5]})
func_objs['product']  = (product_peak, {'c':[-3,2],'w':[0.5,0.5]})
func_objs['franke2d'] = (franke2d,[])
func_objs['ishigami'] = (ishigami,[])

# func_obj,c,w = func_objs['gaussian']
# func_obj,c,w = func_objs['product']
func_obj, params = func_objs['franke2d']
# func_obj = func_objs['ishigami']

# # define polynomial basis
dist = cp.Iid(cp.Uniform(-1,1),2)
poly_orth,_ = cp.orth_ttr(5,dist,normed=True,retall=True)

####*****************************************************************
####  Testing data 
####*****************************************************************
x_test  = dist.sample(test_size, rule='R') 
X_test  = poly_orth(*x_test).T 
# y_test  = func_obj(x_test.T,c,w)
y_test  = func_obj(x_test.T, params)
# y_test  = gaussian_peak(x_test.T,c,w)
# # y_test  = product_peak(x_test.T,c,w)
Y_test  = np.array(y_test).reshape((test_size,1))
error_all = []
for subset_size in subset_size_:
    print('**'*40)
    print('>>> Sample size:')
    print('\tSubset size: {0:8d}, Candidate size: {1:8d}, Test size: {2:8d}'.format(subset_size, train_size, test_size))
    print('>>> Objective function:')
    print('\t'+func_obj.__name__ + ' , Parameters:{0}'.format(params))
    print('**'*40)

####*****************************************************************
####  Crude Monte Carlo, Repeat multi times
####*****************************************************************
## Training with all samples
    error = 0.0
    print('Crude Monte Carlo with all samples...')
    x_train = dist.sample(train_size, rule='R') 
    X_train = poly_orth(*x_train).T

    # y_train = gaussian_peak(x_train.T,c,w)
    # y_train = product_peak(x_train.T,c,w)
    y_train = func_obj(x_train.T, params)
    Y_train = np.array(y_train).reshape((train_size,1))

    beta_hat = nla.inv(np.dot(X_train.T, X_train))
    beta_hat = np.dot(beta_hat, X_train.T)
    beta_hat = np.dot(beta_hat, Y_train)
    assert len(beta_hat) == len(poly_orth)

    Y_pred = np.dot(X_test, beta_hat)
    # error = nla.norm(Y_pred - Y_test)
    error = np.sqrt(np.mean((Y_pred - Y_test)**2))
    print('Done!')
## training with subset samples 
    n_repeat  = 50
    error_MC = 0.0
    print('Crude Monte Carlo with subset samples...')
# setup toolbar
    widgets = [progressbar.Percentage(),
               ' ', progressbar.Bar(),
               ' ', progressbar.ETA(),
               ' ', progressbar.AdaptiveETA()]
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=n_repeat)
    pbar.start()

    for i_repeat in range(n_repeat):
        x_train_MC  = dist.sample(subset_size, rule='R') 
        X_train_MC  = poly_orth(*x_train_MC).T
        # y_train_MC  = gaussian_peak(x_train_MC.T ,c,w)
        # y_train_MC  = product_peak(x_train_MC.T ,c,w)
        y_train_MC  = func_obj(x_train_MC.T, params)
        Y_train_MC  = np.array(y_train_MC).reshape((subset_size,1))

        beta_hat_MC = nla.inv(np.dot(X_train_MC.T, X_train_MC))
        beta_hat_MC = np.dot(beta_hat_MC, X_train_MC.T)
        beta_hat_MC = np.dot(beta_hat_MC, Y_train_MC)
        assert len(beta_hat_MC) == len(poly_orth)
        # make prediction
        Y_pred = np.dot(X_test, beta_hat_MC)
        error_MC = error_MC + np.sqrt(np.mean((Y_pred - Y_test)**2)/ (i_repeat+1))
        pbar.update(i_repeat + 1)
    pbar.finish()
    print('Done!')
####*****************************************************************
####  Quasi Monte Carlo, Sobol sequence 
####*****************************************************************
    error_QMC = 0.0
    print('Quasi Monte Carlo with sobol sequence...')
    x_train_QMC = dist.sample(subset_size, rule='S') 
    X_train_QMC = poly_orth(*x_train_QMC).T
    # y_train_QMC = gaussian_peak(x_train_QMC.T,c,w)
    # y_train_QMC = product_peak(x_train_QMC.T,c,w)
    y_train_QMC = func_obj(x_train_QMC.T, params)
    Y_train_QMC = np.array(y_train_QMC).reshape((subset_size,1))

    beta_hat_QMC = nla.inv(np.dot(X_train_QMC.T, X_train_QMC))
    beta_hat_QMC = np.dot(beta_hat_QMC, X_train_QMC.T)
    beta_hat_QMC = np.dot(beta_hat_QMC, Y_train_QMC)
    assert len(beta_hat_QMC) == len(poly_orth)


## solve Y = X*beta with all points 
# print('X_train.shape: {}, Y_train.shape:{}'.format(X_train.shape, Y_train.shape))

    Y_pred = np.dot(X_test, beta_hat_QMC)
    error_QMC = np.sqrt(np.mean((Y_pred - Y_test)**2))
    print('Done!')
# print ('Error with QMC: \t\t {:5.3e}'.format(nla.norm(error_QMC)))


####*****************************************************************
####  Quasi Monte Carlo, Sobol sequence 
####*****************************************************************
    error_quasi_opt = 0.0
    print('Quasi-optimal experimental design...')
# # Quasi Optimal experiment design 
    I = get_quasi_optimal(subset_size,X_train)

# solve Y = X*beta with quasi optimal design points
    X_quasi_opt = X_train[np.array(I, dtype=np.int32),:]
    Y_quasi_opt = Y_train[np.array(I, dtype=np.int32),:]
    beta_hat_quasi_opt = nla.inv(np.dot(X_quasi_opt.T, X_quasi_opt))
    beta_hat_quasi_opt = np.dot(beta_hat_quasi_opt, X_quasi_opt.T)
    beta_hat_quasi_opt = np.dot(beta_hat_quasi_opt, Y_quasi_opt)
    assert len(beta_hat_quasi_opt) == len(poly_orth)

    Y_pred = np.dot(X_test, beta_hat_quasi_opt)
    error_quasi_opt = np.sqrt(np.mean((Y_pred - Y_test)**2))
    print('Done!')
# print ('Error with Quasi Optimal: \t {:5.3e}'.format(nla.norm(error_quasi_opt)))


    print ('Error with MC (Averaged): \t {:5.3e}'.format(error_MC))
    print ('Error with QMC: \t\t {:5.3e}'.format(error_QMC))
    print ('Error with Quasi Optimal: \t {:5.3e}'.format(error_quasi_opt))
    print ('Error with all candidates: \t {:5.3e}'.format(error))
    error_all.append([error_MC, error_QMC, error_quasi_opt, error])

# f_hat = cp.fit_regression(poly_orth,x_train.T,Y_train)
# f_fit = np.array([f_hat(*val) for val in x_test])

# error2 = f_fit - Y_test
# print (nla.norm(error2))

np.save(func_obj.__name__, error_all)

fig = plt.figure()
plt.semilogy(subset_size_, np.array(error_all))
# ax.scatter(x_test[:1000,0],x_test[:1000,1],Y_test[:1000,0])
# ax.scatter(x_test[:1000,0],x_test[:1000,1],Y_pred[:1000,0])
plt.legend(['Averaged MC','Quasi MC', 'Quasi Optimal Design','All'])
plt.show()


# def mc_sampling(nsamples, nrepeat, poly, dist):
    # x = dist.sample(nsamples)
    # y = poly(*x.T).T


