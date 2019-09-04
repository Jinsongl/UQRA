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
import chaospy as cp
import matplotlib.pyplot as plt

dist_x = cp.Uniform(2,3)
moms_real = np.array([dist_x.mom(i) for i in np.arange(7)])
print(moms_real)
moms_pce = np.squeeze(np.load('Uniform_Legendremoments.npy'))
# moms_pce = np.squeeze(np.load('Uniform_Hermite.npy'))
print(moms_pce.shape)
# plt.plot(meta_orders, np.squeeze(metric_moms[0,:,0,:]))
moms_pce_mean = np.mean(moms_pce, axis=1)
print(1-moms_pce_mean/ moms_real)

# moms_error_per_set = []
# for i in np.arange(moms_pce.shape[1]):
    # # print(moms_pce[:,i,:])
    # moms_error_per_set.append(abs(1- np.squeeze(moms_pce[:,i,:])/ moms_real))
# print(moms_error)
# print(np.array(moms_error_per_set).shape)

moms_pce_mean_error = abs(1- moms_pce_mean/ moms_real)
plt.figure()
for i in np.arange(1,moms_pce_mean_error.shape[1]):
    plt.semilogy(np.arange(1,13), moms_pce_mean_error[:,i], label=r'M{}'.format(i+1))
    plt.legend()
    plt.savefig('Uniform_Legendre.eps')

