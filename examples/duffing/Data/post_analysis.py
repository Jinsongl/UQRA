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
import matplotlib.pyplot as plt
import sklearn.gaussian_process as skgp
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
filename = 'training_set_'
fig = plt.figure()
datax = []
datay = []
for i in np.arange(10):
    _datax = np.load(filename+'x_{:d}.npy'.format(i))
    _datay = np.load(filename+'y_{:d}.npy'.format(i))
    datax.append(np.squeeze(_datax[0].T))
    y = np.max(_datay[:,:,:,1], axis=2)
    datay.append(np.squeeze(y))
datax = np.array(datax)
datay = np.array(datay)
# plt.plot(datax.T,datay.T,'-.o')
# plt.xlabel(r'Spectrum parameter $c$')
# plt.ylabel(r'Max$(Y_t)$')
# plt.savefig('Max_vs_c_observation.eps')


# X = datax.reshape((datax.size,1 ))
# y = datay.reshape((datay.size,1 ))
X = np.mean(datax,axis=0)[:,np.newaxis]
y = np.mean(datay,axis=0)
print(X.shape,y.shape)
# # kernel = DotProduct() + WhiteKernel()
# kernel = RBF()
# gpr = skgp.GaussianProcessRegressor(kernel=kernel,random_state=0).fit(datax[0,:], datay[0,:])
# # gpr.score(X, y) 

# # print(gpr.score(datax, datay) )
# # print(gpr.predict(datax) )
# # print(datay)



plt.figure(0)
plt.scatter(datax,datay,c='b',s=50,zorder=10,edgecolors='b',label=r'$y_i$')
kernel = 1.0 * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 3)) \
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+2))
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0).fit(X, y)

X_ = np.linspace(0, 25, 100)
y_mean, y_cov = gp.predict(X_[:, np.newaxis], return_cov=True)
print(y_mean.shape)
plt.plot(X_, y_mean, 'k', lw=3, zorder=9,label=r'$\tilde{M}^{GP}$')
plt.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
                 y_mean + np.sqrt(np.diag(y_cov)),
                 alpha=0.5, color='k')

# plt.plot(X_, 0.5*np.sin(3*X_), 'r', lw=3, zorder=9)
plt.scatter(X[:, 0], y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0),label=r'$E[y]$')
# plt.title("Initial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s"
          # % (kernel, gp.kernel_,
             # gp.log_marginal_likelihood(gp.kernel_.theta)))
plt.tight_layout()

plt.legend()


plt.show()


