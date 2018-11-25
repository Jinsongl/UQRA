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
import numpy.linalg as LA
import scipy.stats as scistats
import matplotlib.pyplot as plt
import sklearn.gaussian_process as skgp
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

def _hermite_moment_nonlinear(u,mu,sigma,mu3,mu4,fisher=True):
    """
    Hermite moment models for nonlinear responses
    """

    if fisher:
        mu4 = mu4 + 3

    if mu4 > 3 :
        h3 = mu3/6
        h4 = (mu4-3)/24
        c4 = (np.sqrt(1+36*h4)-1)/18
        c3 = h3/(1+6*c4)
        k  = 1/(np.sqrt(1+2*c3**2+6*c4**2))
        x  = mu + k*sigma*(u+c3*(u**2-1) + c4*(u**3-3*u))

    elif mu4 < 3: 
        h4 = (mu4-3) / 24
        h3 = mu3 / 6
        a  = h3/(3*h4)
        b  = -1/(3*h4)
        k  = (b - 1 - a**2)**3
        c = 1.5*b*(a+u) - a**3
        x = mu + sigma * ((np.sqrt(c**2+k)+c)**1/3 - (np.sqrt(c**2+k)-c)**1/3 - a)
    else:
        x = mu + sigma * u

    return x



def _mean_up_crossing_rate(data, dt=1):
    """
    Calculate the mean upcrossing rate for data of shape [m,ndofs]
    """
    data_shape = data.shape
    data_mean = np.mean(data,axis=0) 
    data = data - data_mean
    data1 = data[:,0:-1]
    data2 = data[:,1:]
    cross_points = data1 * data2
    up_points = data2 - data1

    mucr = []
    for i in np.arange(data.shape[0]):
        candidatex = np.arange(data1.shape[1])
        condition1 = cross_points[i,:] < 0
        condition2 = up_points[i,:] > 0
        condition = condition1 & condition2
        up_cross_idx = candidatex[np.where(condition)]
        T = (up_cross_idx[-1] - up_cross_idx[0]) * dt ## Only counting complete circles
        mucr.append((len(up_cross_idx)-1)/T)
    return np.array(mucr).reshape(-1,1)




np.set_printoptions(precision=3)
filename = 'GQ/GQ9_'
datax = []
datay = []
time_ramp = 50 

for i in np.arange(10):
    _datax = np.load(filename+'x{:d}.npy'.format(i))
    _datay = np.load(filename+'y{:d}.npy'.format(i))
    datax.append(np.squeeze(_datax[0])) ## quadrature, (coords, weights)
    dt = _datay[0,0,1,0] - _datay[0,0,0,0]
    time_ramp_idx = int(time_ramp/dt)
    _datay = _datay[:,:,time_ramp_idx:,:]
    y_max = np.max(_datay[:,:,:,1], axis=2)
    y_std = np.std(_datay[:,:,:,1], axis=2)
    y_mu3 = scistats.skew(_datay[:,:,:,1], axis=2)
    y_mu4 = scistats.kurtosis(_datay[:,:,:,1], axis=2)
    y_mucr=_mean_up_crossing_rate(np.squeeze(_datay[:,:,:,1]), dt=dt)
    print('Skewness:{} '.format(y_mu3.T))
    print('Kurtosis:{} '.format(y_mu4.T))
    # print(_datay[0,0,1,0] , _datay[0,0,0,0])
    # datay.append([np.squeeze(y_max), np.squeeze(y_std), np.squeeze(y_mu3), np.squeeze(y_mu4)])
    datay.append([np.squeeze(y_max), np.squeeze(y_std), np.squeeze(y_mu3), np.squeeze(y_mu4), np.squeeze(y_mucr)])

datax = np.array(datax)
datay = np.array(datay)
print('*******')
print(np.array(datax).shape, np.array(datay).shape)


# filename = 'MC2500'
# datax = np.load('MC2500x.npy')
# datay = np.load('MC2500y.npy')
# X = datax.T
# y = np.max(datay[:,:,:,1],axis=2)
# print(X.shape,y.shape)
# plt.plot(datax.T,datay.T,'-.o')
# plt.xlabel(r'Spectrum parameter $c$')
# plt.ylabel(r'Max$(Y_t)$')
# plt.savefig('Max_vs_c_observation.eps')


# X = datax.reshape((datax.size,1 ))
# y = datay.reshape((datay.size,1 ))
X = np.mean(datax,axis=0).reshape(datax.shape[1],-1)
y_max = np.squeeze(datay[:,0,:])
# print(y_max)
y_max_mean= np.mean(y_max, axis=0)
# print(y_max_mean)
y_max = y_max - y_max_mean
# print(y_max)
y_max_std = np.std(y_max, axis=0,ddof=1 )
y_max_cov = np.cov(y_max.T)
eigenvalues,eigenvectors= LA.eig(y_max_cov)
idx = eigenvalues.argsort()[::-1]   
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]
evals_cumsum =np.cumsum(eigenvalues)
evals_perctage = evals_cumsum/np.sum(eigenvalues)
figs, axes = plt.subplots(2,1)
axes[0].plot(eigenvalues, '-ro')
axes[0].set_xlabel('n')
axes[0].set_ylabel('Eigenvalues')
ax = axes[0].twinx()
ax.plot(evals_perctage, '-bo')
ax.set_ylabel('Eigenvalues Percentage')
axes[1].plot(datax[0,:],eigenvectors) ## plot(M), column wise
plt.tight_layout()
plt.show()
# print(y_max_mean**2)
# print(y_max_mean.shape, y_max_std.shape)

# # kernel = DotProduct() + WhiteKernel()
# kernel = RBF()
# gpr = skgp.GaussianProcessRegressor(kernel=kernel,random_state=0).fit(datax[0,:], datay[0,:])
# # gpr.score(X, y) 

# # print(gpr.score(datax, datay) )
# # print(gpr.predict(datax) )
# # print(datay)



# plt.figure(0)
# plt.scatter(X,y,c='b',s=50,zorder=10,edgecolors='b',label=r'$y_i$')
# kernel = 1.0 * RBF(length_scale=1, length_scale_bounds=(1e-2, 5)) \
    # + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e+1))
# # gp = GaussianProcessRegressor(kernel=kernel, alpha=y_std).fit(X, y)
# gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0).fit(X, y)

# X_ = np.linspace(0, 25, 100)
# y_mean, y_cov = gp.predict(X_[:, np.newaxis], return_cov=True)
# print(y_mean.shape)
# plt.plot(X_, y_mean, 'k', lw=3, zorder=9,label=r'$\tilde{M}^{GP}$')
# plt.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
                 # y_mean + np.sqrt(np.diag(y_cov)),
                 # alpha=0.5, color='k')

# # plt.plot(X_, 0.5*np.sin(3*X_), 'r', lw=3, zorder=9)
# plt.scatter(X[:, 0], y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0),label=r'$E[y]$')
# # plt.title("Initial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s"
          # # % (kernel, gp.kernel_,
             # # gp.log_marginal_likelihood(gp.kernel_.theta)))

# plt.legend()


# plt.show()


