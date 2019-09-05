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
import pandas as pd
import chaospy as cp
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
LO, UP = 0, 10
np.set_printoptions(precision=3)

def foo(x, seed=None, noise=True):
    x = np.array(x)
    np.random.seed(seed) if seed else np.random.seed()
    dist_zeta = cp.Normal()
    # noise = 0.1*(x**2+1) * dist_zeta.sample(len(x))
    # y = np.zeros(x.size)
    # idx = np.where((x>LO) * (x<UP))
    y0 = 0.5*(x**2 + 1)
    if noise:
        y0 = y0 + 0.1*(x**2+1) * dist_zeta.sample(x.size)
    # y = np.where((x>LO)*(x<UP),y0,0)
    return y0

# def build_surrogate_models(nmodels, orth_poly, x, w):


dist_zeta = cp.Normal()
dist_x = cp.Normal(5,1)
# dist_x = cp.Truncnorm(lo=LO, up=UP,mu=5,sigma=1)
seeds = np.arange(1,200)
colors = ['r','g','b','y','c']


nquad = 100
npoly_order = nquad-1
nmodels = 2
target_exceed = 1e-4

x = dist_x.sample(1e7)
y = foo(x)
# y = foo(x, noise=False)
mean_mcs = np.mean(y)
vart_mcs = np.std(y)**2
print('E[x]: {:.2f}'.format(np.mean(y)))
print('Var[x]:{:.2f}'.format(np.std(y)**2))

# ####>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## true system evaluation
x = np.arange(0,10,0.1)

y0 = 0.5*x**2 + 1 +  0.1*(x**2+1)
y_plus_std = y0 #np.where((x>LO)*(x<UP),y0,0)

y0 = 0.5*x**2 + 1 -  0.1*(x**2+1)
y_minus_std = y0 #np.where((x>LO)*(x<UP),y0,0)

y0 = 0.5*x**2 + 1 
y_mean = y0 #np.where((x>LO)*(x<UP),y0,0)

# x_target = dist_x.inv([target_exceed, 1-target_exceed])
# plt.figure()
# plt.plot(x, y_mean, '-k', label=r'$\mu_{y}$')
# plt.plot(x, y_plus_std, '-.k', label=r'$\mu_{y} \pm \sigma$')
# plt.plot(x, y_minus_std, '-.k')
# plt.plot([x_target[0],x_target[0]],[-10,70], '-.r')
# plt.plot([x_target[1],x_target[1]],[-10,70], '-.r',label='Limit for exceedence {:.1E}'.format(target_exceed))
# plt.xlabel(r'$x \sim N(5,1)$')
# plt.ylim(-10,70)
# plt.title(r'$y=0.5(x^2+1)+\epsilon, x\sim N(5,1)$')
# plt.legend()
# plt.savefig(r'trueSystem.eps')
# ####>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# #### Plot quadrature points distribution 
# x_target = dist_x.inv([target_exceed, 1-target_exceed])
# print(x_target)
# plt.figure()
# for iquad in np.arange(nquad,0,-1):
    # orth_poly = cp.orth_ttr(npoly_order, dist_zeta)
    # zeta_quad, zeta_weight = cp.generate_quadrature(iquad-1,dist_zeta,rule='G')
    # x_quad = dist_x.inv(dist_zeta.cdf(zeta_quad))
    # plt.plot(x_quad, np.ones(x_quad.shape)*(iquad),'k*')

# plt.plot([x_target[0],x_target[0]],[1,nquad+1], '-.r')
# plt.plot([x_target[1],x_target[1]],[1,nquad+1], '-.r',label='Limit for exceedence {:.1E}'.format(target_exceed))
# plt.xlabel(r'$x \sim N(5,1)$')
# plt.legend()
# plt.savefig(r'quad_distribution.eps')



####>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#### Surrogate model
orth_poly = cp.orth_ttr(npoly_order, dist_zeta)
zeta_quad, zeta_weight = cp.generate_quadrature(nquad-1,dist_zeta,rule='G')
x_quad = dist_x.inv(dist_zeta.cdf(zeta_quad))
assert nquad == x_quad.shape[1]
y_quad = []

# ###>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ### build quadrature surrogate models
# print('Building PCE model with {:d} quadrature points'.format(x_quad.shape[1]))
# surrogate_models = []
# for i in range(nmodels):
    # print('\tSurrogate model {:d}'.format(i))
    # evals = foo(x_quad, seed = seeds[i])
    # y_quad.append(evals)
    # foo_hat, foo_coeffs = cp.fit_quadrature(orth_poly, zeta_quad, zeta_weight, evals.T, retall=True)
    # surrogate_models.append(foo_hat)
# y_quad = np.array(y_quad)

# ###>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ## Plot surrogate model
# for i in range(nmodels):
    # _x = np.hstack((x_quad.reshape((x_quad.size,)), x))
    # _y = foo(_x,seed = seeds[i])
    # # y_quad = _y[:x_quad.size] 
    # xy = np.array([_x,_y]).T
    # xy = xy[np.argsort(xy[:,0]),:].T
    # x0 = xy[0,:]
    # y0 = xy[1,:]

    # foo_hat = surrogate_models[i]
    # _zeta = dist_zeta.inv(dist_x.cdf(x0))
    # f_hat = foo_hat(_zeta.T).reshape((_zeta.size,))
    # # f_hat1 = f_hat.reshape((f_hat.size,))

    # plt.figure()
    # plt.plot(x, y_mean, '-k', label=r'$\mu_{y}$')
    # plt.plot(x, y_plus_std, '-.k', label=r'$\mu_{y} \pm \sigma$')
    # plt.plot(x, y_minus_std, '-.k')

    # plt.plot(x_quad.T, y_quad[i,:].T,'bo', label=r'quad points')
    # plt.plot(x0, y0, label=r'realization')
    # plt.plot(x0, f_hat, color='y',label=r'label=PCE({:d},{:d})'.format(npoly_order,nquad))
    # plt.title(r'$y=0.5(x^2+1)+\epsilon, x\sim N(5,1)$')
    # plt.xlabel(r'x')
    # plt.ylim(-10,70)
    # plt.legend()
    # plt.savefig(r'./surrogates_plots/surroge_model{}_{}p_{}q.eps'.format(i,npoly_order, nquad))

###>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
### Build surrogate model with mean
sys_mean = []
sys_vart = []
sys_var1 = []
sys_var2 = []
nmodels = np.arange(4,151,2)

for imodels in nmodels:
    print('Building PCE model with {:d} quadrature points'.format(x_quad.shape[1]))
    y_quad = np.array([foo(x_quad, seed=seeds[i]) for i in range(imodels)])
# y_quad = np.array([foo(x_quad, noise=False) for i in range(nmodels)])
# surrogate_models = []
# for i in range(nmodels):
        # print('\tSurrogate model {:d}'.format(i))
        # evals = foo(x_quad, seed = seeds[i])
        # y_quad.append(evals)
        # foo_hat, foo_coeffs = cp.fit_quadrature(orth_poly, zeta_quad, zeta_weight, evals.T, retall=True)
        # surrogate_models.append(foo_hat)
# y_quad = np.array(y_quad)

###>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
### Plot surrogate model with mean
# y_quad = np.squeeze(y_quad[:,0,:])
    y_quad_mean = np.mean(np.squeeze(y_quad[:,0,:]), axis=0)
    y_quad_var = np.std(np.squeeze(y_quad[:,0,:]), axis=0)**2
    foo_hat_mean, foo_coeffs_mean = cp.fit_quadrature(orth_poly, zeta_quad, zeta_weight, np.squeeze(y_quad_mean), retall=True)
    foo_hat_var, foo_coeffs_var= cp.fit_quadrature(orth_poly, zeta_quad, zeta_weight, np.squeeze(y_quad_var), retall=True)
    print('nModels:{:d}'.format(imodels))
    print('E[x]:{:.2f}'.format(foo_coeffs_mean[0]))
    sys_mean.append(foo_coeffs_mean[0])
    var1 = np.sum(foo_coeffs_mean[1:]**2)
    var2 = foo_coeffs_var[0]
    sys_var1.append(var1)
    sys_var2.append(var2)
    sys_vart.append(var1+var2)
    print('Explained Var1[x]:{:.2f}'.format(var1))
    print('Unexplained Var2[x]:{:.2f}'.format(var2))
    print('Var1+Var2={:.2f}'.format(var1+var2))
    _zeta = dist_zeta.inv(dist_x.cdf(x))
    f_hat_mean = foo_hat_mean(_zeta.T).reshape((_zeta.size,))

    # plt.figure()
    # plt.plot(x, y_mean, '-k', label=r'$\mu_{y}$')
    # plt.plot(x, y_plus_std, '-.k', label=r'$\mu_{y} \pm \sigma$')
    # plt.plot(x, y_minus_std, '-.k')
    # plt.plot(x_quad.T, np.squeeze(y_quad).T, 'bo')
    # plt.plot(x, f_hat_mean, color='y',label=r'label=PCE({:d},{:d})'.format(npoly_order,nquad))
    # plt.ylim(-10,70)
    # plt.xlim(0,15)
# # y_quad = np.vstack((y_quad,y_quad_mean))
# # surrogate_models.append(foo_hat)
# # print(y_quad_mean.shape)
    # plt.savefig(r'Surrogate_model_with_mean_{}m{}p{}q.eps'.format(imodels, npoly_order, nquad))

fig, axes = plt.subplots(2,1, sharex=True)
axes[0].plot(nmodels, np.array(sys_mean), '-o', markersize=4, label=r'$g_0$')
axes[0].plot(np.array([nmodels[0],nmodels[-1]]), np.array([mean_mcs, mean_mcs]), 'r-.', label=r'MCS')
axes[0].legend()
axes[0].grid()
axes[0].set_title(r'E[Y]')
# axes[0].set_xlabel('number of short-term simulations')

axes[1].plot(nmodels, np.array(sys_vart), '-o', markersize=4,label=r'Total variance')
axes[1].plot(nmodels, np.array(sys_var1), '-o', markersize=4,label=r'$Var[E[Y|X]]$')
axes[1].plot(nmodels, np.array(sys_var2), '-o', markersize=4,label=r'$E[Var[Y|X]]$')
axes[1].plot(np.array([nmodels[0],nmodels[-1]]), np.array([vart_mcs, vart_mcs]), 'r-.', label=r'MCS')
axes[1].legend(prop={'size': 6})
axes[1].set_title(r'Var[Y]')
axes[1].set_xlabel('number of short-term simulations')
axes[1].set_xlim((nmodels[0], nmodels[-1]))
axes[1].grid()
plt.savefig(r'statsVSnmodels.eps')


fig, axes = plt.subplots(2,1, sharex=True)
axes[0].plot(nmodels, np.array(sys_mean)/mean_mcs, '-o', markersize=4, label=r'$g_0$')
# axes[0].plot(np.array([nmodels[0],nmodels[-1]]), np.array([1, 1]), 'r-.', label=r'MCS')
axes[0].legend()
axes[0].set_title(r'Normalized E[Y]')
axes[0].grid()
# axes[0].set_xlabel('number of short-term simulations')

axes[1].plot(nmodels, np.array(sys_vart)/vart_mcs, '-o', markersize=4,label=r'Total variance')
axes[1].plot(nmodels, np.array(sys_var1)/vart_mcs, '-o', markersize=4,label=r'$Var[E[Y|X]]$')
axes[1].plot(nmodels, np.array(sys_var2)/vart_mcs, '-o', markersize=4,label=r'$E[Var[Y|X]]$')
# axes[1].plot(np.array([nmodels[0],nmodels[-1]]), np.array([vart_mcs, vart_mcs]), 'r-.', label=r'MCS')
axes[1].legend(prop={'size': 6})
axes[1].set_title(r'Normalized Var[Y]')
axes[1].set_xlabel('number of short-term simulations')
axes[1].set_xlim((nmodels[0], nmodels[-1]))
axes[1].grid()
plt.savefig(r'statsVSnmodels_norm.eps')


# ###>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# # ### Prediction with one surrogate model
# # ### benchmark from true sys
# nsamples = int(1e5)
# samples_x = dist_x.sample(nsamples)
# samples_zeta = dist_zeta.inv(dist_x.cdf(samples_x))
# samples_y_true = foo(samples_x)
# data = np.array([samples_x, samples_y_true]).T
# df = pd.DataFrame(data, columns=["x", "y"])

# # sns.jointplot(x="x", y="y", data=df, kind='reg');
# # fh = sns.lineplot(x,y_mean,color='k',linestyle='-')
# # fh.plot(x,y_minus_std,color='k',linestyle='-.')
# # fh.plot(x,y_plus_std,color='k',linestyle='-.')
# # plt.savefig('TrueJointPdf.eps')

# y_pred = []
# for i in range(nmodels):
    # print('Predicting with surrogate model {}'.format(i))
    # foo_hat = surrogate_models[i]
    # f_hat = foo_hat(samples_zeta.T).reshape((samples_zeta.size,))
    # y_pred.append(f_hat)

    # plt.figure()
    # sns.distplot(f_hat,label=r'PCE({:d},{:d})'.format(npoly_order,nquad))
    # sns.distplot(samples_y_true,label='True')
    # plt.ylim(0,0.5)
    # plt.xlim(-10,50)
    # plt.xlabel('y')
    # plt.ylabel('$f_Y(y)$')
    # plt.title(r'PDF comparison with {:.1E} samples'.format(nsamples))
    # plt.legend()
    # plt.savefig('./pdf_surrogates/pdf_surrogate{}_{}p{}q'.format(i, npoly_order, nquad))

####>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ### Prediction with random surrogate model
# ### benchmark from true sys

# print('Exceedence prediction with PCE model built with {:d} quadrature points'.format(nquad))
# nsamples_mcs = int(1/target_exceed*100)
# samples_x = dist_x.sample(nsamples_mcs)
# samples_zeta = dist_zeta.inv(dist_x.cdf(samples_x))
# samples_y_true = foo(samples_x)
# data = np.array([samples_x, samples_y_true]).T
# df = pd.DataFrame(data, columns=["x", "y"])
# ecdf_true = ECDF(samples_y_true)
# # sns.jointplot(x="x", y="y", data=df, kind='reg');
# # fh = sns.lineplot(x,y_mean,color='k',linestyle='-')
# # fh.plot(x,y_minus_std,color='k',linestyle='-.')
# # fh.plot(x,y_plus_std,color='k',linestyle='-.')
# # plt.savefig('TrueJointPdf.eps')

# nrepeat = 10
# nsamples = int(1/target_exceed)
# plt.figure()
# plt.semilogy(ecdf_true.x,1-ecdf_true.y,'-r',label='MCS({:.1E})'.format(nsamples_mcs))
# for i in range(nrepeat):
    # print('\tRepeating {}'.format(i))
    # np.random.seed()
    # samples_x = dist_x.sample(nsamples)
    # samples_zeta = dist_zeta.inv(dist_x.cdf(samples_x))
    # randnum = np.random.randint(0,nmodels, nsamples)
    # random_model = [surrogate_models[i] for i in randnum]
    # f_hat = np.array([ foo_hat(samples_zeta[i]) for i, foo_hat in enumerate(random_model)])

    # ecdf_f_hat = ECDF(f_hat.flatten())
    # if i==nrepeat-1:
        # plt.semilogy(ecdf_f_hat.x,1-ecdf_f_hat.y,'-.k', label='PCE({:d},{:d})'.format(npoly_order,nquad))
    # else:
        # plt.semilogy(ecdf_f_hat.x,1-ecdf_f_hat.y,'-.k')
# plt.title('ECDF: MCS vs random selected PCEs ({:d})'.format(nmodels))
# plt.ylim(1/nsamples,1)
# plt.xlim(-10,60)
# plt.legend()
# plt.savefig('./ecdf_random_surrogate/ecdf_random_surrogate{}_{}p{}q{}m'.format(i, npoly_order, nquad, nmodels))


## Prediction mean value 

####>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# plt.hist(f_pred.T, normed=True, cumulative=True, label='CDF', histtype='step', alpha=0.8)
# plt.hist(y, normed=True, cumulative=True, label='CDF', histtype='step', alpha=0.8)


# plt.plot(samples_x, y, 'o')
# plt.show()

# ######--------------------------------------------------------------------------
# ####>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# #### Random field method, ref Roger Ghanem
# np.set_printoptions(precision=3,suppress=True)
# # #1. Get samples (use quadrature points instead)

# orth_poly = cp.orth_ttr(npoly_order, dist_zeta)
# zeta_quad, zeta_weight = cp.generate_quadrature(nquad-1,dist_zeta,rule='G')
# x_quad = dist_x.inv(dist_zeta.cdf(zeta_quad))
# y_quad = np.array([foo(x_quad,seed=seeds[i],noise=True) for i in range(nmodels)])
# y_quad = np.squeeze(y_quad[:,0,:])
# y_quad_mean = np.mean(y_quad, axis=0)
# # foo_hat, foo_coeffs = cp.fit_quadrature(orth_poly, zeta_quad, zeta_weight, y_quad_mean, retall=True)
# x2zeta = dist_zeta.inv(dist_x.cdf(x))
# # f_hat = foo_hat(x2zeta.T)
# # print(foo_hat.coeffs())
# # print(foo_hat)
# # print(y_quad)
# # print(y_quad_mean)
# y_quad = y_quad - y_quad_mean
# # print(y_quad)
# # plt.figure()
# # plt.plot(x_quad.T, y_quad.T, 'bo', label=r'quad')
# # plt.plot(x,f_hat, 'y',label=r'surrogate')
# # plt.title(r'$y=0.5(x^2+1)+\epsilon, x\sim N(5,1)$')
# # plt.text(6, 10, r'$0.5\xi^2 + 5\xi+ 13$' )
# # plt.xlabel(r'x')
# # plt.ylim(-10,70)
# # plt.savefig(r'deterministic_pce_{}p_{}_q.eps'.format(npoly_order, nquad))
# # plt.show()

# # #2. Get the correlation function
# y_cov = np.cov(y_quad.T)
# # print(y_cov)
# # y_cov1 = np.dot(y_quad.T, y_quad)/(nmodels-1)



