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
import scipy.stats as spstats
import math
import context
from doe.doe_generator import samplegen
from scipy.stats.kde import gaussian_kde
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt

np.set_printoptions(precision=5,suppress=True)

pltcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# pltlinestyles = ['solid','dashed','dashdotted','dotted','loosely dashed','loosely dashdotted']
pltlinestyles = [ (0, ()),(0, (5, 5)),(0, (3, 5, 1, 5)),(0, (1, 5)),(0, (3, 5, 1, 5, 1, 5)), 
        (0, (5, 10)), (0, (3, 10, 1, 10)), (0, (3, 10, 1, 10, 1, 10)),(0, (1, 10)), 
        (0, (5, 1)),  (0, (3, 1, 1, 1)),   (0, (3, 1, 1, 1, 1, 1)),(0, (1, 1))  ]

pltmarkers = ['o','v','s','+','*']
ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(math.floor(n/10)%10!=1)*(n%10<4)*n%10::4])
def _pce_approx(dist_x, dist_zeta, npoly_order, nquad=15, rule=None):
    orth_poly, norms = cp.orth_ttr(npoly_order, dist_zeta, retall=True)
    zeta_quad, zeta_weight = samplegen('GQ', nquad, dist_zeta, rule=rule)
    x_quad = dist_x.inv(dist_zeta.cdf(zeta_quad)).T
    x_hat,coeffs = cp.fit_quadrature(orth_poly, zeta_quad, zeta_weight, x_quad, retall=True)
    # x_hat,coeffs = cp.fit_quadrature(orth_poly, zeta_quad, zeta_weight, x_quad, norms=norms,retall=True)

    return x_hat, coeffs
def _metric5(p,dist,y):
    """
    Reference: On the accuracy of the polynomial chaos approximation R.V. Field Jr., M. Grigoriu
     measure for the difference between the tails of Y and Y_hat at the p% upper fractile.
    """
    p_inv = dist.inv(1-p)
    count = sum(i > p_inv for i in y)
    return count/len(y)

nsamples = 1e6
p_fractile = 0.01
npoly_order = np.arange(1,5)
q = np.linspace(0,1,100)[1:-1]
exp_lambda = 3
dist_x = cp.Exponential(exp_lambda)  ## Gamma(1,1)
# gamma_shape, gamma_scale = 2,1
# dist_x2 = cp.Gamma(gamma_shape,gamma_scale)
dist_zeta1   = cp.Normal()
dist_zeta2   = cp.Gamma()
samples_zeta1 = dist_zeta1.sample(nsamples)
samples_zeta2 = dist_zeta2.sample(nsamples)
x1_hat = []
x2_hat = []
x1_coeffs = []
x2_coeffs = []
x1_stats_hat = []
x2_stats_hat = []
x1_ecdf_hat = []
x2_ecdf_hat = []
x1_pdf_kde  = []
x2_pdf_kde  = []

x_inv = dist_x.inv(q)
# x2_inv = dist_x2.inv(q)
zeta1_inv = dist_zeta1.inv(q)
zeta2_inv = dist_zeta2.inv(q)
for ipoly_order in npoly_order:
    print('npoly_order: {:d}'.format(ipoly_order))
    _x1_hat, _x1_coeffs = _pce_approx(dist_x, dist_zeta1, ipoly_order, rule='hermite')
    _x2_hat, _x2_coeffs = _pce_approx(dist_x, dist_zeta2, ipoly_order, rule='lag')
    _x1_samples = np.squeeze(_x1_hat(samples_zeta1))
    _x2_samples = np.squeeze(_x2_hat(samples_zeta2))
    x1_stats_hat.append([np.mean(_x1_samples), np.std(_x1_samples),\
            spstats.skew(_x1_samples), spstats.kurtosis(_x1_samples),\
            _metric5(p_fractile, dist_x, _x1_samples)])
    x2_stats_hat.append([np.mean(_x2_samples), np.std(_x2_samples),\
            spstats.skew(_x2_samples), spstats.kurtosis(_x2_samples),\
            _metric5(p_fractile, dist_x, _x2_samples)])
    x1_ecdf_hat.append(ECDF(_x1_samples))
    x2_ecdf_hat.append(ECDF(_x2_samples))
    x1_pdf_kde.append(gaussian_kde(_x1_samples))
    x2_pdf_kde.append(gaussian_kde(_x2_samples))

    x1_hat.append(_x1_hat)
    x1_coeffs.append(_x1_coeffs)
    x2_hat.append(_x2_hat)
    x2_coeffs.append(_x2_coeffs)

print(np.array(x1_stats_hat).shape)
#### =========================================================================
figs, axes = plt.subplots(2,2,figsize=(12, 9))
##>>> figs[0,0]: coefficients
axes[0,0].plot(np.arange(npoly_order[-1]+1), x1_coeffs[-1],'-s',color=pltcolors[0], label=r'Hermite-chaos'.format(exp_lambda))
axes[0,0].plot(np.arange(npoly_order[-1]+1), x2_coeffs[-1],'-s',color=pltcolors[1],label=r'Laguerre-chaos'.format(exp_lambda))
axes[0,0].set_xlabel(r'$P$')
axes[0,0].set_ylabel(r'$u_i$')
axes[0,0].set_title('Coefficients')
# axes[0,0].plot(x21_coeffs,'-s',label=r'Gamma(1,1):Laguerre-chaos')
axes[0,0].legend()
##>>> figs[0,1]: moments 
mom_names = ['mean', 'std', 'skewness', 'kurtosis']
moms_exact = [exp_lambda,exp_lambda,2,6]
x1_stats_hat = np.array(x1_stats_hat)
x2_stats_hat = np.array(x2_stats_hat)
for i in np.arange(4):
    print(abs(x1_stats_hat[:,i]), moms_exact[i])
    print(abs(x1_stats_hat[:,i]-moms_exact[i])/moms_exact[i])
    axes[0,1].semilogy(npoly_order, abs(x1_stats_hat[:,i]-moms_exact[i])/moms_exact[i],\
            linestyle=pltlinestyles[i], marker=pltmarkers[i],color=pltcolors[0],label=r'{}'.format(mom_names[i]))
    axes[0,1].semilogy(npoly_order, abs(x2_stats_hat[:,i]-moms_exact[i])/moms_exact[i],\
            linestyle=pltlinestyles[i], marker=pltmarkers[i],color=pltcolors[1])

axes[0,1].set_xlabel(r'$P$')
axes[0,1].set_ylabel(r'$\epsilon$')
axes[0,1].set_title(r'Relative error $\epsilon = |\frac{\hat{\mu_i} - \mu_i}{\mu_i}|$')
axes[0,1].legend(loc=1)

##>>> figs[1,0]: exceedence plot 

q = np.linspace(0,1,nsamples)[1:-1]
x_cdfx = dist_x.inv(q)
x_cdfy = q
axes[1,0].semilogy(x_cdfx,1-x_cdfy, '-.k',label=r'exact')

for i in np.arange(0,len(x1_ecdf_hat)):
    axes[1,0].semilogy(x1_ecdf_hat[i].x,1-x1_ecdf_hat[i].y,linestyle=pltlinestyles[i],color=pltcolors[0], label=r'${}$-order'.format(ordinal(i+1)))
    axes[1,0].semilogy(x2_ecdf_hat[i].x,1-x2_ecdf_hat[i].y,linestyle=pltlinestyles[i],color=pltcolors[1])#, label=r'${}$-order'.format(i+1)

axes[1,0].set_xlabel(r'$u$')
axes[1,0].set_ylabel(r'$Probability$')
axes[1,0].set_title(r'Exceedence probability $P(X>x_0)$')
axes[1,0].set_ylim((10/nsamples,1))
axes[1,0].legend()

##>>> figs[1,1]: upper 1% fractile 
axes[1,1].plot(npoly_order, x1_stats_hat[:,4],linestyle=pltlinestyles[0],color=pltcolors[0], marker=pltmarkers[2])
axes[1,1].plot(npoly_order, x2_stats_hat[:,4],linestyle=pltlinestyles[0],color=pltcolors[1], marker=pltmarkers[2])

axes[1,1].set_xlabel(r'$P$')
axes[1,1].set_ylabel(r'$Probability$')
axes[1,1].set_title(r'Upper 1% fractile P($\hat{X}$>$F_X^{-1}(0.99))$')
# axes[1,1].legend()
# ##>>> figs[2,0]: pdf BAD
# x = dist_x.inv(q)
# x_pdf = dist_x.pdf(x)

# axes[2,0].plot(x, x_pdf, '-k' ,label=r'exact')

# for i in np.arange(0,len(x1_pdf_kde),2):
    # print('{}'.format(i))
    # x1_kde = x1_pdf_kde[i]
    # x2_kde = x2_pdf_kde[i]
    # axes[2,0].plot(x, x1_kde(x),linestyle=pltlinestyles[2], marker=pltmarkers[i],color=pltcolors[i])
    # axes[2,0].plot(x, x2_kde(x),linestyle=pltlinestyles[3], marker=pltmarkers[i],color=pltcolors[i])




plt.savefig('Exp_pce.eps')
plt.savefig('Exp_pce.png')
# axes[1,0].plot(x22_coeffs,'-s',label=r'Gamma(2,1):Laguerre-chaos')
# axes[1,0].legend()
# plt.show()
# gumbel_loc, gumbel_scale= 3,4
# dist_zeta = cp.Normal()
# dist_unif = cp.Uniform(0,1)
# dist_norm = cp.Normal(gumbel_loc, gumbel_scale)
# dist_gumbel = cp.Logweibul(gumbel_scale, gumbel_loc)  # gumbel = logweibull
# # dist_gumbel = cp.Exponential()
# # dist_gumbel = cp.Normal(2,4)
# # dist_weibull = cp.Weibull(scale=gumbel_scale, shift=gumbel_loc) 
# ####>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# # #### Data explore
# # x = np.arange(-15,15,0.1)
# # zeta_pdf = dist_zeta.pdf(x)
# # norm_pdf = dist_norm.pdf(x)
# # gumbel_pdf = dist_gumbel.pdf(x)


# # plt.figure()
# # plt.plot(x, zeta_pdf, label='$N(0,1)$')
# # plt.plot(x, norm_pdf, label='$N({},{})$'.format(gumbel_loc, gumbel_scale))
# # plt.plot(x, gumbel_pdf, label='$Gum(scale={},loc={}))$'.format(gumbel_scale, gumbel_loc))
# # plt.legend()
# # plt.show()

# ####>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# #### fit gumbel variable
# npoly_order = 8
# nquad = 15
# orth_poly, norms = cp.orth_ttr(npoly_order, dist_zeta, retall=True)
# zeta_quad, zeta_weight = cp.generate_quadrature(nquad,dist_zeta,rule='G')
# unif_quad, unif_weight = cp.generate_quadrature(nquad,dist_unif,rule='G')
# # print(unif_quad)

# gumbel_quad = dist_gumbel.inv(unif_quad).T
# gumbel_hat,coeffs = cp.fit_quadrature(orth_poly, dist_zeta.inv(unif_quad), unif_weight, gumbel_quad, norms=norms, retall=True)
# print(coeffs)

# gumbel_quad = dist_gumbel.inv(dist_zeta.cdf(zeta_quad)).T
# gumbel_hat,coeffs = cp.fit_quadrature(orth_poly, zeta_quad, zeta_weight, gumbel_quad, norms=norms, retall=True)
# print(coeffs)
# print(coeffs.shape)
# plt.figure()
# plt.plot(np.arange(coeffs.size), np.squeeze(coeffs[:]), '-s')
# plt.show()

# gumbel_x = np.arange(-5,5,0.1)
# gumbel_pdf = dist_gumbel.pdf(gumbel_x)

# gumbel_x2zeta = dist_zeta.inv(dist_gumbel.cdf(gumbel_x))
# gumbel_pdf_hat = gumbel_hat(gumbel_x2zeta)

# plt.figure()
# plt.plot(gumbel_x, gumbel_pdf)
# plt.plot(gumbel_x, np.squeeze(gumbel_pdf_hat))
# plt.xlim(-4,14)
# plt.ylim(0,1)
# plt.show()




