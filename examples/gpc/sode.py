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
from scipy.integrate import odeint
from doe.doe_generator import samplegen
from utilities.get_stats import _central_moms
from scipy.stats.kde import gaussian_kde
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt

np.set_printoptions(precision=5,suppress=True)

pltcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# pltlinestyles = ['solid','dashed','dashdotted','dotted','loosely dashed','loosely dashdotted']
"""
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
"""

pltlinestyles = [ (0, (1, 5)),(0, (3, 5, 1, 5)),(0, (5, 5)),(0, ()), (0, (3, 1, 1, 1, 1, 1)), 
        (0, (5, 10)), (0, (3, 10, 1, 10)), (0, (3, 10, 1, 10, 1, 10)),(0, (1, 10)), 
        (0, (5, 1)),  (0, (3, 1, 1, 1)),(0, (3, 5, 1, 5, 1, 5)),  (0, (1, 1))  ]*10

pltmarkers = ['o','v','s','d','+','*']
ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(math.floor(n/10)%10!=1)*(n%10<4)*n%10::4])

def _pce_approx(dist_x, dist_zeta, npoly_order, nquad=150, rule=None):
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
def _estimate_pdf():
    pass

def _error(x_hat, x, e=1e-3):
    """
    return the error from x_hat and x. If x < e(tolerance), absolute error is return, otherwise, relative error is returned
    """
    # print(x_hat, x)
    error_abs = abs(x_hat - x)
    error = abs(error_abs / x) if min(abs(x)) > e else error_abs
    return error

def _cal_eijk(orth_poly, dist):
    """
    e[i,j,k] = <Phi, Phj, Phk>
    """
    P = len(orth_poly)
    e = np.zeros([P,P,P])
    for i, ipoly in enumerate(orth_poly):
        for j, jpoly in enumerate(orth_poly):
            for k, kpoly in enumerate(orth_poly):
                prod_poly = ipoly * jpoly * kpoly
                # prod_dist = cp.QoI_Dist(prod_poly, dist, sample=sample)
                e[i,j,k] = cp.E(prod_poly, dist)
    return e

def sode_pce(y,t,k,e,norms):
    """
    len(k) = P
    e.shape = (P,P,P)
    """
    assert len(k) == e.shape[0]==e.shape[1]==e.shape[2]
    dydt = []
    P = len(k)
    k = np.squeeze(k)
    for l in np.arange(P):
        _dydtl = 0
        for i in np.arange(P):
            for j in np.arange(P):
                _dydtl = _dydtl + e[i,j,l] * k[i] * y[j]
        dydt.append(-1/norms[l]*_dydtl)
    return dydt

def sode(y,t,k):
    dydt = -k*y
    return dydt



nsamples = 1e7
p1,p2 = 0.01, 0.001
npoly_order = 4#np.arange(1,10)
q = np.linspace(0,1,101)[1:-1]
t = np.linspace(0,1,101)
# exp_lambda = 3
# dist_x = cp.Exponential(exp_lambda)  ## Gamma(1,1)
# gamma_shape, gamma_scale = 2,1
# dist_x = cp.Gamma(gamma_shape,gamma_scale)
# beta_a, beta_b = 0.5,0.5 
# dist_x = cp.Beta(beta_a, beta_b)
dist_k = cp.Normal()
# mom_names = ['mean', 'std', 'skewness', 'kurtosis']
# moms_exact = _central_moms(dist_x) 

y0 = 1 
k_samples = dist_k.sample(nsamples)
y_sol = []
for ik in k_samples:
    sol = odeint(sode, y0, t, args=(ik,))
    y_sol.append(sol)
np.save('sode_true.npy', np.array(y_sol))

y_mc = np.load('sode_true.npy')
y_mc_mean = np.squeeze(np.mean(y_mc,axis=0))
y_mc_var  = np.squeeze(np.std(y_mc, axis=0)**2)
print(y_mc_mean.shape)
print(y_mc_var.shape)


k1_hat = []
k2_hat = []
k1_coeffs = []
k2_coeffs = []
k1_stats_hat = []
k2_stats_hat = []
k1_ecdf_hat = []
k2_ecdf_hat = []
k1_pdf_kde  = []
k2_pdf_kde  = []
y1_error = []
y2_error = []
y1_sol = []
y2_sol = []

dist_zeta1   = cp.Normal()
dist_zeta2   = cp.Uniform(0,1)
for npoly_order in np.arange(1,5):
    _, _k1_coeffs = _pce_approx(dist_k, dist_zeta1, npoly_order)
    _, _k2_coeffs = _pce_approx(dist_k, dist_zeta2, npoly_order)
    orth_poly1, norms1   = cp.orth_ttr(npoly_order, dist_zeta1, retall=True)
    orth_poly2, norms2   = cp.orth_ttr(npoly_order, dist_zeta2, retall=True)
    e1 = _cal_eijk(orth_poly1, dist_zeta1)
    e2 = _cal_eijk(orth_poly2, dist_zeta2)

    k1_coeffs.append(_k1_coeffs)
    k2_coeffs.append(_k2_coeffs)

    np.random.seed(100)
    y0 = [0,]*(npoly_order+1)
    y0[0] = 1
    sol = odeint(sode_pce, y0, t, args=(_k1_coeffs,e1,norms1))
    y1_sol.append(sol)
    _y_mean = sol[:,0]
    _y_var = np.sum(sol**2,axis=1) - _y_mean**2
    print(_y_mean.shape, y_mc_mean.shape)
    y1_error.append([_error(_y_mean,y_mc_mean), _error(_y_var,y_mc_var)])

y1_error = np.array(y1_error)
y2_error = np.array(y2_error)

plt.figure()
plt.plot(t, y_mc_mean, '-r', label=r'mc')
for i in np.arange(npoly_order+1):
    plt.plot(t,sol[:,i],linestyle=pltlinestyles[i],color='k', label=r'$y{}$'.format(i))
plt.legend()

plt.figure()
plt.semilogy(np.arange(npoly_order)+1, np.squeeze(y1_error[:,0,-1]), marker='d',label=r'Mean')
plt.semilogy(np.arange(npoly_order)+1, np.squeeze(y1_error[:,1,-1]), marker='s',label=r'Variance')
plt.legend()
plt.show()



# samples_zeta1 = dist_zeta1.sample(nsamples)
# samples_zeta2 = dist_zeta2.sample(nsamples)

# x_inv = dist_x.inv(q)
# # x2_inv = dist_x2.inv(q)
# zeta1_inv = dist_zeta1.inv(q)
# zeta2_inv = dist_zeta2.inv(q)
# for ipoly_order in npoly_order:
    # print('npoly_order: {:d}'.format(ipoly_order))
    # _x1_hat, _x1_coeffs = _pce_approx(dist_x, dist_zeta1, ipoly_order, rule='hermite')
    # # _x2_hat, _x2_coeffs = _pce_approx(dist_x, dist_zeta2, ipoly_order, rule='e')
    # _x2_hat, _x2_coeffs = _pce_approx(dist_x, dist_zeta2, ipoly_order, rule='e')
    # _x1_samples = np.squeeze(_x1_hat(samples_zeta1))
    # _x2_samples = np.squeeze(_x2_hat(samples_zeta2))
    # x1_stats_hat.append([np.mean(_x1_samples), np.std(_x1_samples),\
            # spstats.skew(_x1_samples), spstats.kurtosis(_x1_samples),\
            # _metric5(p1, dist_x, _x1_samples),_metric5(p2, dist_x, _x1_samples)])
    # x2_stats_hat.append([np.mean(_x2_samples), np.std(_x2_samples),\
            # spstats.skew(_x2_samples), spstats.kurtosis(_x2_samples),\
            # _metric5(p1, dist_x, _x2_samples),_metric5(p2, dist_x, _x2_samples)])
    # x1_ecdf_hat.append(ECDF(_x1_samples))
    # x2_ecdf_hat.append(ECDF(_x2_samples))
    # # x1_pdf_kde.append(gaussian_kde(_x1_samples))
    # # x2_pdf_kde.append(gaussian_kde(_x2_samples))

    # x1_hat.append(_x1_hat)
    # x1_coeffs.append(_x1_coeffs)
    # x2_hat.append(_x2_hat)
    # x2_coeffs.append(_x2_coeffs)
# print(x1_coeffs[-1])
# print(np.array(x1_stats_hat).shape)
# #### =========================================================================
# figs, axes = plt.subplots(2,2,figsize=(12, 9))
# ##>>> figs[0,0]: coefficients
# axes[0,0].plot(np.arange(npoly_order[-1]+1), x1_coeffs[-1],'-s',color=pltcolors[0], label=r'Hermite-chaos')
# axes[0,0].plot(np.arange(npoly_order[-1]+1), x2_coeffs[-1],'-s',color=pltcolors[1], label=r'Legendre-chaos')
# axes[0,0].set_xlabel(r'Index $i$')
# axes[0,0].set_ylabel(r'$u_i$')
# axes[0,0].set_title('Coefficients')
# # axes[0,0].plot(x21_coeffs,'-s',label=r'Gamma(1,1):Laguerre-chaos')
# axes[0,0].legend()
# ##>>> figs[0,1]: moments 
# x1_stats_hat = np.array(x1_stats_hat)
# x2_stats_hat = np.array(x2_stats_hat)
# for i in np.arange(4):
    # print(abs(x1_stats_hat[:,i]), moms_exact[i])
    # print(abs(x1_stats_hat[:,i]-moms_exact[i])/moms_exact[i])
    # axes[0,1].semilogy(npoly_order, _error(x1_stats_hat[:,i], moms_exact[i]),\
            # linestyle=pltlinestyles[4-i], marker=pltmarkers[i],color=pltcolors[0],label=r'{}'.format(mom_names[i]))
    # axes[0,1].semilogy(npoly_order, _error(x2_stats_hat[:,i], moms_exact[i]),\
            # linestyle=pltlinestyles[4-i], marker=pltmarkers[i],color=pltcolors[1])

# axes[0,1].set_xlabel(r'$P$')
# axes[0,1].set_ylabel(r'$\epsilon$')
# axes[0,1].set_title(r'Relative error $\epsilon = |\frac{\hat{\mu_i} - \mu_i}{\mu_i}|$')
# axes[0,1].legend()

# ##>>> figs[1,0]: exceedence plot 

# q = np.linspace(0,1,nsamples)[1:-1]
# x_cdfx = dist_x.inv(q)
# x_cdfy = q
# axes[1,0].semilogy(x_cdfx,1-x_cdfy, '-.k',label=r'exact')

# for j,i in enumerate(np.arange(0,len(x1_ecdf_hat),2)):
    # axes[1,0].semilogy(x1_ecdf_hat[i].x,1-x1_ecdf_hat[i].y,linestyle=pltlinestyles[j],color=pltcolors[0], label=r'${}$-order'.format(ordinal(i+1)))
    # axes[1,0].semilogy(x2_ecdf_hat[i].x,1-x2_ecdf_hat[i].y,linestyle=pltlinestyles[j],color=pltcolors[1])#, label=r'${}$-order'.format(i+1)

# axes[1,0].set_xlabel(r'$u$')
# axes[1,0].set_ylabel(r'$Probability$')
# axes[1,0].set_title(r'Exceedence probability $P(X>x_0)$')
# axes[1,0].set_ylim((10/nsamples,1))
# axes[1,0].set_xlim(0,5)
# axes[1,0].legend()

# ##>>> figs[1,1]: upper alpha % fractile 
# axes[1,1].plot(npoly_order, np.ones(npoly_order.shape)*p1, '-.k')
# axes[1,1].plot(npoly_order, x1_stats_hat[:,4],linestyle=pltlinestyles[2],color=pltcolors[0], marker=pltmarkers[2], label=r'$\alpha={}$(Hermite-chaos)'.format(1-p1))
# axes[1,1].plot(npoly_order, x2_stats_hat[:,4],linestyle=pltlinestyles[2],color=pltcolors[1], marker=pltmarkers[2], label=r'$\alpha={}$(Legendre-chaos)'.format(1-p1))
# axes[1,1].set_xlabel(r'$P$')
# axes[1,1].set_ylabel(r'$Probability$')

# # axes2 = axes[1,1].twinx()
# axes[1,1].plot(npoly_order, np.ones(npoly_order.shape)*p2, '-.r')
# axes[1,1].plot(npoly_order, x1_stats_hat[:,5],linestyle=pltlinestyles[3],color=pltcolors[0], marker=pltmarkers[3], label=r'$\alpha={}$(Hermite-chaos)'.format(1-p2))
# axes[1,1].plot(npoly_order, x2_stats_hat[:,5],linestyle=pltlinestyles[3],color=pltcolors[1], marker=pltmarkers[3], label=r'$\alpha={}$(Legendre-chaos)'.format(1-p2))
# # axes[1,1]2.set_ylabel(r'$Probability (\alpha={})$'.format(1-p2))
# # axes[1,1]2.tick_params('y',colors='r')
# # axes2.legend()

# axes[1,1].text(npoly_order[0], p1, r'$\alpha={}$'.format(1-p1))
# axes[1,1].text(npoly_order[0], p2, r'$\alpha={}$'.format(1-p2),color='r')
# axes[1,1].set_title(r'Upper $(1-\alpha)$-fractile P($\hat{X}$>$F_X^{-1}(\alpha))$')
# axes[1,1].legend()
# # ##>>> figs[2,0]: pdf BAD
# # x = dist_x.inv(q)
# # x_pdf = dist_x.pdf(x)

# # axes[2,0].plot(x, x_pdf, '-k' ,label=r'exact')

# # for i in np.arange(0,len(x1_pdf_kde),2):
    # # print('{}'.format(i))
    # # x1_kde = x1_pdf_kde[i]
    # # x2_kde = x2_pdf_kde[i]
    # # axes[2,0].plot(x, x1_kde(x),linestyle=pltlinestyles[2], marker=pltmarkers[i],color=pltcolors[i])
    # # axes[2,0].plot(x, x2_kde(x),linestyle=pltlinestyles[3], marker=pltmarkers[i],color=pltcolors[i])

# plt.savefig('Figures/beta_pce.eps')
# plt.savefig('Figures/beta_pce.png')





























# # axes[1,0].plot(x22_coeffs,'-s',label=r'Gamma(2,1):Laguerre-chaos')
# # axes[1,0].legend()
# # plt.show()
# # gumbel_loc, gumbel_scale= 3,4
# # dist_zeta = cp.Normal()
# # dist_unif = cp.Uniform(0,1)
# # dist_norm = cp.Normal(gumbel_loc, gumbel_scale)
# # dist_gumbel = cp.Logweibul(gumbel_scale, gumbel_loc)  # gumbel = logweibull
# # # dist_gumbel = cp.Exponential()
# # # dist_gumbel = cp.Normal(2,4)
# # # dist_weibull = cp.Weibull(scale=gumbel_scale, shift=gumbel_loc) 
# # ####>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# # # #### Data explore
# # # x = np.arange(-15,15,0.1)
# # # zeta_pdf = dist_zeta.pdf(x)
# # # norm_pdf = dist_norm.pdf(x)
# # # gumbel_pdf = dist_gumbel.pdf(x)


# # # plt.figure()
# # # plt.plot(x, zeta_pdf, label='$N(0,1)$')
# # # plt.plot(x, norm_pdf, label='$Ndef _estimate_pdf()({},{})$'.format(gumbel_loc, gumbel_scale))D
# # # plt.plot(x, gumbel_pdf, label='$Gum(scale={},loc={}))$'.format(gumbel_scale, gumbel_loc))
# # # plt.legend()
# # # plt.show()

# # ####>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# # #### fit gumbel variable
# # npoly_order = 8
# # nquad = 15
# # orth_poly, norms = cp.orth_ttr(npoly_order, dist_zeta, retall=True)
# # zeta_quad, zeta_weight = cp.generate_quadrature(nquad,dist_zeta,rule='G')
# # unif_quad, unif_weight = cp.generate_quadrature(nquad,dist_unif,rule='G')
# # # print(unif_quad)

# # gumbel_quad = dist_gumbel.inv(unif_quad).T
# # gumbel_hat,coeffs = cp.fit_quadrature(orth_poly, dist_zeta.inv(unif_quad), unif_weight, gumbel_quad, norms=norms, retall=True)
# # print(coeffs)

# # gumbel_quad = dist_gumbel.inv(dist_zeta.cdf(zeta_quad)).T
# # gumbel_hat,coeffs = cp.fit_quadrature(orth_poly, zeta_quad, zeta_weight, gumbel_quad, norms=norms, retall=True)
# # print(coeffs)
# # print(coeffs.shape)
# # plt.figure()
# # plt.plot(np.arange(coeffs.size), np.squeeze(coeffs[:]), '-s')
# # plt.show()

# # gumbel_x = np.arange(-5,5,0.1)
# # gumbel_pdf = dist_gumbel.pdf(gumbel_x)

# gumbel_x2zeta = dist_zeta.inv(dist_gumbel.cdf(gumbel_x))
# gumbel_pdf_hat = gumbel_hat(gumbel_x2zeta)

# plt.figure()
# plt.plot(gumbel_x, gumbel_pdf)
# plt.plot(gumbel_x, np.squeeze(gumbel_pdf_hat))
# plt.xlim(-4,14)
# plt.ylim(0,1)
# plt.show()




