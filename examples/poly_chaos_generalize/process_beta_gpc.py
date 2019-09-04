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
from scipy.misc import factorial
import chaospy as cp
import scipy.stats as spstats
import math
import context
from doe.doe_generator import samplegen
from utilities.get_stats import _central_moms
from scipy.stats.kde import gaussian_kde
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt

# np.set_printoptions(precision=5,suppress=True)

pltcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']*10
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

pltmarkers = ['o','v','s','d','+','*']*10
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
def _estimate_pdf(kde, x):
    return kde(x)

def _error(x_hat, x, e=1e-3):
    """
    return the error from x_hat and x. If x < e(tolerance), absolute error is return, otherwise, relative error is returned
    """
    # print(x_hat, x)
    error_abs = abs(x_hat - x)
    error = abs(error_abs / x) if x > e else error_abs
    return error

def _correlation_x(x1,x2,lc):
    xx1, xx2 = np.meshgrid(x1,x2)
    dx = xx1 - xx2
    res = np.exp(-(dx/lc)**2)
    return res

def _gamma_process(evals, evectors, n, xi):
    nevals = len(evals)
    nevectors, ndim = evectors.shape
    assert nevals == nevectors
    assert n < nevals
    assert len(xi) >= n
    
    gamma_x = np.zeros(ndim)
    var_explained = np.zeros(ndim) 
    for i in np.arange(n):
        gamma_x = gamma_x + np.sqrt(evals[i]) * evectors[i,:] * xi[i]
        var_explained = var_explained + (np.sqrt(evals[i]) * evectors[i,:])**2

    return gamma_x, var_explained

def _sorted_eig(dist_x, ngrid_x, n=None):
    

    x_grids = np.linspace(dist_x.range()[0], dist_x.range()[1], ngrid_x)
    corr_grids_x = _correlation_x(x_grids,x_grids,lc)
    eigenvalues,eigenvectors= LA.eig(corr_grids_x)
    idx = eigenvalues.argsort()[::-1]   
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    nevals = len(eigenvalues)
    nevectors, ndim = eigenvectors.shape
    assert nevals == nevectors
    if n:
        assert n <= nevals
        var_explained_x = np.zeros(eigenvectors[0,:].shape) 
        for i in np.arange(n):
            var_explained_x =  var_explained_x + (np.sqrt(eigenvalues[i]) * eigenvectors[i,:])**2
        return eigenvalues[:n], eigenvectors[:n,:], var_explained_x
    else:
        var_explained_x = np.zeros(eigenvectors[0,:].shape) 
        for i in np.arange(nevals):
            var_explained_x =  var_explained_x + (np.sqrt(eigenvalues[i]) * eigenvectors[i,:])**2

        return eigenvalues, eigenvectors, var_explained_x

def _cal_coefficients(dist_x, dist_zeta, dist_xi, npoly_order, nkl, ngrid_x, nquad=150, rule=None):
    """
    Calculate coefficients of u_xi in equation (6). After KL expansion
    """
    ## calculate coefficients of u_zeta (3)
    x_hat_zeta, u_zeta = _pce_approx(dist_x, dist_zeta, npoly_order,nquad=nquad,rule=rule)
    eig_vals, eig_vectors, var_nkl = _sorted_eig(dist_x, ngrid_x, nkl)
    orth_poly_xi, norms = cp.orth_ttr(npoly_order, dist_xi, retall=True)
    u_xi = [] 

    for i, iPsi in enumerate(orth_poly_xi):
        ## order of polynomial iPsi of shape(i,j...)
        p_shape = iPsi.keys[-1]
        print(p_shape)
        p = np.sum(p_shape)

        ui_1 = factorial(p) * u_zeta[p] / norms[i] 
        ui_2 = 1
        for k,j in enumerate(p_shape): # k: xi_k, j: how many times for xi_k. -> xi_k^j in the highest order of polynomial
            ui_2 = ui_2 * (np.sqrt(eig_vals[k])*eig_vectors[k,:])**j
        u_xi.append(ui_1 * ui_2/ np.sqrt(var_nkl)**p )

    return u_xi, orth_poly_xi

nsamples = 1e4
p1,p2 = 0.01, 0.001
ngrid_x = 11
nkl = 2
npoly_order = 10#np.array([7]) #np.arange(1,10)#np.array([1,2,3,5])#
lc=0.5
q = np.linspace(0,1,100)[1:-1]


# exp_lambda = 3
# dist_x = cp.Exponential(exp_lambda)  ## Gamma(1,1)
# gamma_shape, gamma_scale = 2,1
# dist_x = cp.Gamma(gamma_shape,gamma_scale)
beta_a, beta_b =4,2 
dist_x = cp.Beta(beta_a, beta_b)
dist_n = cp.Normal()
### Calculate spatial correlation 

# evals_cumsum = np.cumsum(eigenvalues)
# evals_portion= evals_cumsum/np.sum(eigenvalues)

# xis = dist_n.sample(ngrid_x)# at most have ngrid_x eigenvalues-eigenvector pairs
# gamma_x, var_normalize = _gamma_process(eigenvalues, eigenvectors, nkl, xis)


# mom_names = ['mean', 'std', 'skewness', 'kurtosis']
# moms_exact = _central_moms(dist_x) #[beta_a/(beta_a+beta_b), (beta_a*beta_b)/((beta_a+beta_b)**2 * (beta_a+beta_b+1)),
        # 2*(beta_b-beta_a)*np.sqrt(beta_a+beta_b+1)/((beta_a+beta_b+2)*np.sqrt(beta_a*beta_b)),
        # 6*((beta_a-beta_b)**2 * (beta_a+beta_b+1)-beta_a*beta_b*(beta_a+beta_b+2))/ ( beta_a*beta_b * (beta_a+beta_b+2)*(beta_a+beta_b+3))]

# dist_zeta1   = cp.Normal()
dist_zeta1   = cp.Normal()
dist_zeta2   = cp.Uniform(0,1)
dist_xi      = cp.Iid(cp.Normal(), nkl)

u_xi, orth_poly_xi = _cal_coefficients(dist_x, dist_zeta1, dist_xi, npoly_order, nkl, ngrid_x, nquad=150, rule=None)
xi_samples = dist_xi.sample(100)
# print(xi_samples)
psi = np.array([ipoly(*xi_samples) for ipoly in orth_poly_xi]).T
ux = np.dot(psi,np.array(u_xi))


x_pdf_kde = gaussian_kde(ux[:,0])
x = np.linspace(0,1)
x_pdf = dist_x.pdf(x)
plt.plot(x,x_pdf_kde(x), '-r', label=r'dd')
plt.plot(x,x_pdf,'-.k', label=r'exact')
plt.show()
# samples_zeta1 = dist_zeta1.sample(nsamples)
# samples_zeta2 = dist_zeta2.sample(nsamples)
# x1_hat = []
# x2_hat = []
# x1_coeffs = []
# x2_coeffs = []
# x1_stats_hat = []
# x2_stats_hat = []
# x1_ecdf_hat = []
# x2_ecdf_hat = []
# x1_pdf_kde  = []
# x2_pdf_kde  = []

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
    # x1_pdf_kde.append(gaussian_kde(_x1_samples))
    # x2_pdf_kde.append(gaussian_kde(_x2_samples))

    # x1_hat.append(_x1_hat)
    # x1_coeffs.append(_x1_coeffs)
    # x2_hat.append(_x2_hat)
    # x2_coeffs.append(_x2_coeffs)
####
#### =========================================================================
#### =========================================================================
# #### =========================================================================
# figs, axes = plt.subplots(3,2,figsize=(9,12))
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

# for j,i in enumerate(np.arange(0,len(x1_ecdf_hat),1)):
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


#### =========================================================================
#### =========================================================================
#### =========================================================================
# print(x1_coeffs[-1])
# ##>>> pdf based on kde 
# figs, axes = plt.subplots(1,2,figsize=(16,6))
# x = dist_x.inv(q)
# x_pdf = dist_x.pdf(x)

# axes[0].plot(x, x_pdf, '-k' ,label=r'exact')
# axes[1].plot(x, x_pdf, '-k' ,label=r'exact')

# # for i in np.arange(0,len(x1_pdf_kde),2):
# for j,i in enumerate(np.arange(4,len(x1_pdf_kde),2)):
    # print('{}{}'.format(i,j))
    # x1_kde = x1_pdf_kde[i]
    # x2_kde = x2_pdf_kde[i]
    # axes[0].plot(x, x1_kde(x),linestyle=pltlinestyles[j], color=pltcolors[0], label=r'${}$-order'.format(ordinal(npoly_order[i])))
    # axes[1].plot(x, x2_kde(x),linestyle=pltlinestyles[j], color=pltcolors[1], label=r'${}$-order'.format(ordinal(npoly_order[i])))
# axes[0].legend()
# axes[1].legend()


# plt.tight_layout()
# plt.savefig('Figures/beta_pce_pdf.eps')
# plt.savefig('Figures/beta_pce_pdf.png')





























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
# # plt.plot(x, norm_pdf, label='$Ndef _estimate_pdf()({},{})$'.format(gumbel_loc, gumbel_scale))D
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




