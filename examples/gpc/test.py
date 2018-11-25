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
import math

import numpy.linalg as LA
ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(math.floor(n/10)%10!=1)*(n%10<4)*n%10::4])

def _correlation_x(x1,x2,lc):
    xx1, xx2 = np.meshgrid(x1,x2)
    dx = xx1 - xx2
    res = np.exp(-(dx/lc)**2)
    return res
def _central_moms(dist, n=np.arange(1,5)):
    """
    Calculate the first central moments of distribution dist 
    """
    mu = [dist.mom(i) for i in n]
    res = []
    mu1 = mu[0]
    mu2 = mu[1] - mu1**2
    mu3 = mu[2] - 3 * mu[0] * mu[1] + 2 * mu[0]**3 
    mu4 = mu[3] - 4* mu[0] * mu[2] + 6 * mu[0]**2*mu[1] - 3 * mu[0]**4 
    sigma = np.sqrt(mu2)
    res = [mu1/1, sigma, mu3/sigma**3, mu4/sigma**4]
    return res
ngrid_x=11

# dist_x = cp.J([cp.Normal(),]*2)
dist = cp.Normal()
dist_x = cp.Iid(dist,2)
orth_poly, norms = cp.orth_ttr(3, dist_x, retall=True)
print(orth_poly)
print(orth_poly[0]*orth_poly[2])
# dist_zeta = cp.Normal(2,4)
# dist_x = cp.Beta(4,2)
# x_grids = np.linspace(dist_x.range()[0], dist_x.range()[1], ngrid_x)
# # print(x_grids)
# corr_grids_x = _correlation_x(x_grids,x_grids,0.5)
# print(corr_grids_x)

# eigenvalues,eigenvectors= LA.eig(corr_grids_x)
# idx = eigenvalues.argsort()[::-1]   
# eigenvalues = eigenvalues[idx]
# eigenvectors = eigenvectors[:,idx]
# print(eigenvalues)
# print(eigenvectors)

# npoly_order= 5
# orth_poly, norms = cp.orth_ttr(npoly_order, dist_zeta, retall=True)
# zeta_quad, zeta_weight = cp.generate_quadrature(8, dist_zeta, rule="C")
# x_quad = dist_x.inv(dist_zeta.cdf(zeta_quad))
# print(zeta_quad.shape, zeta_weight.shape,x_quad.shape)
# x_hat, coeffs0 = cp.fit_quadrature(orth_poly, zeta_quad, zeta_weight, x_quad.T, norms=norms,retall=True)
# x_hat, coeffs1 = cp.fit_quadrature(orth_poly, zeta_quad, zeta_weight, x_quad.T, retall=True)
# # print(x_hat)
# print(norms)
# # print(coeffs)
# # print(norms.reshape(len(norms),-1) * coeffs)

# print(coeffs0/coeffs1)
