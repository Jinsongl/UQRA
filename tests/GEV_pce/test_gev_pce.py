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
np.set_printoptions(precision=3,suppress=True)
gumbel_loc, gumbel_scale= 3,4
dist_zeta = cp.Normal()
dist_unif = cp.Uniform(0,1)
dist_norm = cp.Normal(gumbel_loc, gumbel_scale)
dist_gumbel = cp.Logweibul(gumbel_scale, gumbel_loc)  # gumbel = logweibull
# dist_gumbel = cp.Exponential()
# dist_gumbel = cp.Normal(2,4)
# dist_weibull = cp.Weibull(scale=gumbel_scale, shift=gumbel_loc) 
####>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# #### Data explore
# x = np.arange(-15,15,0.1)
# zeta_pdf = dist_zeta.pdf(x)
# norm_pdf = dist_norm.pdf(x)
# gumbel_pdf = dist_gumbel.pdf(x)


# plt.figure()
# plt.plot(x, zeta_pdf, label='$N(0,1)$')
# plt.plot(x, norm_pdf, label='$N({},{})$'.format(gumbel_loc, gumbel_scale))
# plt.plot(x, gumbel_pdf, label='$Gum(scale={},loc={}))$'.format(gumbel_scale, gumbel_loc))
# plt.legend()
# plt.show()

####>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#### fit gumbel variable
npoly_order = 8
nquad = 15
orth_poly, norms = cp.orth_ttr(npoly_order, dist_zeta, retall=True)
zeta_quad, zeta_weight = cp.generate_quadrature(nquad,dist_zeta,rule='G')
unif_quad, unif_weight = cp.generate_quadrature(nquad,dist_unif,rule='G')
# print(unif_quad)

gumbel_quad = dist_gumbel.inv(unif_quad).T
gumbel_hat,coeffs = cp.fit_quadrature(orth_poly, dist_zeta.inv(unif_quad), unif_weight, gumbel_quad, norms=norms, retall=True)
print(coeffs)

gumbel_quad = dist_gumbel.inv(dist_zeta.cdf(zeta_quad)).T
gumbel_hat,coeffs = cp.fit_quadrature(orth_poly, zeta_quad, zeta_weight, gumbel_quad, norms=norms, retall=True)
print(coeffs)
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




