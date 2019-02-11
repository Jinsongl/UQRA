#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""
Codes in Chaospy document
"""

import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt



def foo(coord, param):
    return param[0] * np.e ** (-param[1] * coord)

coord = np.linspace(0,10,100)
dist = cp.J(cp.Uniform(1,2), cp.Uniform(0.1,0.2))


# samples = dist.sample(50)
# evals = np.array([foo(coord, sample) for sample in samples.T]).T
# plt.plot(coord, evals)
# plt.show()
# evals = [foo(coord, sample) for sample in samples.T]

# # Monte Carlo simulaiton

samples = dist.sample(1000, 'H')
evals = [foo(coord, sample) for sample in samples.T]

# expected = np.mean(evals,0)
# deviation = np.std(evals,0)

# # Point Collocation method

orthPoly = cp.orth_ttr(8, dist)
# foo_hat = cp.fit_regression(orthPoly, samples, evals)
foo_hat = cp.fit_regression(orthPoly, samples, evals)
# print(dir(foo_hat))
print('coeffs:\n{}'.format(foo_hat.coeffs()))
print('dim:\n{}'.format(foo_hat.dim))
print('keys:\n{}'.format(foo_hat.keys))
print('shape:\n{}'.format(foo_hat.shape))
foo_hat(samples)

## how to print coeffs

# # Pseudo-spectral Projection

x,w = cp.generate_quadrature(8,dist,rule='C')
evals = [foo(coord, val) for val in x.T]
# foo_hat = cp.fit_quadrature(orthPoly, x, w, evals)

print(x.shape)
print(w.shape)
print(np.asarray(evals).shape)
# y_hat = foo_hat(*samples)
# print(y_hat.shape)


# cp.seed(1000)
# dist = cp.Normal()
# x = cp.variable(1)
# poly = cp.Poly([x,x**2])
# print(poly)
# qoi_dist = cp.QoI_Dist(poly, dist)
# print(qoi_dist)
