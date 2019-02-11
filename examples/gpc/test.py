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
np.set_printoptions(precision=2)
def model_solver(q):
    # return q**2
    return q[0]*np.e**-q[1]+1
    # return [q[0]*q[1], q[0]*np.e**-q[1]+1]

np.random.seed(100)
distribution = cp.Iid(cp.Normal(0, 1), 2)
# distribution = cp.Normal()
# absissas, weights = cp.generate_quadrature(2, distribution, rule="G")
absissas = distribution.sample(100)
expansion = cp.orth_ttr(2, distribution)
solves = [model_solver(absissa) for absissa in absissas.T]
# approx = cp.fit_quadrature(expansion, absissas, weights, solves)
f_hat, coeffs = cp.fit_regression(expansion, absissas, solves, retall=True)

# polynomials = cp.Poly([1, x, y])
# abscissas = [[-1,-1,1,1], [-1,1,-1,1]]
# evals = [0,1,1,2]
# f_hat, coeffs = cp.fit_regression(polynomials, abscissas, evals, retall=True)
print(expansion)
print(cp.around(f_hat,2))
print(coeffs)
print(f_hat.coeffs())
