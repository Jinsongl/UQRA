#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""

import chaospy as cp
import numpy as np
def sff(w,c=2):
    s1 = c/(c**2 + w**2)
    s2 = 1/((w**2-2**2)**2 + (2*0.2*2)**2)
    s = s1*s2
    return s

w1 = 10;
w = np.linspace(-w1,w1,1e7)
s1 = sff(w)
a1 = np.sum(s1) * (w[1] - w[0])

dist_w = cp.Uniform(-w1, w1)
# dist_w = cp.Uniform(-1, 1)
for iquad in range(200):
    print('nquad={:d}'.format(iquad))
    absissas, weights = cp.generate_quadrature(iquad, dist_w, rule="c")
    absissas1, weights1 = cp.generate_quadrature(iquad, dist_w, rule="e")

# print(absissas, '\n', absissas1)
# print(weights, '\n', weights1)
    s2 = sff(absissas)
    s21 = sff(absissas1)
    a2 = sum(s2[0,:] * weights)
    a21 = sum(s21[0,:] * weights1)
    # print(a1,a2,a21)
    print('a1={:10.4f}, a2={:10.4f}, a21={:9.4f}'.format(a1,a2*2*w1,a21*2*w1))


