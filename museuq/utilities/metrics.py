#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import numpy as np
from scipy.stats.mstats import mquantiles 
from numpy.linalg import norm

# ord	norm for matrices	        norm for vectors
# None	Frobenius norm	                2-norm
# ‘fro’	Frobenius norm	                –
# ‘nuc’	nuclear norm	                –
# inf	max(sum(abs(x), axis=1))	max(abs(x))
# -inf	min(sum(abs(x), axis=1))	min(abs(x))
# 0	–	                        sum(x != 0)
# 1	max(sum(abs(x), axis=0))	as below
# -1	min(sum(abs(x), axis=0))	as below
# 2	2-norm (largest sing. value)	as below
# -2	smallest singular value	i       as below
# other	–	                        sum(abs(x)**ord)**(1./ord)

def moments(self,x, orders=np.arange(7)):
    x = np.squeeze(x)
    assert x.ndim == 1
    xx = np.array([x**i for i in orders])
    xx_moms = np.mean(xx, axis=1)
    return xx_moms
