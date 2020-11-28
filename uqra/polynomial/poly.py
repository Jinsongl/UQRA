#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import numpy as np
import itertools, math
from ._polybase import PolyBase
import scipy.stats as stats
from .hermite import Hermite
from .legendre import Legendre


def orthogonal(ndim, p, poly_name):
    poly_name = poly_name.lower()
    if poly_name   == 'leg':
        orth_basis = Legendre(d=ndim,deg=p)
    elif poly_name == 'hem':
        orth_basis = Hermite(d=ndim,deg=p, hem_type='physicists')
    elif poly_name == 'heme':
        orth_basis = Hermite(d=ndim,deg=p, hem_type='probabilists')
    else:
        raise NotImplementedError
    return orth_basis



