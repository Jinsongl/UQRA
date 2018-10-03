#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""
Joint Distribution for Wind & Wave Dist(Hs, Tp)
Reference: 
        Norway 5:
        Li L, Gao Z, Moan T. Joint environmental data at five european offshore sites for design of combined wind and wave
        energy concepts. 32nd International Conference on Ocean, Offshore, and Arctic Engineering, Nantes, France, Paper
        No. OMAE2013-10156, 2013.
"""

import chaospy as cp
import numpy as np
# from genVar import *
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm

def dist_Hs(value, key='phy'):
    """
    Return Hs distribution based on given quantile or physical value
    """
    h0          = 5.0
    mu_LHM      = 0.871
    sigma_LHM   = 0.506
    alpha_HM    = 1.433
    beta_HM     = 2.547
    
    dist1 = cp.Lognormal(mu_LHM, sigma_LHM)
    dist2 = cp.Weibull(alpha_HM, beta_HM) 
    cdf0 = dist1.cdf(h0)

    # if kwargs is not None:
        # if len(kwargs) != 1:
            # raise ValueError("Only one key-value pair is expected")
        # key, value = kwargs.popitem()
    if key == "phy":
        dist = dist1 if value <= h0 else dist2
    elif key == "ppf":
        dist = dist1 if value <= cdf0 else dist2 
    else:
        raise NotImplementedError("Key:" + str(key) + " is not defined")
    # else:
        # raise NotImplementedError("kwargs is not defined")
    return dist

def distC_Tp(*args):
    """
    Conditional distribution of Tp given Hs
    len(var) == 1, Tp|Hs
    """
    if len(args) == 1: 
        c1, c2, c3 = 1.886, 0.365, 0.312
        d1, d2, d3 = 0.001, 0.105, -0.264
        h = args[0]
        mu_LTC = c1 + c2 * h ** c3

        sigma_LTC = (d1 + d2 * np.exp(d3 * h))** 0.5

        dist = cp.Lognormal(mu_LTC, sigma_LTC)
        return dist
    else:
        raise NotImplementedError("Conditional Tp distribution defined only for one or two conditional variables")

