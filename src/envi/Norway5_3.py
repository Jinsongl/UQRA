#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""
Joint Distribution for Wind & Wave Dist(Uw, Hs, Tp)
Reference: 
        Norway 5:
        Li L, Gao Z, Moan T. Joint environmental data at five european offshore sites for design of combined wind and wave
        energy concepts. 32nd International Conference on Ocean, Offshore, and Arctic Engineering, Nantes, France, Paper
        No. OMAE2013-10156, 2013.
"""

import chaospy as cp
import numpy as np

########################################################
########################################################

def dist_Uw(isTrunc=False,bnd=[3,25]):
# Marginal distribution of 10-meter wind speed
    a_u, b_u = 2.029, 9.409
    if isTrunc:
        dist = cp.Truncweibull(bnd[0],bnd[1],a_u, b_u)
    else:
        dist = cp.Weibull(a_u, b_u)
    return dist


def distC_Hs(*var):
# Hs distribution conditional on Uw
    a1, a2, a3 = 2.136, 0.013, 1.709
    b1, b2, b3 = 1.816, 0.024, 1.787
    Uw = var[0]
    a_h = a1 + a2 * Uw ** a3
    b_h = b1 + b2 * Uw ** b3
    dist = cp.Weibull(a_h, b_h)
    return dist


def distC_Tp(*var):
    """
    Conditional distribution of Tp given Hs & Tp
    len(var) == 2, Tp|(Uw,Hs)
    """
    print var
    if len(var) == 2:
        theta, gamma = -0.255, 1.0
        e1, e2, e3 = 8.0, 1.938, 0.486
        f1, f2, f3 = 2.5, 3.001, 0.745
        k1, k2, k3 = -0.001, 0.316, -0.145 

        Uw,h = var 
        
        Tp_bar = e1 + e2 * h**e3 
        u_bar = f1 + f2 * h**f3
        niu_Tp = k1 + k2 * np.exp(h*k3)

        mu_Tp = Tp_bar * (1 + theta * ((Uw - u_bar)/u_bar)**gamma)

        mu_lnTp = np.log(mu_Tp / (np.sqrt(1 + niu_Tp**2)))
        sigma_lnTp = np.sqrt(np.log(niu_Tp**2 + 1))

        dist = cp.Lognormal(mu_lnTp, sigma_lnTp)
        
        return dist
    else:
        raise NotImplementedError("Conditional Tp distribution defined only for one or two conditional variables")
