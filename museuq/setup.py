#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

##            # |   zeta    | Wiener-Askey chaos | support
## # ==============================================================
## # Continuous | Gaussian  | Hermite-chaos      |  (-inf, inf)
##              | Gamma     | Laguerre-chaos     |  [0, inf ) 
##              | Beta      | Jacobi-chaos       |  [a,b] 
##              | Uniform   | Legendre-chaos     |  [a,b] 
## # --------------------------------------------------------------
## # Discrete   | Poisson   | 
##              | Binomial  | 
##              | - Binomial| 
##              | hypergeometric
## 
## dist_zeta = cp.Normal(0,1)  # shape=1, scale=1, shift=0
"""

# import context
import chaospy as cp, chaospy as cp
from .simParameters import simParameters

def setup(model_name, dist_zeta, prob_fails):
    simparam = simParameters(model_name, dist_zeta, prob_fails = prob_fails)
    simparam.update_dir()
    return simparam

if __name__ == 'main':
    setup()
