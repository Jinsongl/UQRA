#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

## >>> 1. Choose Wiener-Askey scheme random variable
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
"""
import numpy as np
## ------------------------------------------------------------------- ###
##  Parameters set-up 
## ------------------------------------------------------------------- ###
prob_fails          = 1e-1              # failure probabilities
# data_train_params = [[1e6], 'R']      # nsamples_test, sample_rule
# data_test_params  = [1e7, 10, 'R']    # nsamples_test, nrepeat, sample_rule
MODEL_NAME          = 'Ishigami'
# MODEL_NAME        = 'BENCH1'

## ------------------------------------------------------------------- ###
##  Define Solver parameters ###
## ------------------------------------------------------------------- ###
## 
## >>> 1. Choose Wiener-Askey scheme random variable
## dist_zeta = cp.Normal(0,1)  # shape=1, scale=1, shift=0
dist_zeta = cp.Uniform(0,1)
dist_zeta = cp.Iid(dist_zeta,3) 

## >>> 2. If transformation needed, like Rosenblatt, need to be done here
## Perform Rosenblatt etc

## >>> 3. Define independent random variable in physical problems
# dist_x = cp.Normal(5,2) # normal mean = 0, normal std=0.25
dist_x = cp.Uniform(-np.pi, np.pi)
dist_x = cp.Iid(dist_x,3) 
