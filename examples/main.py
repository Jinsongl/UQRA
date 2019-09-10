#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import context
import museuq
import numpy as np, chaospy as cp, os, sys
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

def main():

    sys.stdout  = museuq.utilities.classes.Logger()
    ## ------------------------ Parameters set-up ----------------------- ###
    prob_fails  = 1e-1              # failure probabilities
    model_name  = 'Ishigami'

    ## ------------------------ Define Solver parameters ---------------------- ###
    ## 1. Choose Wiener-Askey scheme random variable
    # dist_zeta = cp.Uniform(0,1)
    dist_zeta = cp.Iid(cp.Uniform(0,1),3) 

    ## 2. If transformation needed, like Rosenblatt, need to be done here
    ## Perform Rosenblatt etc

    ## 3. Define independent random variable in physical problems
    # dist_x = cp.Uniform(-np.pi, np.pi)
    dist_x = cp.Iid(cp.Uniform(-np.pi, np.pi),3) 
    simparams = museuq.setup(model_name, dist_zeta, dist_x, prob_fails)

    error_params=None
    simparams.set_error(error_params)
    print(simparams.model_name)

    museuq.run_doe(simparams)

if __name__ == '__main__':
    main()
