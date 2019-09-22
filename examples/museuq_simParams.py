#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.
import context
import museuq
import numpy as np, chaospy as cp, os, sys
import warnings
from museuq.utilities import helpers as museuq_helpers 
from museuq.utilities import metrics as museuq_metrics
from museuq.utilities import dataIO as museuq_dataio 
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()

def main():
    ## ------------------------ Parameters set-up ----------------------- ###
    prob_fails  = 1e-3              # failure probabilities
    model_name  = 'linear_oscillator'
    ## 1. Choose Wiener-Askey scheme random variable
    # dist_zeta = cp.Uniform(-1,1)
    # dist_zeta = cp.Gamma(4,1)
    dist_zeta = cp.Iid(cp.Normal(),2) 

    ## 2. If transformation needed, like Rosenblatt, need to be done here
    ## Perform Rosenblatt etc

    ## 3. Define independent random variable in physical problems
    # dist_x = cp.Uniform(-np.pi, np.pi)
    dist_x = cp.Iid(cp.Normal(3, 5),2) 

    error_params=None
    simparams = museuq.setup(model_name, dist_zeta, dist_x, prob_fails)
    simparams.set_error(error_params)
    simparams.disp()

if __name__ == '__main__':
    main()
