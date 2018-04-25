#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""

def mc_wrapper(func,dist,n_samp,n_repeat,*args):
    """
    wrapper for crude monte carlo simulation
    arguments:
    func: functiont to run monte carlo
    dist: chaospy dist class, distribution for the input x, could be multidimension
    n_samp: number of samples for each trial
    n_repeat: number of repated times
    rule: monte carlo sampling rule
    """
    
    print('\tRunning Monte Carlo for function {0}, sampling rule: {1}}'.format(func.__name__,rule ))
    print('\tNumber of samples:{0}, Number of repeat:{1}'.format(n_samp, n_repeat))
    n_rands = len(dist)
    x_samples = dist.sample(n_samp);

    y = func(*x_samples,*args)
    assert len(x_train) == n_samp


