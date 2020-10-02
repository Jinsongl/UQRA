#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import uqra
import numpy as np, os, sys
import scipy.stats as stats
from tqdm import tqdm
# warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()

class Data():
    pass

def main():

    ## ------------------------ Displaying set up ------------------- ###
    print('\n#################################################################################')
    print(' >>>  Running UQRA '.format(__file__))
    print('#################################################################################\n')
    np.random.seed(100)
    np.set_printoptions(precision=4)
    np.set_printoptions(threshold=8)
    np.set_printoptions(suppress=True)
    np.random.seed(100)

    ## ------------------------ Simulation Parameters ----------------- ###
    n_cand = int(1e5)
    ndim   = 2
    alphas = [1.2, 2.0]
    pce_degs = np.arange(2,4)
    u_dist = [stats.uniform(),]*ndim
    data   = Data()
    for ialpha in alphas:
        for deg in pce_degs:
            kappa = []
            pce = uqra.Legendre(d=ndim, deg=deg)

            ## MCS
            x = uqra.MCS(u_dist).samples(size=int(ialpha*pce.num_basis))
            X = pce.vandermonde(x)
            ## condition number, kappa = max(svd)/min(svd)
            _, sig_value, _ = np.linalg.svd(X)
            kappa.append(max(abs(sig_value)) / min(abs(sig_value)))

            ## MCS-D

            ## MCS-S

            ## CLS
            x = uqra.CLS('CLS1', ndim).samples(size=int(ialpha*pce.num_basis))
            X = pce.vandermonde(x)
            ## condition number, kappa = max(svd)/min(svd)
            _, sig_value, _ = np.linalg.svd(X)
            kappa.append(max(abs(sig_value)) / min(abs(sig_value)))
            ## CLS-D
            ## CLS-S

            print(kappa)

if __name__ == '__main__':
    main()
