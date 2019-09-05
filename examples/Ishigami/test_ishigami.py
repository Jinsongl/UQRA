#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""


import chaospy as cp
import numpy as np
from environment import environment
from metaModel import metaModel
from simParams import simParameter
from run_sim import run_sim
from solver.dynamic_models import deterministic_lin_sdof 
from solver.static_models import ishigami
# import utility.dataIO as dataIO
# import utility.getStats as getStats
# from meta.metaModel import *
# from solver.collec import *
import sys
sys.path.append("/MUSEUQ/src/")

import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import time

def main():
    dist_x = [cp.Uniform(-np.pi, np.pi), cp.Uniform(-np.pi, np.pi), cp.Uniform(-np.pi, np.pi)]
    dist_zeta   = [cp.Normal(), cp.Normal(),cp.Normal()] 
    # dist_zeta   = [cp.Exponential(1), cp.Exponential(1)] 
    # qoi2analysis = [0] 
    ## ------------------------------------------------------------------- ###
    ##  Define simulation parameters  ###
    ## ------------------------------------------------------------------- ###
    quad_simparam = simParameter('Norway5_2','MC', [1e4])
    # quad_simparam.setNumOutputs(3)
    quad_simparam.set_seed([1,100])
    #### ------------------------------------------------------------------- ###
    #### Define meta model parameters  ###
    #### ------------------------------------------------------------------- ###
    quad_metamodel   = metaModel('PCE', [5], quad_simparam.doe_method, dist_zeta)
    # ## ------------------------------------------------------------------- ###
    # ## Define environmental conditions ###
    # ## ------------------------------------------------------------------- ###
    siteEnvi = environment(quad_simparam.site)
    # ## ------------------------------------------------------------------- ###
    # ## Run Simulations for training data  ###
    # ## ------------------------------------------------------------------- ###
    f_obsX, f_obsY = run_sim(dist_x, ishigami, quad_simparam, quad_metamodel)
    # ## ------------------------------------------------------------------- ###
    # ## Fitting meta model  ###
    # ## ------------------------------------------------------------------- ###
    for a in f_obsX:
        print(a.shape)

    for a in f_obsY:
        print(a.shape)

    # for a in f_obsYstats:
        # print(a.shape)

    # istats, iqoi = 5, 0 # absmax
    datax = [x[:len(dist_zeta),:] for x in f_obsX] 
    datay = f_obsY #[y[:,istats, iqoi] for y in f_obsY] 
    # dataw = [x[len(dist_zeta),:] for x in f_obsX] 
    
    quad_metamodel.fit_model(datax, datay)
    # print(quad_metamodel.f_hats)
    # ## ------------------------------------------------------------------- ###
    # ## Cross Validation  ###
    # ## ------------------------------------------------------------------- ###
    # quad_metamodel.cross_validate(validatax, validatay)
    # print ('    Fitting Error:', quad_metamodel.fit_l2error)
    # print ('    Cross Validation Error:', quad_metamodel.cv_l2errors)

    # ### ------------------------------------------------------------------- ###
    # ### Prediction  ###
    # ### ------------------------------------------------------------------- ###
    # models_chosen = [[0,1],[1,0]] 
    # meta_pred = quad_metamodel.predict(1E2,R=10)




    metamodel1 = quad_metamodel.f_hats[0][0]
    datay1 = metamodel1(*datax[0])
    print(datay[0].shape)
    print(datay1.shape)

    n_bins = 100 
    fig, ax = plt.subplots(figsize=(8, 4))

# plot the cumulative histogram
    n, bins, patches = ax.hist(datay[0], n_bins, normed=1, histtype='step',
                               cumulative=True, label='True value')

    ax.hist(datay1.T, n_bins, normed=1, histtype='step',cumulative=True, label='Fitted value' )
# tidy up the figure
    ax.grid(True)
    ax.legend(loc='right')
    ax.set_title('Cumulative step histograms')
    ax.set_xlabel('Annual rainfall (mm)')
    ax.set_ylabel('Likelihood of occurrence')

    plt.show()




if __name__ == '__main__':
    main()

