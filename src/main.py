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
# import environment as envi
from run_sim import run_sim
from solver.SDOF import deterministic_lin_sdof as solver_func 
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

def main():

    dist_zeta   = [cp.Exponential(1), cp.Exponential(1)] 
    qoi2analysis = [0,1] 
    ## ------------------------------------------------------------------- ###
    ##  Define simulation parameters  ###
    ## ------------------------------------------------------------------- ###
    quad_simparam = simParameter('Norway5_2','QUAD', [2,3], qoi2analysis)
    # quad_simparam.setNumOutputs(3)
    quad_simparam.set_seed([1,100])
    #### ------------------------------------------------------------------- ###
    #### Define meta model parameters  ###
    #### ------------------------------------------------------------------- ###
    quad_metamodel   = metaModel('PCE', [5,6], quad_simparam.doe_method, dist_zeta)
    # ## ------------------------------------------------------------------- ###
    # ## Define environmental conditions ###
    # ## ------------------------------------------------------------------- ###
    siteEnvi = environment(quad_simparam.site)
    # ## ------------------------------------------------------------------- ###
    # ## Run Simulations for training data  ###
    # ## ------------------------------------------------------------------- ###
    f_obsX, f_obsY, f_obsYstats = run_sim(siteEnvi, solver_func, quad_simparam, quad_metamodel)
    # ## ------------------------------------------------------------------- ###
    # ## Fitting meta model  ###
    # ## ------------------------------------------------------------------- ###
    istats, iqoi = 5, 0 # absmax
    datax = [x[:len(dist_zeta),:] for x in f_obsX] 
    datay = [y[:,istats, iqoi] for y in f_obsY] 
    dataw = [x[len(dist_zeta),:] for x in f_obsX] 
    
    quad_metamodel.fit_model(datax, datay, dataw)
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
    models_chosen = [[0,1],[1,0]] 
    quad_metamodel.predict(1E2,models_chosen,R=10)


if __name__ == '__main__':
    main()

