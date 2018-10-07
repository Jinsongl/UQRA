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
    qoi2analysis = [0,1,2] 
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
    for f_obsx in f_obsX:
        print(f_obsx.shape) 
    for f in f_obsY:
        print(f.shape)
    for f in f_obsYstats:
        print(f.shape)
    # ## ------------------------------------------------------------------- ###
    # ## Fitting meta model  ###
    # ## ------------------------------------------------------------------- ###
    QoI = [2,4] ## The fifth statistics (max(abs)) of the third output
    quad_metamodel.fit_model(f_obsX[:len(dist_zeta)], f_obsYstats[5])
    # ## ------------------------------------------------------------------- ###
    # ## Cross Validation  ###
    # ## ------------------------------------------------------------------- ###
    # quad_metamodel.crossValidate(valiData)
    # print ('    Fitting Error:', quad_metamodel.fitError)
    # print ('    Cross Validation Error:', quad_metamodel.CVError)

    # ### ------------------------------------------------------------------- ###
    # ### Prediction  ###
    # ### ------------------------------------------------------------------- ###
    # quad_metamodel.predict(1E6,quad_simparam,R=10)


if __name__ == '__main__':
    main()

