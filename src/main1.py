#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import chaospy as cp
import numpy as np
import envi.environment as envi
import sampling.genVar as spl 
import solver.SDOF as slv 
import utility.dataIO as dataIO
# import utility.getStats as getStats
from meta.metaModel import *
from solver.collec import *


import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D



def main():
    
    # # # Underlying random variables for selected Wiener-Askey polynomial.
    dist_zeta   = [cp.Exponential(1), cp.Exponential(1)] 
    data = np.genfromtxt("validateData.csv", delimiter=',') 
    valiData = Data(data[:,:2],data[:,-1])


    # ## ------------------------------------------------------------------- ###
    # ##  Define simulation parameters  ###
    # ## ------------------------------------------------------------------- ###
    # QuadParams = simParameter('Norway5_2','QUAD', 5)
    # QuadParams.setNumOutputs(3)
    # QuadParams.setSeed([1,100])
    # #### ------------------------------------------------------------------- ###
    # #### Define meta model parameters  ###
    # #### ------------------------------------------------------------------- ###
    # QuadModel   = metaModel(QuadParams.site,    'QUAD', 5, dist_zeta)
    # # ## ------------------------------------------------------------------- ###
    # # ## Define environmental conditions ###
    # # ## ------------------------------------------------------------------- ###
    # siteEnvi = envi.environment(QuadParams.site)
    # # ## ------------------------------------------------------------------- ###
    # # ## Run Simulations for training data  ###
    # # ## ------------------------------------------------------------------- ###
    # f_obsX, f_obsY = RunSim(siteEnvi,slv.SDOF,QuadParams,  QuadModel)
    # # ## ------------------------------------------------------------------- ###
    # # ## Fitting meta model  ###
    # # ## ------------------------------------------------------------------- ###
    # QoI = [2,4] ## The fifth statistics (max(abs)) of the third output
    # trainData1 = Data(f_obsX[:,:-len(dist_zeta)], f_obsY[QoI[0],:,QoI[1]])
    # QuadModel.fitModel(trainData1)
    # # ## ------------------------------------------------------------------- ###
    # # ## Cross Validation  ###
    # # ## ------------------------------------------------------------------- ###
    # QuadModel.crossValidate(valiData)
    # print '    Fitting Error:', QuadModel.fitError
    # print '    Cross Validation Error:', QuadModel.CVError

    # # ### ------------------------------------------------------------------- ###
    # # ### Prediction  ###
    # # ### ------------------------------------------------------------------- ###
    # QuadModel.predict(1E6,QuadParams,R=10)






    ## ------------------------------------------------------------------- ###
    ##  Define simulation parameters  ###
    ## ------------------------------------------------------------------- ###

    q = spl.getQuant(1E7,base=10.0, alpha=2) 
    QuantParams = simParameter('Norway5_3','QUANT',[[q,q]])
    # q1 = np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-2, 0.1,0.3,0.5,0.7,0.9,1-1e-2,1-1e-3,1-1e-4,1-1e-5, 1-1e-6])
    # q2 = np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-2, 0.1,0.3,0.5,0.7,0.9,1-1e-2,1-1e-3,1-1e-4,1-1e-5, 1-1e-6])
    # QuantParams = simParameter('Norway5_2','QUANT',[[q1,q2]])
    QuantParams.setNumOutputs(3)
    QuantParams.setSeed([1,100])
    ### ------------------------------------------------------------------- ###
    ### Define meta model parameters  ###
    ### ------------------------------------------------------------------- ###
    QuantModel  = metaModel(QuantParams.site,   'QUANT',[5,], dist_zeta)
    ## ------------------------------------------------------------------- ###
    ## Define environmental conditions ###
    ## ------------------------------------------------------------------- ###
    siteEnvi = envi.environment(QuantParams.site)
    print(siteEnvi.getdistFun())
    ## ------------------------------------------------------------------- ###
    ## Run Simulations for training data  ###
    ## ------------------------------------------------------------------- ###
    # f_obsX, f_obsY = RunSim(siteEnvi,slv.SDOF,QuantParams, QuantModel)
    ## ------------------------------------------------------------------- ###
    ## Fitting meta model  ###
    ## ------------------------------------------------------------------- ###
    # QoI = [2,4] ## The fifth statistics (max(abs)) of the third output
    # trainData = Data(f_obsX[:,:-len(dist_zeta)], f_obsY[QoI[0],:,QoI[1]])
    # QuantModel.fitModel(trainData)
    ## ------------------------------------------------------------------- ###
    ## Cross Validation  ###
    ## ------------------------------------------------------------------- ###
    # QuantModel.crossValidate(valiData)
    # print '    Fitting Error:', QuantModel.fitError
    # print '    Cross Validation Error:', QuantModel.CVError
    # ### ------------------------------------------------------------------- ###
    # ### Prediction  ###
    # ### ------------------------------------------------------------------- ###
    # QuantModel.predict(1E4,QuantParams,R=2)


 



    # ## ------------------------------------------------------------------- ###
    # ##  Define simulation parameters  ###
    # ## ------------------------------------------------------------------- ###
    # MCParams = simParameter('Norway5_2','MC', 50)
    # MCParams.setNumOutputs(3)
    # MCParams.setSeed([1,100])
    # # #### ------------------------------------------------------------------- ###
    # # #### Define meta model parameters  ###
    # # #### ------------------------------------------------------------------- ###
    # MCModel     = metaModel(MCParams.site,      'MC',   [5,], dist_zeta)
    # # # ## ------------------------------------------------------------------- ###
    # # # ## Define environmental conditions ###
    # # # ## ------------------------------------------------------------------- ###
    # siteEnvi = envi.environment(MCParams.site)
    # # # ## ------------------------------------------------------------------- ###
    # # # ## Run Simulations for training data  ###
    # # # ## ------------------------------------------------------------------- ###
    # f_obsX, f_obsY = RunSim(siteEnvi,slv.SDOF,MCParams,    MCModel)
    # # # ## ------------------------------------------------------------------- ###
    # # # ## Fitting meta model  ###
    # # # ## ------------------------------------------------------------------- ###

    # QoI = [2,4] ## The fifth statistics (max(abs)) of the third output
    # trainData = Data(f_obsX[:,:-len(dist_zeta)], f_obsY[2,:,4])
    # MCModel.fitModel(trainData)
    # # # ## ------------------------------------------------------------------- ###
    # # # ## Cross Validation  ###
    # # # ## ------------------------------------------------------------------- ###
    # MCModel.crossValidate(valiData)
    # print '    Fitting Error:', MCModel.fitError
    # print '    Cross Validation Error:', MCModel.CVError
    # # # ### ------------------------------------------------------------------- ###
    # # # ### Prediction  ###
    # # # ### ------------------------------------------------------------------- ###
    # MCModel.predict(1E6,MCParams,R=10)







    # ## ------------------------------------------------------------------- ###
    # ##  Define simulation parameters  ###
    # ## ------------------------------------------------------------------- ###
    # EDParams = simParameter('Norway5_2','ED',1E3)
    # EDParams.setNumOutputs(3)
    # EDParams.setSeed([1,100])
    # #### ------------------------------------------------------------------- ###
    # #### Define meta model parameters  ###
    # #### ------------------------------------------------------------------- ###
    # EDModel     = metaModel(EDParams.site,      'ED',   5, dist_zeta)
    # # ## ------------------------------------------------------------------- ###
    # # ## Define environmental conditions ###
    # # ## ------------------------------------------------------------------- ###
    # siteEnvi = envi.environment(EDParams.site)
    # # ## ------------------------------------------------------------------- ###
    # # ## Run Simulations for training data  ###
    # # ## ------------------------------------------------------------------- ###
    # f_obsX, f_obsY = RunSim(siteEnvi,slv.SDOF,EDParams,    EDModel)
    # # ## ------------------------------------------------------------------- ###
    # # ## Fitting meta model  ###
    # # ## ------------------------------------------------------------------- ###
    # QoI = [2,4] ## The fifth statistics (max(abs)) of the third output
    # trainData = Data(f_obsX[:,:-len(dist_zeta)], f_obsY[2,:,4])
    # EDModel.fitModel(trainData)
    # # ## ------------------------------------------------------------------- ###
    # # ## Cross Validation  ###
    # # ## ------------------------------------------------------------------- ###
    # EDModel.crossValidate(valiData)
    # print '    Fitting Error:', EDModel.fitError
    # print '    Cross Validation Error:', EDModel.CVError
    # # ### ------------------------------------------------------------------- ###
    # # ### Prediction  ###
    # # ### ------------------------------------------------------------------- ###
    # EDModel.predict(1E6,EDParams,R=10)

if __name__ == '__main__':
    main()



    ##########################################################################
    ########### Generate random variables ########## 
