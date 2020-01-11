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
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel,WhiteKernel)
from tqdm import tqdm
from museuq.utilities import helpers as uqhelpers
from museuq.utilities import metrics_collections as uq_metrics
from museuq.utilities import dataIO 
from museuq.environment import Kvitebjorn
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()

def main():
    ## ------------------------ Parameters set-up ----------------------- ###
    ndim        = 3
    model_name  = 'ishigami'
    ## 1. Choose Wiener-Askey scheme random variable
    dist_x      = cp.Iid(cp.Uniform(-np.pi, np.pi),ndim) 
    dist_zeta   = cp.Iid(cp.Uniform(-1, 1),ndim) 
    ## 2. If transformation needed, like Rosenblatt, need to be done here
    ## Perform Rosenblatt etc
    simparams = museuq.simParameters(model_name, dist_zeta)
    # simparams.set_error()  # no observation error for sdof
    simparams.info()


    #### ----------------------- Build PCE Surrogate Model -------------------- ###


    # ### 1. PCE model based on quadrature design points, fitting with GLK and OLS

    poly_order_max  = 15
    poly_orders     = np.arange(2, poly_order_max + 1)
    pce_fit_method  = 'GLK'
    metric_mse      = []
    metric_r2_adj   = []
    metric_mquantile= []

    u_valid = np.arange(3).reshape(3,1)
    y_valid = np.arange(1)

    for ipoly_order in poly_orders:
        ### ============ Get training points ============
        quad_order = ipoly_order + 1
        filename = 'DoE_QuadLeg{:d}.npy'.format(quad_order)
        data_set = np.load(os.path.join(simparams.data_dir, filename))
        print('  > {:<10s}: {:s}'.format('filename', filename))
        u_train  = data_set[0:ndim,:] 
        x_train  = data_set[ndim:2*ndim,:] 
        w_train  = data_set[-2,:]
        y_train  = data_set[-1,:]


        ### ============ Get Surrogate Model for each QoI============
        pce_model = museuq.PCE(ipoly_order, dist_zeta)
        pce_model.fit(u_train, y_train, w=w_train, method=pce_fit_method)

        y_pred  = pce_model.predict(u_valid)
        metric_mse.append(uq_metrics.mean_squared_error(y_valid, y_pred))
        metric_r2_adj.append(uq_metrics.r2_score_adj(y_valid, y_pred, len(pce_model.active_)))
        filename = 'DoE_QuadLeg{:d}_PCE_{:s}_valid.npy'.format(quad_order, pce_fit_method)
        np.save(os.path.join(simparams.data_dir, filename), np.vstack((u_valid, y_valid, y_pred)))

        u_valid = np.hstack((u_valid, u_train))
        y_valid = np.hstack((y_valid, y_train))
        if ipoly_order == poly_orders[0]:
            u_valid = u_valid[:,1:]
            y_valid = y_valid[1:]


        ## run MCS to get mquantile
        filename = 'DoE_McsE6R0.npy'
        data_set = np.load(os.path.join(simparams.data_dir, filename))
        u_samples= data_set[0:ndim,:]
        x_samples= data_set[ndim: 2*ndim,:]
        y_samples= pce_model.predict(u_samples)
        filename = 'DoE_McsE6R0_PCE{:d}_{:s}.npy'.format(ipoly_order, pce_fit_method)
        np.save(os.path.join(simparams.data_dir, filename), y_samples)
        metric_mquantile.append(uq_metrics.mquantiles(y_samples, 1-1e-4))


    filename = 'mse_DoE_QuadLeg{:d}_PCE_{:s}.npy'.format(ipoly_order, pce_fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(metric_mse))

    filename = 'mquantile_DoE_QuadLeg{:d}_PCE_{:s}.npy'.format(ipoly_order, pce_fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(metric_mquantile))

    filename = 'r2_DoE_QuadLeg{:d}_PCE_{:s}.npy'.format(ipoly_order, pce_fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(metric_r2_adj))


if __name__ == '__main__':
    main()
