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

    metamodel_class = 'PCE'
    quad_orders     = range(3,10)

    # ### 1. PCE model based on quadrature design points, fitting with GLK and OLS

    pce_fit_method  = 'GLK'
    metric_mse      = []
    metric_r2_adj   = []
    metric_mquantile= []

    valid_u = np.arange(3).reshape(3,1)
    valid_y = np.arange(1)

    for iquad_orders in quad_orders:
        poly_order = iquad_orders - 1
        ### ============ Get training points ============
        filename = 'DoE_QuadLeg{:d}.npy'.format(iquad_orders)
        data_set = np.load(os.path.join(simparams.data_dir, filename))
        print('  > {:<10s}: {:s}'.format('filename', filename))
        train_u  = data_set[0:ndim,:] 
        train_x  = data_set[ndim:2*ndim,:] 
        train_w  = data_set[-2,:]
        train_y  = data_set[-1,:]


        ### ============ Get Surrogate Model for each QoI============
        pce_model = museuq.PCE(poly_order, dist_zeta)
        pce_model.fit(train_u, train_y, w=train_w, method=pce_fit_method)

        pred_y  = pce_model.predict(valid_u)
        metric_mse.append(uq_metrics.mean_squared_error(valid_y, pred_y))
        metric_r2_adj.append(uq_metrics.r2_score_adj(valid_y, pred_y, len(pce_model.active_)))
        filename = 'DoE_QuadLeg{:d}_{:s}_{:s}_valid.npy'.format(iquad_orders, metamodel_class, pce_fit_method)
        np.save(os.path.join(simparams.data_dir, filename), np.vstack((valid_u, valid_y, pred_y)))

        valid_u = np.hstack((valid_u, train_u))
        valid_y = np.hstack((valid_y, train_y))
        if iquad_orders == quad_orders[0]:
            valid_u = valid_u[:,1:]
            valid_y = valid_y[1:]


        ## run MCS to get mquantile
        filename = 'DoE_McsE6R0.npy'
        data_set = np.load(os.path.join(simparams.data_dir, filename))
        samples_u= data_set[0:ndim,:]
        samples_x= data_set[ndim: 2*ndim,:]
        samples_y= pce_model.predict(samples_u)
        metric_mquantile.append(uq_metrics.mquantiles(samples_y, 1-1e-4))

    filename = 'mse_DoE_QuadLeg{:d}_{:s}_{:s}.npy'.format(iquad_orders, metamodel_class, pce_fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(metric_mse))

    filename = 'mquantile_DoE_QuadLeg{:d}_{:s}_{:s}.npy'.format(iquad_orders, metamodel_class, pce_fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(metric_mquantile))

    filename = 'r2_DoE_QuadLeg{:d}_{:s}_{:s}.npy'.format(iquad_orders, metamodel_class, pce_fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(metric_r2_adj))


    # ### 2. PCE model based on Optimal design points, fitting with OLS

    # alpha           = [1.0, 1.1, 1.3, 1.5]
    # opt_cri         = 'D'
    # pce_fit_method  = 'OLS'
    # for iquad_orders in quad_orders:
        # poly_order = iquad_orders - 1
        # for r in range(10):
            # basis = cp.orth_ttr(poly_order,dist_zeta)
            # for ia in alpha:
                # ### ============ Get training points ============
                # num_basis= min(int(len(basis)*ia), int(1E6))
                # filename = 'DoE_McsE6R{:d}_q{:d}_Opt{:s}{:d}.npy'.format(r, iquad_orders,opt_cri, num_basis)
                # data_set = np.load(os.path.join(simparams.data_dir,'DoE_McsE6_OPT_PCE', filename))
                # print('  > {:<10s}: {:s}'.format('filename', filename))
                # print('    {:<10s}: {}'.format('data shape', data_set.shape))
                # train_u  = data_set[1:3,:] 
                # train_x  = data_set[3:5,:] 
                # train_y  = data_set[5  ,:] 

                # ### ============ Get Surrogate Model for each QoI============
                # pce_model = museuq.PCE(poly_order, dist_zeta)
                # pce_model.fit(train_u, train_y, fit_method=pce_fit_method)

                # ### ============ Validating surrogate models at training points ============

                # filename = 'DoE_McsE6R{:d}.npy'.format(r)
                # data_set = np.load(os.path.join(simparams.data_dir, filename))
                # valid_u  = data_set[0:2,:]
                # valid_x  = data_set[2:4,:]
                # valid_y  = data_set[4  ,:]

                # pce_model.predict(valid_u, valid_y, metrics=metrics2cal,prob=upper_tail_probs,moment=moment2cal)
                # pce_valid_y_ecdf = uqhelpers.get_exceedance_data(pce_model.pred_y, 1e-5)

                # filename = 'DoE_McsE6{:d}_q{:d}_Opt{:s}{:d}_{:s}_{:s}_y.npy'.format(r, iquad_orders, opt_cri, num_basis,metamodel_class, pce_fit_method)
                # np.save(os.path.join(simparams.data_dir, filename), pce_model.pred_y[0])

                # filename = 'DoE_McsE6{:d}_q{:d}_Opt{:s}{:d}_{:s}_{:s}_y_ecdf.npy'.format(r, iquad_orders, opt_cri, num_basis,metamodel_class, pce_fit_method)
                # np.save(os.path.join(simparams.data_dir, filename), pce_valid_y_ecdf)

                # filename = 'DoE_McsE6{:d}_q{:d}_Opt{:s}{:d}_{:s}_{:s}_score.npy'.format(r, iquad_orders, opt_cri, num_basis,metamodel_class, pce_fit_method)
                # np.save(os.path.join(simparams.data_dir, filename), pce_model.scores[0])

if __name__ == '__main__':
    main()
