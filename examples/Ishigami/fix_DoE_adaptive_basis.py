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
from tqdm import tqdm
from museuq.utilities import helpers as uqhelpers
from museuq.utilities import metrics_collections as uq_metrics
from sklearn.model_selection import KFold
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()


def is_stopping_met(mse, mquantiles, ipoly_order, **kwargs):
    """
    mse: iterable type with size 2
    mquantiles: iterable type with size 3

    """
    stopping_mse        = kwargs.get('stopping_mse', 1e-3)
    stopping_mquantile  = kwargs.get('stopping_mquantile', 0.05)
    poly_order_max      = kwargs.get('poly_order_max', 20)

    is_poly_order_met= ipoly_order <= poly_order_max
    if is_poly_order_met: 
        ## keep running if not enought data as long as p < p_max
        if len(mse) <= 2 or len(mquantiles) <= 3:
            return False
        is_mse_met       = all(imse < stopping_mse for imse in mse)
        mquantiles_diff  = abs(mquantiles[1:] - mquantiles[:-1]) /  mquantiles[:-1]
        is_mquantile_met = all(iquantile < stopping_mquantile for iquantile in mquantiles)
        if is_mse_met and is_mquantile_met:
            return True
        else:
            return False
    else:
        return True



def main():
    ## ------------------------ Parameters set-up ----------------------- ###
    ndim        = 3
    model_name  = 'Ishigami'
    ## 1. Choose Wiener-Askey scheme random variable
    dist_x      = cp.Iid(cp.Uniform(-np.pi, np.pi),ndim) 
    dist_zeta   = cp.Iid(cp.Uniform(-1, 1),ndim) 
    ## 2. If transformation needed, like Rosenblatt, need to be done here
    ## Perform Rosenblatt etc
    simparams = museuq.simParameters(model_name, dist_zeta)
    # simparams.set_error()  # no observation error for sdof
    simparams.info()


    ### ============ Stopping Criteria ============
    poly_order_max  = 3
    poly_orders     = np.arange(2, poly_order_max + 1)

    pce_fit_method  = 'OLSLARS'
    metric_mse_loo  = []
    metric_r2_adj   = []
    metric_mquantile= []

    ### ============ Get training points ============
    filename = 'DoE_LhsE3R0.npy'
    data_set = np.load(os.path.join(simparams.data_dir, filename))
    print('  > {:<10s}: {:s}'.format('filename', filename))
    u_data  = data_set[0:ndim,:]
    x_data  = data_set[ndim:2*ndim,:]
    w_data  = data_set[-2,:]
    y_data  = data_set[-1,:]
    ### ============ Get Surrogate Model for each QoI============
    for ipoly_order in tqdm(poly_orders, ascii=True, desc="   - "):
        pce_model = museuq.PCE(ipoly_order, dist_zeta)
        pce_model.fit(u_data, y_data, method=pce_fit_method)
        y_pred  = pce_model.predict(u_data)
        metric_mse_loo.append(pce_model.cv_error)
        metric_r2_adj.append(uq_metrics.r2_score_adj(y_data, y_pred, len(pce_model.active_)))

        ### run MCS to get mquantile
        filename = 'DoE_McsE6R0.npy'
        data_set = np.load(os.path.join(simparams.data_dir, filename))
        u_samples= data_set[0:ndim,:]
        x_samples= data_set[ndim: 2*ndim,:]
        y_samples= pce_model.predict(u_samples)
        filename = 'DoE_McsE6R0_PCE{:d}_{:s}.npy'.format(ipoly_order, pce_fit_method)
        np.save(os.path.join(simparams.data_dir, filename), y_samples)
        metric_mquantile.append(uq_metrics.mquantiles(y_samples, 1-1e-4))

    filename = 'mse_DoE_LhsE3_PCE{:d}_{:s}.npy'.format(ipoly_order, pce_fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(metric_mse_loo))

    filename = 'mquantile_DoE_LhsE3_PCE{:d}_{:s}.npy'.format(ipoly_order, pce_fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(metric_mquantile))

    filename = 'r2_DoE_LhsE3_PCE{:d}_{:s}.npy'.format(ipoly_order, pce_fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(metric_r2_adj))

    ### ============ Get Surrogate Model for each QoI============
    # pce_fit_method  = 'OLSLARS'
    # r2_adj     = []
    # mquantiles = []
    # for ipoly_order in poly_orders:
        # pce_model = museuq.PCE(ipoly_order, dist_zeta)
        # pce_model.fit(u_data.T, y_data, method=pce_fit_method)
        # y_pred    = pce_model.predict(u_data.T)
        # r2_adj.append(uq_metrics.r2_score_adj(y_data, y_pred, len(pce_model.active_)))

        # # run MCS to get mquantile
        # filename = 'DoE_McsE6R0.npy'
        # data_set = np.load(os.path.join(simparams.data_dir, filename))
        # u_samples= data_set[0:ndim,:]
        # x_samples= data_set[ndim: 2*ndim,:]
        # y_samples= pce_model.predict(u_samples)
        # mquantiles.append(uq_metrics.mquantiles(y_samples, 1-1e-4))


    # filename = 'mquantile_DoE_LhsE3{:d}_PCE_{:s}.npy'.format(ipoly_order, pce_fit_method)
    # np.save(os.path.join(simparams.data_dir, filename), np.array(mquantiles))

    # filename = 'r2_DoE_LhsE3{:d}_PCE_{:s}.npy'.format(ipoly_order, pce_fit_method)
    # np.save(os.path.join(simparams.data_dir, filename), np.array(r2_adj))



if __name__ == '__main__':
    main()
