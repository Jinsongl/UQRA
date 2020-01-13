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

def main():
    ## ------------------------ Parameters set-up ----------------------- ###
    ndim        = 3
    dist_x      = cp.Iid(cp.Uniform(-np.pi, np.pi),ndim) 
    dist_zeta   = cp.Iid(cp.Uniform(-1, 1),ndim) 
    simparams   = museuq.simParameters('Ishigami', dist_zeta)
    solver      = museuq.Ishigami()
    # fit_method  = 'OLS'

    ### ============ Adaptive parameters ============
    plim        = (2,15)
    n_budget    = 1000
    n_newsamples= 10
    simparams.set_adaptive_parameters(n_budget=n_budget, plim=plim, r2_bound=0.9, q_bound=0.05)
    simparams.info()

    ### ============ Stopping Criteria ============
    poly_orders     = np.arange(plim[0], plim[1])
    fit_method      = 'OLSLARS'
    poly_order      = plim[0]
    error_loo       = []
    mquantiles      = []
    r2_score_adj    = []
    f_hat           = None

    ### ============ Get training points ============
    filename= 'DoE_LhsE3R0.npy'
    data_set= np.load(os.path.join(simparams.data_dir, filename))
    u_train = data_set[0:ndim,:]
    x_train = data_set[ndim:2*ndim,:]
    w_train = data_set[-2,:]
    y_train = data_set[-1,:]
    n_eval  = u_train.shape[1]


    while simparams.is_adaptive_continue(n_eval, poly_order=poly_order,
            r2_adj=r2_score_adj, mquantiles=mquantiles):
        print('==> Adaptive simulation continue...')
        ### ============ Build Surrogate Model ============
        quad_order  = poly_order + 1
        pce_model   = museuq.PCE(poly_order, dist_zeta)
        pce_model.fit(u_train, y_train, w=w_train, method=fit_method)
        y_pred      = pce_model.predict(u_train)

        filename = 'DoE_McsE6R0.npy'
        data_set = np.load(os.path.join(simparams.data_dir, filename))
        u_samples= data_set[0:ndim,:]
        x_samples= data_set[ndim: 2*ndim,:]
        y_samples= pce_model.predict(u_samples)

        ### ============ updating parameters ============
        # error_mse.append(uq_metrics.mean_squared_error(y_valid, y_pred))
        error_loo.append(pce_model.cv_error)
        r2_score_adj.append(uq_metrics.r2_score_adj(y_train, y_pred, len(pce_model.active_)))
        mquantiles.append(uq_metrics.mquantiles(y_samples, 1-1e-4))
        poly_order += 1
        f_hat       = pce_model

    poly_order -= 1
    print('------------------------------------------------------------')
    print('>>> Adaptive simulation done:')
    print('------------------------------------------------------------')
    print(' - {:<25s} : {}'.format('Polynomial order (p)', poly_order))
    print(' - {:<25s} : {}'.format('Active basis', f_hat.active_))
    print(' - {:<25s} : {}'.format('# Evaluations ', n_eval))
    print(' - {:<25s} : {}'.format('R2_adjusted ', np.around(r2_score_adj, 2)))
    print(' - {:<25s} : {}'.format('mquantiles', np.around(np.squeeze(np.array(mquantiles)), 2)))

    filename = 'mquantile_DoE_LhsE3_PCE{:d}_{:s}_path.npy'.format(poly_order, fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(mquantiles))

    filename = 'r2_DoE_DoE_LhsE3_PCE{:d}_{:s}_path.npy'.format(poly_order, fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(r2_score_adj))

    filename = 'cv_error_DoE_DoE_LhsE3_PCE{:d}_{:s}_path.npy'.format(poly_order, fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(error_loo))


    mquantiles = []
    for r in tqdm(range(10), ascii=True, desc="   - " ):
        filename = 'DoE_McsE6R{:d}.npy'.format(r)
        data_set = np.load(os.path.join(simparams.data_dir, filename))
        u_samples= data_set[0:ndim,:]
        x_samples= data_set[ndim: 2*ndim,:]
        y_samples= f_hat.predict(u_samples)
        mquantiles.append(uq_metrics.mquantiles(y_samples, [1-1e-4, 1-1e-5, 1-1e-6]))
        filename = 'DoE_McsE6R{:d}_PCE{:d}_{:s}.npy'.format(r, poly_order, fit_method)
        np.save(os.path.join(simparams.data_dir, filename), y_samples)

    filename = 'mquantile_DoE_McsE6_PCE{:d}_{:s}.npy'.format(poly_order, fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(mquantiles))

    # filename = 'mse_DoE_QuadLeg{:d}_PCE_{:s}.npy'.format(poly_order, fit_method)


if __name__ == '__main__':
    main()
