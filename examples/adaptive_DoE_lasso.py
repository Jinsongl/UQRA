#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import museuq, warnings
import numpy as np, chaospy as cp, os, sys
from tqdm import tqdm
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()


def main():
    ## ------------------------ Parameters set-up ----------------------- ###
    ndim        = 3
    dist_x      = cp.Iid(cp.Uniform(-np.pi, np.pi),ndim) 
    dist_zeta   = cp.Iid(cp.Uniform(-1, 1),ndim) 
    simparams   = museuq.simParameters('Ishigami', dist_zeta)
    solver      = museuq.Ishigami()

    ### ============ Adaptive parameters ============
    plim        = (2,15)
    n_budget    = 2040 
    simparams.set_adaptive_parameters(n_budget=n_budget, plim=plim, r2_bound=0.9, q_bound=0.01)
    simparams.info()

    ### ============ Stopping Criteria ============
    fit_method      = 'LASSOLARS'
    poly_order      = plim[1]
    k_sparsity      = 75 # guess. K sparsity to meet RIP condition 
    cv_error        = []
    mquantiles      = []
    r2_score_adj    = []
    f_hat           = None

    ### ============ Get design points ============
    filename= 'DoE_McsE6R0_q15_OptD2040.npy'
    data_set= np.load(os.path.join(simparams.data_dir, filename))
    u_data = data_set[:ndim,:]
    x_data = data_set[ndim:2*ndim,:]
    y_data = data_set[-1,:]
    n_eval = 50
    n_eval = max(n_eval, 2 * k_sparsity) ## for ols, oversampling rate at least 2
    n_eval_path = [n_eval,]
    n_new  = 5
    ### 1. Initial design with OPT-D

    while simparams.is_adaptive_continue(n_eval, poly_order=poly_order,
            r2_adj=r2_score_adj, mquantiles=mquantiles, cv_error=cv_error):
        print(' > Adaptive simulation continue...')
        ### ============ Get training points ============
        u_train = u_data[:,:n_eval]
        y_train = y_data[:n_eval]
        ### ============ Build Surrogate Model ============
        pce_model   = museuq.PCE(poly_order, dist_zeta)
        pce_model.fit(u_train, y_train, method=fit_method)
        y_train_hat = pce_model.predict(u_train)

        ### ============ calculating & updating metrics ============
        filename = 'DoE_McsE6R0.npy'
        data_set = np.load(os.path.join(simparams.data_dir, filename))
        u_samples= data_set[0:ndim,:]
        x_samples= data_set[ndim: 2*ndim,:]
        y_samples= pce_model.predict(u_samples)

        cv_error.append(pce_model.cv_error)
        r2_score_adj.append(museuq.metrics.r2_score_adj(y_train, y_train_hat, len(pce_model.active_)))
        mquantiles.append(museuq.metrics.mquantiles(y_samples, 1-1e-4))
        n_eval_path.append(n_eval)
        ### ============ updating parameters ============
        n_eval     += n_new
        f_hat       = pce_model

    print('------------------------------------------------------------')
    print('>>> Adaptive simulation done:')
    print('------------------------------------------------------------')
    print(' - {:<25s} : {}'.format('Polynomial order (p)', poly_order))
    print(' - {:<25s} : {} -> #{:d}'.format('Active basis', f_hat.active_, len(f_hat.active_)))
    print(' - {:<25s} : {}'.format('# Evaluations ', n_eval))
    print(' - {:<25s} : {}'.format('R2_adjusted ', np.around(r2_score_adj, 2)))
    print(' - {:<25s} : {}'.format('mquantiles', np.around(np.squeeze(np.array(mquantiles)), 2)))

    filename = 'mquantile_DoE_q15_OptD2040_PCE{:d}_{:s}_path.npy'.format(poly_order, fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(mquantiles))

    filename = 'r2_DoE_q15_OptD2040_PCE{:d}_{:s}_path.npy'.format(poly_order, fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(r2_score_adj))

    filename = 'cv_error_DoE_q15_OptD2040_PCE{:d}_{:s}_path.npy'.format(poly_order, fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(cv_error))

    filename = 'n_eval_DoE_q15_OptD2040_PCE{:d}_{:s}_path.npy'.format(poly_order, fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(n_eval_path))

    mquantiles = []
    for r in tqdm(range(10), ascii=True, desc="   - " ):
        filename = 'DoE_McsE6R{:d}.npy'.format(r)
        data_set = np.load(os.path.join(simparams.data_dir, filename))
        u_samples= data_set[0:ndim,:]
        x_samples= data_set[ndim: 2*ndim,:]
        y_samples= f_hat.predict(u_samples)
        mquantiles.append(museuq.metrics.mquantiles(y_samples, [1-1e-4, 1-1e-5, 1-1e-6]))
        filename = 'DoE_McsE6R{:d}_q15_OptD2040_PCE{:d}_{:s}.npy'.format(r, poly_order, fit_method)
        np.save(os.path.join(simparams.data_dir, filename), y_samples)

    filename = 'mquantile_DoE_q15_OptD2040_PCE{:d}_{:s}.npy'.format(poly_order, fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(mquantiles))


if __name__ == '__main__':
    main()