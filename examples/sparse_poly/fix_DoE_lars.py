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
    dist_x      = [stats.uniform(-np.pi, np.pi),] * ndim
    dist_zeta   = [stats.uniform(-1, 1),] * ndim
    simparams   = museuq.simParameters('Ishigami', dist_zeta)
    solver      = museuq.Ishigami()

    ### ============ Adaptive parameters ============
    n_budget    = 1000
    plim        = (2,20)
    simparams.set_adaptive_parameters(n_budget=n_budget, plim=plim, r2_bound=0.9, q_bound=0.05)
    simparams.info()

    ### ============ Stopping Criteria ============
    fit_method  = 'LASSOLARS'
    poly_order  = plim[0]
    cv_error    = []
    mquantiles  = []
    r2_score_adj= []
    f_hat       = None

    ### ============ Get training points ============
    filename= 'DoE_LhsE3R0.npy'
    data_set= np.load(os.path.join(simparams.data_dir, filename))
    u_train = data_set[0:ndim,:]
    x_train = data_set[ndim:2*ndim,:]
    y_train = data_set[-1,:]
    n_eval  = u_train.shape[1]


    while simparams.is_adaptive_continue(n_eval, poly_order=poly_order,
            r2_adj=r2_score_adj, mquantiles=mquantiles, cv_error=cv_error):
        print(' > Adaptive simulation continue...')
        ### ============ Build Surrogate Model ============
        pce_model   = museuq.PCE(poly_order, dist_zeta)
        pce_model.fit(u_train, y_train, method=fit_method)
        y_train_hat = pce_model.predict(u_train)

        filename = 'DoE_McsE6R0.npy'
        data_set = np.load(os.path.join(simparams.data_dir, filename))
        u_samples= data_set[0:ndim,:]
        x_samples= data_set[ndim: 2*ndim,:]
        y_samples= pce_model.predict(u_samples)
        ### ============ updating parameters ============

        cv_error.append(pce_model.cv_error)
        r2_score_adj.append(museuq.metrics.r2_score_adj(y_train, y_train_hat, len(pce_model.active_)))
        mquantiles.append(museuq.metrics.mquantiles(y_samples, 1-1e-4))
        poly_order += 1
        if f_hat is None:
            f_hat  = pce_model
        else:
            if f_hat.cv_error > pce_model.cv_error:
                f_hat = pce_model


    poly_order -= 1
    print('------------------------------------------------------------')
    print('>>> Adaptive simulation done:')
    print('------------------------------------------------------------')
    print(' - {:<25s} : {}'.format('Polynomial order (p)', poly_order))
    print(' - {:<25s} : {} ->#{:d}'.format('Active basis', f_hat.active_, len(f_hat.active_)))
    print(' - {:<25s} : {}'.format('# Evaluations ', n_eval))
    print(' - {:<25s} : {}'.format('R2_adjusted ', np.around(r2_score_adj, 2)))
    print(' - {:<25s} : {}'.format('mquantiles', np.around(np.squeeze(np.array(mquantiles)), 2)))

    filename = 'mquantile_DoE_LhsE3_PCE{:d}_{:s}_path.npy'.format(poly_order, fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(mquantiles))

    filename = 'r2_DoE_LhsE3_PCE{:d}_{:s}_path.npy'.format(poly_order, fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(r2_score_adj))

    filename = 'cv_error_DoE_LhsE3_PCE{:d}_{:s}_path.npy'.format(poly_order, fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(cv_error))


    mquantiles = []
    for r in tqdm(range(10), ascii=True, desc="   - " ):
        filename = 'DoE_McsE6R{:d}.npy'.format(r)
        data_set = np.load(os.path.join(simparams.data_dir, filename))
        u_samples= data_set[0:ndim,:]
        x_samples= data_set[ndim: 2*ndim,:]
        y_samples= f_hat.predict(u_samples)
        mquantiles.append(museuq.metrics.mquantiles(y_samples, [1-1e-4, 1-1e-5, 1-1e-6]))
        filename = 'DoE_McsE6R{:d}_PCE{:d}_{:s}.npy'.format(r, poly_order, fit_method)
        np.save(os.path.join(simparams.data_dir, filename), y_samples)

    filename = 'mquantile_DoE_LhsE3_PCE{:d}_{:s}.npy'.format(poly_order, fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(mquantiles))

    # filename = 'mse_DoE_QuadLeg{:d}_PCE_{:s}.npy'.format(poly_order, fit_method)


if __name__ == '__main__':
    main()
