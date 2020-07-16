#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import uqra
import numpy as np, chaospy as cp, os, sys
import warnings
from tqdm import tqdm
from uqra.utilities import helpers as uqhelpers
from uqra.utilities import metrics_collections as uq_metrics
from sklearn.model_selection import KFold
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()


def main():
    ## ------------------------ Parameters set-up ----------------------- ###
    ndim        = 3
    dist_x      = cp.Iid(cp.Uniform(-np.pi, np.pi),ndim) 
    dist_zeta   = cp.Iid(cp.Uniform(-1, 1),ndim) 
    simparams   = uqra.simParameters('Ishigami', dist_zeta)
    solver      = uqra.Ishigami()
    fit_method  = 'OLSLARS'

    ### ============ Adaptive parameters ============
    plim        = (2,15)
    n_budget    = 1000
    n_newsamples= 10
    simparams.set_adaptive_parameters(n_budget=n_budget, plim=plim, r2_bound=0.9, q_bound=0.05)
    # simparams.set_error()  # no observation error for sdof
    simparams.info()


    ### ============ Initialization  ============
    # P0 = len(cp.orth_ttr(plim[0], dist_zeta))
    # alpha = 0.8
    # n_samples_done  = min(2*P0, alpha*n_budget)
    # n_samples_done  = max(P0, n_samples_done)
    n_eval          = 50
    poly_order      = plim[1]
    r2_score_adj    = []
    cv_error        = []
    mquantiles      = []

    ### 1. Initial design with OPT-D

    while simparams.is_adaptive_continue(n_eval, poly_order=poly_order,
            r2_adj=r2_score_adj, mquantiles=mquantiles, cv_error=cv_error):
        print('==> Adaptive simulation continue...')

        data_dir= simparams.data_dir
        filename= 'DoE_McsE6R0.npy'
        data_set= np.load(os.path.join(data_dir, filename))
        u_data  = data_set[0:ndim, :]
        x_data  = data_set[ndim:2*ndim, :]
        y_data  = data_set[-1, :].reshape(1,-1)
        print('Candidate samples filename: {:s}'.format(filename))
        print('   >> Candidate sample set shape: {}'.format(u_data.shape))
        basis   = cp.orth_ttr(plim[1], dist_zeta)
        design_matrix = basis(*u_data).T
        print('   >> Candidate Design matrix shape: {}'.format(design_matrix.shape))
        
        doe = uqra.OptimalDesign('D', n_samples=n_eval)
        doe_index = doe.adaptive(design_matrix, n_samples = n_eval, is_orth=True)
        print(len(doe_index))
        u_doe = u_data[:,doe_index]
        x_doe = x_data[:,doe_index]
        y_doe = solver.run(x_doe)

        # data = np.concatenate((doe.I.reshape(1,-1),doe.u,doe.x, solver.y.reshape(1,-1)), axis=0)
        # filename = os.path.join(data_dir, 'DoE_McsE6R{:d}_q{:d}_OptD{:d}'.format(r,plim[1], n_eval))
        # np.save(filename, data)

        ### ============ Build Surrogate Model ============
        u_train   = np.concatenate((u_train, u_doe), axis=0)
        y_train   = np.concatenate((y_train, y_doe), axis=0)
        pce_model = uqra.PCE(plim[1], dist_zeta)
        pce_model.fit(u_train, y_train, method=fit_method)
        y_pred    = pce_model.predict(u_train)

        filename = 'DoE_McsE6R0.npy'
        data_set = np.load(os.path.join(simparams.data_dir, filename))
        u_samples= data_set[0:ndim,:]
        x_samples= data_set[ndim: 2*ndim,:]
        y_samples= pce_model.predict(u_samples)

        ### ============ updating parameters ============
        # error_mse.append(uq_metrics.mean_squared_error(y_valid, y_pred))
        cv_error.append(pce_model.cv_error)
        r2_score_adj.append(uq_metrics.r2_score_adj(y_train, y_pred, len(pce_model.active_)))
        mquantiles.append(uq_metrics.mquantiles(y_samples, 1-1e-4))
        n_eval += u_doe.shape[1]
        f_hat  = pce_model


    poly_order -= 1
    print('------------------------------------------------------------')
    print('>>> Adaptive simulation done:')
    print('------------------------------------------------------------')
    print(' - {:<25s} : {}'.format('Polynomial order (p)', poly_order))
    print(' - {:<25s} : {}'.format('Active basis', f_hat.active_))
    print(' - {:<25s} : {}'.format('# Evaluations ', n_eval))
    print(' - {:<25s} : {}'.format('R2_adjusted ', np.around(r2_score_adj, 2)))
    print(' - {:<25s} : {}'.format('mquantiles', np.around(np.squeeze(np.array(mquantiles)), 2)))

    # filename = 'mquantile_DoE_QuadLeg{:d}_PCE_{:s}.npy'.format(iquad_orders, fit_method)
    # np.save(os.path.join(simparams.data_dir, filename), np.array(mquantiles))

    # filename = 'r2_DoE_QuadLeg{:d}_PCE_{:s}.npy'.format(iquad_orders, fit_method)
    # np.save(os.path.join(simparams.data_dir, filename), np.array(r2_adj))



if __name__ == '__main__':
    main()
