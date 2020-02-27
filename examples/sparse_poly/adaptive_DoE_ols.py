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
import numpy as np, os, sys, math
import scipy.stats as stats
from tqdm import tqdm
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()



def main():
    """
    Fajraoui, Noura, Stefano Marelli, and Bruno Sudret. "Sequential design of experiment for sparse polynomial chaos expansions." SIAM/ASA Journal on Uncertainty Quantification 5.1 (2017): 1061-1085.
    """
    ## ------------------------ Parameters set-up ----------------------- ###
    np.random.seed(100)
    ndim, deg   = 2, 20
    # dist_x      = [stats.uniform(-np.pi, np.pi),] * ndim
    dist_zeta   = [stats.uniform(-1, 1),] * ndim
    simparams   = museuq.simParameters('Sparse_poly', dist_zeta)
    orth_poly   = museuq.Legendre(d=ndim, deg=deg)
    solver      = museuq.sparse_poly(orth_poly)
    print(solver.poly.basis_degree)
    # print(abs(solver.poly.coef)/ max(abs(solver.poly.coef)))
    print(np.argsort(abs(solver.poly.coef)))


    ### ============ Adaptive parameters ============
    n_eval      = 1000
    n_new       = 50
    n_budget    = 5000
    poly_order  = 25
    simparams.set_adaptive_parameters(n_budget=n_budget,plim=(2,100), r2_bound=0.9, q_bound=0.01)
    simparams.info()

    ### ============ Stopping Criteria ============
    orth_poly       = museuq.Legendre(d=ndim,deg=poly_order)
    k_sparsity      = orth_poly.num_basis 
    cv_error        = []
    mquantiles      = []
    r2_score_adj    = []
    pce_model       = museuq.PCE(dist_zeta, poly_order)
    # n_eval          = max(n_eval, 2 * k_sparsity) ## for ols, oversampling rate at least 2
    # print(pce_model.orth_poly.basis_degree)

    ### ============ Get design points ============
    mcs_data_set= np.load('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/MCS/Uniform/DoE_McsE6R0.npy')
    oed_d_set   = np.load('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/OED/Uniform/DoE_McsE5R0_d{:d}_p{:d}_D.npy'.format(ndim, poly_order))
    curr_idx= oed_d_set[:n_eval].tolist()
    u_train = mcs_data_set[:ndim, curr_idx]
    x_train = solver.map_domain(u_train, dist_zeta) 
    y_train = solver.run(x_train)
    n_eval_path = [n_eval,]
    # print(curr_idx)
    while simparams.is_adaptive_continue(n_eval, poly_order=poly_order,
            r2_adj=r2_score_adj, mquantiles=mquantiles, cv_error=cv_error):
        print(' > Adaptive simulation continue...')
        ### ============ Get training points ============
        ### ============ Build Surrogate Model ============
        pce_model.fit_olslars(u_train, y_train, n_splits=10)
        y_train_hat = pce_model.predict(u_train)

        ### ============ calculating & updating metrics ============
        u_test= mcs_data_set[0:ndim,:]
        y_test= pce_model.predict(u_test)

        cv_error.append(pce_model.cv_error)
        r2_score_adj.append(museuq.metrics.r2_score_adj(y_train, y_train_hat, len(pce_model.active_)))
        mquantiles.append(museuq.metrics.mquantiles(y_test, 1-1e-4))
        n_eval_path.append(n_eval)

        ### ============ Getting new samples ============
        vander = pce_model.orth_poly.vandermonde(mcs_data_set[0:ndim, :100000])  
        vander = vander[:, pce_model.active_]
        new_samples = museuq.OptimalDesign('D',curr_set=curr_idx).samples(vander,n_samples=n_new,orth_basis=True)
        u_train_ = mcs_data_set[:ndim, new_samples]
        x_train_ = solver.map_domain(u_train_, dist_zeta) 
        y_train_ = solver.run(x_train_)
        u_train = np.hstack((u_train, u_train_))
        x_train = np.hstack((x_train, x_train_))
        y_train = np.hstack((y_train, y_train_))
        curr_idx += new_samples
        print(curr_idx)
        print(new_samples)
        ### ============ updating parameters ============
        n_eval     += n_new
        # k_sparsity  = len(cp.orth_ttr(poly_order, dist_zeta))
        # n_eval      = max(n_eval, 2 * k_sparsity) ## for ols, oversampling rate at least 2
        # pce_model       = pce_model

    print(pce_model.model.basis_degree)
    # print(pce_model.basis[pce_model.active_])
    # poly_order -= 1
    # print('------------------------------------------------------------')
    # print('>>> Adaptive simulation done:')
    # print('------------------------------------------------------------')
    # print(' - {:<25s} : {}'.format('Polynomial order (p)', poly_order))
    # print(' - {:<25s} : {} -> #{:d}'.format('Active basis', pce_model.active_, len(pce_model.active_)))
    # print(' - {:<25s} : {}'.format('# Evaluations ', n_eval))
    # print(' - {:<25s} : {}'.format('R2_adjusted ', np.around(r2_score_adj, 2)))
    # print(' - {:<25s} : {}'.format('mquantiles', np.around(np.squeeze(np.array(mquantiles)), 2)))

    # filename = 'mquantile_DoE_q15_OptD2040_PCE{:d}_{:s}_path.npy'.format(poly_order, fit_method)
    # np.save(os.path.join(simparams.data_dir, filename), np.array(mquantiles))

    # filename = 'r2_DoE_q15_OptD2040_PCE{:d}_{:s}_path.npy'.format(poly_order, fit_method)
    # np.save(os.path.join(simparams.data_dir, filename), np.array(r2_score_adj))

    # filename = 'cv_error_DoE_q15_OptD2040_PCE{:d}_{:s}_path.npy'.format(poly_order, fit_method)
    # np.save(os.path.join(simparams.data_dir, filename), np.array(cv_error))

    # filename = 'n_eval_DoE_q15_OptD2040_PCE{:d}_{:s}_path.npy'.format(poly_order, fit_method)
    # np.save(os.path.join(simparams.data_dir, filename), np.array(n_eval_path))

    # mquantiles = []
    # for r in tqdm(range(10), ascii=True, desc="   - " ):
        # filename = 'DoE_McsE6R{:d}.npy'.format(r)
        # mcs_data_set = np.load(os.path.join(simparams.data_dir, filename))
        # u_test= mcs_data_set[0:ndim,:]
        # x_samples= mcs_data_set[ndim: 2*ndim,:]
        # y_test= pce_model.predict(u_test)
        # mquantiles.append(museuq.metrics.mquantiles(y_test, [1-1e-4, 1-1e-5, 1-1e-6]))
        # filename = 'DoE_McsE6R{:d}_q15_OptD2040_PCE{:d}_{:s}.npy'.format(r, poly_order, fit_method)
        # np.save(os.path.join(simparams.data_dir, filename), y_test)

    # filename = 'mquantile_DoE_q15_OptD2040_PCE{:d}_{:s}.npy'.format(poly_order, fit_method)
    # np.save(os.path.join(simparams.data_dir, filename), np.array(mquantiles))


if __name__ == '__main__':
    main()
