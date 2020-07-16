#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import numpy as np, chaospy as cp, os, sys
import uqra, warnings
from tqdm import tqdm
from uqra.environment import Kvitebjorn
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()

def get_validation_data(quad_order, plim, n_lhs, ndim, data_dir=os.getcwd):
    """
    For GLK method, number of evaluations for one polynomial order p is predefined (p+1)**ndim
    Thus, plim[1] is constrained by n_budget.
    How to do cross validation in GLK? 
    One can use all the data evaluated except the ones used to fit model 
    Evaluated data set are chosen for poly_order in range plim. If there are left-over resource, those are sampled with LHS
    """
    u = [] 
    x = [] 
    y = []
    for ipoly_order in range(plim[0], plim[1]+1):
        iquad_order = ipoly_order+1
        if iquad_order == quad_order:
            pass
        else:
            filename = 'DoE_QuadHem{:d}.npy'.format(iquad_order)
            data_set = np.load(os.path.join(data_dir, filename))
            u.append(data_set[0:ndim,:] )
            x.append(data_set[ndim:2*ndim,:])
            filename = 'DoE_QuadHem{:d}_y.npy'.format(iquad_order)
            data_set = np.load(os.path.join(data_dir, filename))
            y.append(data_set)
            # y.append(data_set[-1,:])

    filename = 'DoE_Lhs{:d}.npy'.format(n_lhs)
    data_set = np.load(os.path.join(data_dir, filename))
    u.append(data_set[0:ndim,:] )
    x.append(data_set[ndim:2*ndim,:])
    filename = 'DoE_Lhs{:d}_y.npy'.format(n_lhs)
    data_set = np.load(os.path.join(data_dir, filename))
    y.append(data_set)

    u = np.concatenate(u, axis=1)
    x = np.concatenate(x, axis=1)
    y = np.concatenate(y, axis=1)
    return u, x, y

def main():
    ## ------------------------ Parameters set-up ----------------------- ###
    ndim        = 2
    nqoi        = 1 ## 1: eta, 2. y
    dist_x      = cp.Iid(cp.Normal(),ndim) 
    dist_zeta   = cp.Iid(cp.Normal(),ndim) 
    simparams   = uqra.simParameters('linear_oscillator', dist_zeta)
    solver      = uqra.linear_oscillator()
    n_sim_short = 10 ## number of short-term simulations

    ### ============ Adaptive parameters ============
    plim        = [3,3]
    n_budget    = 1000
    n_quad      = (plim[0]+1)**ndim 
    n_lhs       = n_budget - n_quad
    while (n_quad + (plim[1] +1+1)**ndim) < n_budget:
        plim[1]+= 1
        n_quad += (plim[1] + 1)**ndim
        n_lhs   = n_budget - n_quad

    simparams.set_adaptive_parameters(n_budget=n_budget, plim=plim, r2_bound=0.9, q_bound=0.05)
    simparams.info()
    print('     - {:<23s}: {:d}'.format('LHS samples', n_lhs))

    #### ----------------------- Build PCE Surrogate Model -------------------- ###
    ### ============ Initialization  ============
    fit_method  = 'GLK'
    poly_order  = plim[0]
    n_eval_curr = 0
    n_eval_next = 0
    mquantiles  = []
    r2_score_adj= []
    cv_error    = []
    f_hat       = uqra.mPCE() 

    while simparams.is_adaptive_continue(n_eval_next, poly_order=poly_order,
            r2_adj=r2_score_adj, mquantiles=mquantiles, cv_error=cv_error):
        print(' > Adaptive simulation continue...')
        ### ============ Get validation points ============
        quad_order      = poly_order + 1
        print('   - Retrieving validation data ...')
        u_valid, x_valid, y_valid = get_validation_data(quad_order,plim, n_lhs, ndim, data_dir=simparams.data_dir)
        print(y_valid.shape)
        ### ============ Get training points ============
        filename    = 'DoE_QuadHem{:d}.npy'.format(quad_order)
        data_set    = np.load(os.path.join(simparams.data_dir, filename))
        u_train     = data_set[0:ndim,:] 
        x_train     = data_set[ndim:2*ndim,:] 
        w_train     = data_set[-2,:]
        filename    = 'DoE_QuadHem{:d}.npy'.format(quad_order)
        data_set    = np.load(os.path.join(simparams.data_dir, filename))
        y_train     = np.squeeze(data_set[:,:,4, nqoi]).T

        ### ============ Build Surrogate Model ============
        # Two PCE models, one for mean response and one for the difference
        y_train_mean = np.mean(y_trian, axis=1)
        y_train_diff = y_trian - y_train_mean

        pce_mean = uqra.PCE(poly_order, dist_zeta) 
        pce_mean.fit(u_train, y_train_mean, w=w_train, method=fit_method)

        y_train_hat = pce_mean.predict(u_train)
        y_valid_hat = pce_mean.predict(u_valid)






        pce_model   = uqra.mPCE(poly_order, dist_zeta)
        pce_model.fit(u_train, y_train, w=w_train, method=fit_method)
        y_train_hat = pce_model.predict(u_train)
        y_valid_hat = pce_model.predict(u_valid)

        filename = 'DoE_McsE6R0.npy'
        data_set = np.load(os.path.join(simparams.data_dir, filename))
        u_samples= data_set[0:ndim,:]
        x_samples= data_set[ndim: 2*ndim,:]
        y_samples= pce_model.predict(u_samples)

        ### ============ updating parameters ============
        pce_model.cv_error = uqra.metrics.mean_squared_error(y_valid, y_valid_hat)
        cv_error.append(pce_model.cv_error)
        r2_score_adj.append(uqra.metrics.r2_score_adj(y_train, y_train_hat, len(pce_model.active_)))
        mquantiles.append(uqra.metrics.mquantiles(y_samples, 1-1e-4))
        poly_order  += 1
        n_eval_curr += u_train.shape[1] 
        n_eval_next = n_eval_curr + (poly_order+1)**ndim
        f_hat = pce_model if f_hat.cv_error > pce_model.cv_error else f_hat

    print('------------------------------------------------------------')
    print('>>> Adaptive simulation done:')
    print('------------------------------------------------------------')
    print(' - {:<25s} : {}'.format('Polynomial order (p)', f_hat.poly_order))
    print(' - {:<25s} : {}'.format('Active basis', f_hat.active_))
    print(' - {:<25s} : {}'.format('# Evaluations ', n_eval_curr))
    print(' - {:<25s} : {}'.format('R2_adjusted ', np.around(r2_score_adj, 2)))
    print(' - {:<25s} : {}'.format('mquantiles', np.around(np.squeeze(np.array(mquantiles)), 2)))


    filename = 'mquantile_DoE_QuadHem_PCE{:d}_{:s}_path.npy'.format(f_hat.poly_order, fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(mquantiles))

    filename = 'r2_DoE_QuadHem_PCE{:d}_{:s}_path.npy'.format(f_hat.poly_order, fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(r2_score_adj))

    filename = 'cv_error_DoE_QuadHem_PCE{:d}_{:s}_path.npy'.format(f_hat.poly_order, fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(cv_error))

    ## run MCS to get mquantile
    mquantiles = []
    for r in tqdm(range(10), ascii=True, desc="   - " ):
        filename = 'DoE_McsE6R{:d}.npy'.format(r)
        data_set = np.load(os.path.join(simparams.data_dir, filename))
        u_samples= data_set[0:ndim,:]
        x_samples= data_set[ndim: 2*ndim,:]
        y_samples= f_hat.predict(u_samples)
        mquantiles.append(uqra.metrics.mquantiles(y_samples, [1-1e-4, 1-1e-5, 1-1e-6]))
        filename = 'DoE_McsE6R{:d}_QuadHem_PCE{:d}_{:s}.npy'.format(r, f_hat.poly_order, fit_method)
        np.save(os.path.join(simparams.data_dir, filename), y_samples)

    filename = 'mquantile_DoE_QuadHem_PCE{:d}_{:s}.npy'.format(f_hat.poly_order, fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(mquantiles))

if __name__ == '__main__':
    main()
