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
            filename = 'DoE_QuadLeg{:d}.npy'.format(iquad_order)
            data_set = np.load(os.path.join(data_dir, filename))
            u.append(data_set[0:ndim,:] )
            x.append(data_set[ndim:2*ndim,:])
            y.append(data_set[-1,:])

    filename = 'DoE_Lhs{:d}.npy'.format(n_lhs)
    data_set = np.load(os.path.join(data_dir, filename))
    u.append(data_set[0:ndim,:] )
    x.append(data_set[ndim:2*ndim,:])
    y.append(data_set[-1,:])

    u = np.concatenate(u, axis=1)
    x = np.concatenate(x, axis=1)
    y = np.concatenate(y)
    return u, x, y

def main():
    ### ------------------------ Parameters set-up ----------------------- ###
    ndim        = 3
    dist_x      = cp.Iid(cp.Uniform(-np.pi, np.pi),ndim) 
    dist_zeta   = cp.Iid(cp.Uniform(-1, 1),ndim) 
    simparams   = museuq.simParameters('Ishigami', dist_zeta)
    solver      = museuq.Ishigami()

    ### ------------------------ Adaptive parameters ------------------------ 
    plim        = [2,2]
    n_budget    = 1000
    n_quad      = (plim[0]+1)**ndim 
    n_lhs       = n_budget - n_quad
    while (n_quad + (plim[1] +1+1)**ndim) < n_budget:
        plim[1]+= 1
        n_quad += (plim[1] + 1)**ndim
        n_lhs   = n_budget - n_quad

    simparams.set_adaptive_parameters(n_budget=n_budget, plim=plim, r2_bound=0.9, q_bound=0.05)
    simparams.info()

    ### ----------------------- Initialization  -------------------- 
    fit_method      = 'GLK'
    poly_order      = plim[0]
    n_eval_curr     = 0
    n_eval_next     = 0
    mquantiles      = []
    r2_score_adj    = []
    cv_error        = []
    f_hat = None

    ### ----------------------- Get validation data for GLK-------------------- 
    

    ### ----------------------- Adaptive step starts-------------------- 
    while simparams.is_adaptive_continue(n_eval_next, poly_order=poly_order,
            r2_adj=r2_score_adj, mquantiles=mquantiles, cv_error=cv_error):
        print('   * Adaptive simulation continue...')
        ### ============ Get training points ============
        quad_order  = poly_order + 1
        filename    = 'DoE_QuadLeg{:d}.npy'.format(quad_order)
        data_set    = np.load(os.path.join(simparams.data_dir, filename))
        u_train     = data_set[0:ndim,:] 
        x_train     = data_set[ndim:2*ndim,:] 
        w_train     = data_set[-2,:]
        y_train     = data_set[-1,:]

        # print('  > {:<10s}: {:s}'.format('filename', filename))
        ### ============ Build Surrogate Model ============
        pce_model   = museuq.PCE(poly_order, dist_zeta)
        pce_model.fit(u_train, y_train, w=w_train, method=fit_method)
        y_train_hat = pce_model.predict(u_train)

        u_valid, x_valid, y_valid = get_validation_data(quad_order, plim, n_lhs, ndim, data_dir=simparams.data_dir)
        y_valid_hat = pce_model.predict(u_valid)

        filename = 'DoE_McsE6R0.npy'
        data_set = np.load(os.path.join(simparams.data_dir, filename))
        u_samples= data_set[0:ndim,:]
        x_samples= data_set[ndim: 2*ndim,:]
        y_samples= pce_model.predict(u_samples)

        ### ============ updating parameters ============
        cv_error.append(uq_metrics.mean_squared_error(y_valid, y_valid_hat))
        r2_score_adj.append(uq_metrics.r2_score_adj(y_train, y_train_hat, len(pce_model.active_)))
        mquantiles.append(uq_metrics.mquantiles(y_samples, 1-1e-4))
        poly_order  += 1
        n_eval_curr += u_train.shape[1] 
        n_eval_next = n_eval_curr + (poly_order+1)**ndim
        f_hat = pce_model

    poly_order -= 1
    print('------------------------------------------------------------')
    print(' > Adaptive simulation done:')
    print('------------------------------------------------------------')
    print(' - {:<25s} : {}'.format('Polynomial order (p)', poly_order))
    print(' - {:<25s} : {}'.format('Active basis', f_hat.active_))
    print(' - {:<25s} : {}'.format('# Evaluations ', n_eval_curr))
    print(' - {:<25s} : {}'.format('R2_adjusted ', np.around(r2_score_adj, 2)))
    print(' - {:<25s} : {}'.format('mquantiles', np.around(np.squeeze(np.array(mquantiles)), 2)))


    filename = 'mquantile_DoE_QuadLeg_PCE{:d}_{:s}_path.npy'.format(poly_order, fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(mquantiles))

    filename = 'r2_DoE_QuadLeg_PCE{:d}_{:s}_path.npy'.format(poly_order, fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(r2_score_adj))

    filename = 'cv_error_DoE_LhsE3_PCE{:d}_{:s}_path.npy'.format(poly_order, fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(cv_error))

    ## run MCS to get mquantile
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

    filename = 'mquantile_DoE_QuadLeg_PCE{:d}_{:s}.npy'.format(poly_order, fit_method)
    np.save(os.path.join(simparams.data_dir, filename), np.array(mquantiles))



if __name__ == '__main__':
    main()
