#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import museuq
import numpy as np, os, sys
import scipy.stats as stats
import warnings
from tqdm import tqdm
from museuq.environment import Kvitebjorn
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()

def cal_poly_limit(n_budget, ndim, start=2):
    """

    """
    plim   = [start, start]
    n_quad = (plim[0]+1)**ndim 
    n_left = n_budget - n_quad
    while (n_quad + (plim[1]+1+1)**ndim) < n_budget:
        plim[1]+= 1
        n_quad += (plim[1] + 1)**ndim
        n_left  = n_budget - n_quad
    return plim, n_left

def get_validation_data(solver, quad_order, plim, n_lhs, data_dir=os.getcwd):
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
            ### quad_order is used to train
            pass
        else:
            filename = 'DoE_QuadLeg{:d}.npy'.format(iquad_order)
            try:
                data_set = np.load(os.path.join(data_dir, filename))
                u.append(data_set[0:ndim,:] )
                x.append(data_set[ndim:2*ndim,:])
                try:
                    y.append(data_set[2*ndim+2,:])
                except IndexError:
                    y.append(solver.run(data_set[ndim:2*ndim,:]))
            except FileNotFoundError:
                pass



    filename = 'DoE_Lhs{:d}.npy'.format(n_lhs)
    data_set = np.load(os.path.join(data_dir, filename))
    u.append(data_set[0:ndim,:] )
    x.append(data_set[ndim:2*ndim,:])
    y.append(data_set[-1,:])

    u = np.concatenate(u, axis=1)
    x = np.concatenate(x, axis=1)
    y = np.concatenate(y, axis=1)
    return u, x, y



def main():
    ### ------------------------ Parameters set-up ----------------------- ###
    ndim, deg   = 2, 20
    # dist_x      = [stats.uniform(-np.pi, np.pi),] * ndim
    dist_zeta   = [stats.uniform(-1, 1),] * ndim
    simparams   = museuq.simParameters('Sparse_poly', dist_zeta)
    orth_poly   = museuq.Legendre(d=ndim, deg=deg)
    solver      = museuq.sparse_poly(orth_poly)

    ### ------------------------ Adaptive parameters ------------------------ 
    n_budget    = 5000
    plim,n_lhs  = cal_poly_limit(n_budget, ndim) 
    simparams.set_adaptive_parameters(n_budget=n_budget, plim=plim, r2_bound=0.9, q_bound=0.05)
    simparams.info()
    print('     - {:<23s}: {:d}'.format('LHS samples', n_lhs))

    ### ------------------------ Getting trainning data ---------------------- 
    quad_u = []
    quad_w = []
    quad_y = []
    for p in range(plim[0], plim[1]+1):
        orth_poly = museuq.Legendre(d=ndim, deg=p)
        idoe_u, idoe_w = museuq.QuadratureDesign(orth_poly).samples(p+1)
        quad_u.append(idoe_u)
        quad_w.append(idoe_w)
        quad_y.append(solver.run(idoe_u))
    _, lhs_u = museuq.LHS(dist_zeta).samples(n_lhs)
    lhs_y = solver.run(lhs_u)

    ### ----------------------- Initialization  -------------------- 
    i = 0
    poly_order  = plim[0]
    n_eval_curr = 0
    n_eval_next = 0
    mquantiles  = []
    r2_score_adj= []
    cv_error    = []

    ### ----------------------- Adaptive step starts-------------------- 
    while simparams.is_adaptive_continue(n_eval_next, poly_order=poly_order,
            r2_adj=r2_score_adj, mquantiles=mquantiles, cv_error=cv_error):
        print(' > Adaptive simulation continue...')
        ### ============ Get training points ============

        # quad_order  = poly_order + 1
        # filename    = 'DoE_QuadLeg{:d}.npy'.format(quad_order)
        # data_set    = np.load(os.path.join(simparams.data_dir, filename))
        # u_train     = data_set[0:ndim,:] 
        # x_train     = data_set[ndim:2*ndim,:] 
        # w_train     = data_set[-2,:]
        # try:
            # y_train = data_set[2*ndim+1,:]
        # except IndexError:
            # y_train = solver.run(*x_train)
            # print(x_train.shape)
            # print(y_train.shape)
            # print(u_train.shape)
            # print(w_train.shape)
            # data = np.concatenate((u_train, x_train, w_train.reshape(1,-1), y_train.reshape(1,-1)), axis=1)
            # np.save(os.path.join(simparams.data_dir, filename), data)

        ## >>> training data 
        u_train = quad_u[i]
        w_train = quad_w[i]
        y_train = quad_y[i]
        ## >>> validation data 
        u_valid = lhs_u
        y_valid = lhs_y
        for j in range(len(quad_u)):
            if j == i:
                pass
            else:
                u_valid = np.hstack((u_valid, quad_u[j]))
                y_valid = np.hstack((y_valid, quad_y[j]))


        ### ============ Build Surrogate Model ============
        pce_model   = museuq.PCE(dist_zeta, poly_order)
        pce_model.fit_quadrature(u_train, w_train, y_train)
        y_train_hat = pce_model.predict(u_train)
        # u_valid,x_valid,y_valid=get_validation_data(quad_order, plim, n_lhs, ndim, data_dir=simparams.data_dir)
        y_valid_hat = pce_model.predict(u_valid)

        data_set = np.load('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/MCS/Uniform/DoE_McsE6R0.npy')
        u_test = data_set[0:ndim,:]
        # x_test = data_set[ndim: 2*ndim,:]
        y_test = pce_model.predict(u_test)

        # ### ============ updating parameters ============
        icv_error = museuq.metrics.mean_squared_error(y_valid, y_valid_hat)
        print(icv_error)
        cv_error.append(icv_error)
        ir2_score_adj = museuq.metrics.r2_score_adj(y_train, y_train_hat, len(pce_model.active_))
        print(ir2_score_adj)
        r2_score_adj.append(ir2_score_adj)
        mquantiles.append(museuq.metrics.mquantiles(y_test, 1-1e-4))
        poly_order  += 1
        i += 1
        n_eval_curr += u_train.shape[1] 
        n_eval_next = n_eval_curr + (poly_order+1)**ndim
        # f_hat = pce_model

    # poly_order -= 1
    # print('------------------------------------------------------------')
    # print(' > Adaptive simulation done:')
    # print('------------------------------------------------------------')
    # print(' - {:<25s} : {}'.format('Polynomial order (p)', poly_order))
    # print(' - {:<25s} : {}'.format('Active basis', f_hat.active_))
    # print(' - {:<25s} : {}'.format('# Evaluations ', n_eval_curr))
    # print(' - {:<25s} : {}'.format('R2_adjusted ', np.around(r2_score_adj, 2)))
    # print(' - {:<25s} : {}'.format('mquantiles', np.around(np.squeeze(np.array(mquantiles)), 2)))


    # filename = 'mquantile_DoE_QuadLeg_PCE{:d}_{:s}_path.npy'.format(poly_order, fit_method)
    # np.save(os.path.join(simparams.data_dir, filename), np.array(mquantiles))

    # filename = 'r2_DoE_QuadLeg_PCE{:d}_{:s}_path.npy'.format(poly_order, fit_method)
    # np.save(os.path.join(simparams.data_dir, filename), np.array(r2_score_adj))

    # filename = 'cv_error_DoE_LhsE3_PCE{:d}_{:s}_path.npy'.format(poly_order, fit_method)
    # np.save(os.path.join(simparams.data_dir, filename), np.array(cv_error))

    # ## run MCS to get mquantile
    # mquantiles = []
    # for r in tqdm(range(10), ascii=True, desc="   - " ):
        # filename = 'DoE_McsE6R{:d}.npy'.format(r)
        # data_set = np.load(os.path.join(simparams.data_dir, filename))
        # u_test= data_set[0:ndim,:]
        # x_test= data_set[ndim: 2*ndim,:]
        # y_test= f_hat.predict(u_test)
        # mquantiles.append(museuq.metrics.mquantiles(y_test, [1-1e-4, 1-1e-5, 1-1e-6]))
        # filename = 'DoE_McsE6R{:d}_PCE{:d}_{:s}.npy'.format(r, poly_order, fit_method)
        # np.save(os.path.join(simparams.data_dir, filename), y_test)

    # filename = 'mquantile_DoE_QuadLeg_PCE{:d}_{:s}.npy'.format(poly_order, fit_method)
    # np.save(os.path.join(simparams.data_dir, filename), np.array(mquantiles))

if __name__ == '__main__':
    main()
