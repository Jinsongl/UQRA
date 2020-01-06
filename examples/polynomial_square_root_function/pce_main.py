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
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()

def main():
    ## ------------------------ Parameters set-up ----------------------- ###
    ndim        = 2
    model_name  = 'polynomial_square_root_function'
    ## 1. Choose Wiener-Askey scheme random variable
    dist_x      = cp.Iid(cp.Normal(),ndim) 
    dist_zeta   = cp.Iid(cp.Normal(),ndim) 
    ## 2. If transformation needed, like Rosenblatt, need to be done here
    ## Perform Rosenblatt etc
    simparams = museuq.simParameters(model_name, dist_zeta)
    # simparams.set_error()  # no observation error for sdof
    simparams.info()

    #### ----------------------- Build PCE Surrogate Model -------------------- ###
    metamodel_class = 'PCE'
    quad_orders     = range(1,10)
    upper_tail_probs= [0.999, 0.9999,0.99999]
    moment2cal      = [1,2,3,4]
    metrics2cal     = [ 'explained_variance_score', 'mean_absolute_error', 'mean_squared_error',
                'median_absolute_error', 'r2_score', 'r2_score_adj', 'moment', 'mquantiles']

    # ### 1. PCE model based on quadrature design points, fitting with GLK and OLS

    pce_fit_method  = 'GLK'
    for iquad_orders in quad_orders:
        poly_order = iquad_orders - 1
        ### ============ Get training points ============
        filename = 'DoE_QuadHem{:d}.npy'.format(iquad_orders)
        data_set = np.load(os.path.join(simparams.data_dir, filename))
        print('  > {:<10s}: {:s}'.format('filename', filename))
        train_u  = data_set[0:2,:] 
        train_x  = data_set[2:4,:] 
        train_w  = data_set[4,:]
        train_y  = data_set[5,:]

        ### ============ Get Surrogate Model for each QoI============
        pce_model = museuq.PCE(poly_order, dist_zeta)
        # print(len(pce_model.basis[0]))
        pce_model.fit(train_u, train_y, w=train_w, fit_method=pce_fit_method)
        # pce_model.fit(train_u, train_y, fit_method=pce_fit_method)
        # print(pce_model.poly_coeffs)
        # print(pce_model.basis_coeffs)

        ### ============ Validating surrogate models at training points ============

        for r in range(10):
            filename = 'DoE_McsE6R{:d}.npy'.format(r)
            data_set = np.load(os.path.join(simparams.data_dir, filename))
            valid_u  = data_set[0:2,:]
            valid_x  = data_set[2:4,:]
            valid_y  = data_set[4  ,:]

            pce_model.predict(valid_u, valid_y, metrics=metrics2cal, prob=upper_tail_probs, moment=moment2cal)
            filename = 'DoE_QuadHem{:d}_{:s}_{:s}_E6R{:d}_y.npy'.format(iquad_orders, metamodel_class, pce_fit_method, r)
            np.save(os.path.join(simparams.data_dir, filename), pce_model.y_pred[0])

            pce_valid_y_ecdf = uqhelpers.get_exceedance_data(pce_model.y_pred, 1e-5)
            filename = 'DoE_QuadHem{:d}_{:s}_{:s}_E6R{:d}_y_ecdf.npy'.format(iquad_orders, metamodel_class, pce_fit_method, r)
            np.save(os.path.join(simparams.data_dir, filename), pce_valid_y_ecdf)

            filename = 'DoE_QuadHem{:d}_{:s}_{:s}_E6R{:d}_score.npy'.format(iquad_orders, metamodel_class, pce_fit_method, r)
            np.save(os.path.join(simparams.data_dir, filename), pce_model.scores[0])



    ### 2. PCE model based on Optimal design points, fitting with OLS

    alpha           = [1.0, 1.1, 1.3, 1.5]
    opt_cri         = 'D'
    pce_fit_method  = 'OLS'
    for iquad_orders in quad_orders:
        poly_order = iquad_orders - 1
        for r in range(10):
            basis = cp.orth_ttr(poly_order,dist_zeta)
            for ia in alpha:
                ### ============ Get training points ============
                num_basis= min(int(len(basis)*ia), int(1E6))
                filename = 'DoE_McsE6R{:d}_q{:d}_Opt{:s}{:d}.npy'.format(r, iquad_orders,opt_cri, num_basis)
                data_set = np.load(os.path.join(simparams.data_dir,'DoE_McsE6_OPT_PCE', filename))
                print('  > {:<10s}: {:s}'.format('filename', filename))
                print('    {:<10s}: {}'.format('data shape', data_set.shape))
                train_u  = data_set[1:3,:] 
                train_x  = data_set[3:5,:] 
                train_y  = data_set[5  ,:] 

                ### ============ Get Surrogate Model for each QoI============
                pce_model = museuq.PCE(poly_order, dist_zeta)
                pce_model.fit(train_u, train_y, fit_method=pce_fit_method)

                ### ============ Validating surrogate models at training points ============

                filename = 'DoE_McsE6R{:d}.npy'.format(r)
                data_set = np.load(os.path.join(simparams.data_dir, filename))
                valid_u  = data_set[0:2,:]
                valid_x  = data_set[2:4,:]
                valid_y  = data_set[4  ,:]

                pce_model.predict(valid_u, valid_y, metrics=metrics2cal,prob=upper_tail_probs,moment=moment2cal)
                pce_valid_y_ecdf = uqhelpers.get_exceedance_data(pce_model.y_pred, 1e-5)

                filename = 'DoE_McsE6{:d}_q{:d}_Opt{:s}{:d}_{:s}_{:s}_y.npy'.format(r, iquad_orders, opt_cri, num_basis,metamodel_class, pce_fit_method)
                np.save(os.path.join(simparams.data_dir, filename), pce_model.y_pred[0])

                filename = 'DoE_McsE6{:d}_q{:d}_Opt{:s}{:d}_{:s}_{:s}_y_ecdf.npy'.format(r, iquad_orders, opt_cri, num_basis,metamodel_class, pce_fit_method)
                np.save(os.path.join(simparams.data_dir, filename), pce_valid_y_ecdf)

                filename = 'DoE_McsE6{:d}_q{:d}_Opt{:s}{:d}_{:s}_{:s}_score.npy'.format(r, iquad_orders, opt_cri, num_basis,metamodel_class, pce_fit_method)
                np.save(os.path.join(simparams.data_dir, filename), pce_model.scores[0])


if __name__ == '__main__':
    main()
