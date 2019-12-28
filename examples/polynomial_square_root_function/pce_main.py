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
from museuq.utilities import metrics_collections as museuq_metrics
from museuq.utilities import dataIO 
from museuq.environment import Kvitebjorn
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()

def main():
    ## ------------------------ Parameters set-up ----------------------- ###
    model_name  = 'polynomial_square_root_function'
    ## 1. Choose Wiener-Askey scheme random variable
    dist_x      = cp.Iid(cp.Normal(),2) 
    dist_zeta   = cp.Iid(cp.Normal(),2) 
    ## 2. If transformation needed, like Rosenblatt, need to be done here
    ## Perform Rosenblatt etc
    simparams = museuq.simParameters(model_name, dist_zeta)
    # simparams.set_error()  # no observation error for sdof
    simparams.info()

    ### ----------------------------- Run Solver -------------------------- ###
    # solver = museuq.Solver(model_name)

    # for r in range(10):
        # filename = 'DoE_McsE6R{:d}.npy'.format(r)
        # data_set = np.load(os.path.join(simparams.data_dir, filename))
        # samples_u= data_set[0:2,:] 
        # samples_x= data_set[2:4,:] 

        # print('input file: {}'.format(filename))
        # print(samples_x.shape)

        # solver.run(samples_x)
        # np.save(os.path.join(simparams.data_dir, filename[:-4]+'_y'), solver.y)


    # for r in range(10):
        # filename = 'DoE_LhsE3R{:d}.npy'.format(r)
        # data_set = np.load(os.path.join(simparams.data_dir, filename))
        # samples_u= data_set[0:2,:] 
        # samples_x= data_set[2:4,:] 

        # print('input file: {}'.format(filename))
        # print(samples_x.shape)

        # solver.run(samples_x)
        # np.save(os.path.join(simparams.data_dir, filename[:-4]+'_y'), solver.y)

    # quad_orders = range(3,20)
    # for iquad_orders in quad_orders:

        # filename = 'DoE_QuadHem{:d}.npy'.format(iquad_orders)
        # data_set = np.load(os.path.join(simparams.data_dir, filename))
        # samples_u= data_set[0:2,:] 
        # samples_x= data_set[2:4,:] 
        # samples_w= data_set[4,:]

        # print('input file: {}'.format(filename))
        # print(samples_x.shape)

        # solver.run(samples_x)
        # np.save(os.path.join(simparams.data_dir, filename[:-4]+'_y'), solver.y)


    #### ----------------------- Build PCE Surrogate Model -------------------- ###
    metamodel_class = 'PCE'
    quad_orders     = range(3,8)
    upper_tail_probs= [0.99,0.999]
    moment2cal      = [1,2,3,4]
    metrics2cal     = [ 'explained_variance_score', 'mean_absolute_error', 'mean_squared_error',
                'median_absolute_error', 'r2_score', 'moment', 'mquantiles']

    ### 1. PCE model based on quadrature design points, fitting with GLK and OLS

    # pce_fit_method  = 'OLS'
    # for iquad_orders in quad_orders:
        # poly_order = iquad_orders - 1
        # ### ============ Get training points ============
        # filename = 'DoE_QuadHem{:d}.npy'.format(iquad_orders)
        # data_set = np.load(os.path.join(simparams.data_dir, filename))
        # print('  > {:<10s}: {:s}'.format('filename', filename))
        # train_u  = data_set[0:2,:] 
        # train_x  = data_set[2:4,:] 
        # train_w  = data_set[4,:]
        # train_y  = data_set[5,:]

        # ### ============ Get Surrogate Model for each QoI============
        # pce_model = museuq.PCE(poly_order, dist_zeta)
        # # print(len(pce_model.basis[0]))
        # # pce_model.fit(train_u, train_y, w=train_w, fit_method=pce_fit_method)
        # pce_model.fit(train_u, train_y, fit_method=pce_fit_method)
        # # print(pce_model.poly_coeffs)
        # # print(pce_model.basis_coeffs)

        # ### ============ Validating surrogate models at training points ============

        # for r in range(10):
            # filename = 'DoE_McsE6R{:d}.npy'.format(r)
            # data_set = np.load(os.path.join(simparams.data_dir, filename))
            # valid_u  = data_set[0:2,:]
            # valid_x  = data_set[2:4,:]
            # valid_y  = data_set[4  ,:]

            # pce_valid_y, pce_valid_score = pce_model.predict(valid_u, valid_y, metrics=metrics2cal, prob=upper_tail_probs, moment=moment2cal)
            # filename = 'DoE_QuadHem{:d}_{:s}_{:s}_E6R{:d}_y.npy'.format(iquad_orders, metamodel_class, pce_fit_method, r)
            # np.save(os.path.join(simparams.data_dir, filename), pce_valid_y)

            # pce_valid_y_ecdf = uqhelpers.get_exceedance_data(pce_valid_y, 1e-5)
            # filename = 'DoE_QuadHem{:d}_{:s}_{:s}_E6R{:d}_y_ecdf.npy'.format(iquad_orders, metamodel_class, pce_fit_method, r)
            # np.save(os.path.join(simparams.data_dir, filename), pce_valid_y_ecdf)

            # filename = 'DoE_QuadHem{:d}_{:s}_{:s}_E6R{:d}_score.npy'.format(iquad_orders, metamodel_class, pce_fit_method, r)
            # np.save(os.path.join(simparams.data_dir, filename), pce_valid_score)



    ### 1. PCE model based on quadrature design points, fitting with GLK and OLS

    alpha           = [1.0, 1.1, 1.3, 1.5]
    opt_cri         = 'D'
    pce_fit_method  = 'OLS'
    for iquad_orders in quad_orders:
        poly_order = iquad_orders - 1
        for r in range(10):
            basis = cp.orth_ttr(poly_order,dist_zeta)
            for ia in alpha:
                ### ============ Get training points ============
                num_basis= min(int(len(basis)*ia), 1000000)
                filename = 'DoE_McsE6R{:d}_q{:d}_Opt{:s}{:d}.npy'.format(r, iquad_orders,opt_cri, num_basis)
                data_set = np.load(os.path.join(simparams.data_dir,'DoE_McsE6_PCE_OLS_OPT', filename))
                print('  > {:<10s}: {:s}'.format('filename', filename))
                print('    {:<10s}: {}'.format('data shape', data_set.shape))
                train_u  = data_set[1:3,:] 
                train_x  = data_set[3:5,:] 
                train_y  = data_set[5  ,:] 

                ### ============ Get Surrogate Model for each QoI============
                pce_model = museuq.PCE(poly_order, dist_zeta)
                # print(len(pce_model.basis[0]))
                # pce_model.fit(train_u, train_y, w=train_w, fit_method=pce_fit_method)
                pce_model.fit(train_u, train_y, fit_method=pce_fit_method)
                # print(pce_model.poly_coeffs)
                # print(pce_model.basis_coeffs)

                ### ============ Validating surrogate models at training points ============

                filename = 'DoE_McsE6R{:d}.npy'.format(r)
                data_set = np.load(os.path.join(simparams.data_dir, filename))
                valid_u  = data_set[0:2,:]
                valid_x  = data_set[2:4,:]
                valid_y  = data_set[4  ,:]

                pce_valid_y, pce_valid_score = pce_model.predict(valid_u, valid_y, metrics=metrics2cal,prob=upper_tail_probs,moment=moment2cal)
                pce_valid_y_ecdf = uqhelpers.get_exceedance_data(pce_valid_y, 1e-5)

                filename = 'DoE_McsE6{:d}_q{:d}_Opt{:s}{:d}_{:s}_{:s}_E6R{:d}_y.npy'.format(r, iquad_orders, opt_cri, num_basis,metamodel_class, pce_fit_method, r)
                np.save(os.path.join(simparams.data_dir, filename), pce_valid_y)

                filename = 'DoE_McsE6{:d}_q{:d}_Opt{:s}{:d}_{:s}_{:s}_E6R{:d}_y_ecdf.npy'.format(r, iquad_orders, opt_cri, num_basis,metamodel_class, pce_fit_method, r)
                np.save(os.path.join(simparams.data_dir, filename), pce_valid_y_ecdf)

                filename = 'DoE_McsE6{:d}_q{:d}_Opt{:s}{:d}_{:s}_{:s}_E6R{:d}_score.npy'.format(r, iquad_orders, opt_cri, num_basis,metamodel_class, pce_fit_method, r)
                np.save(os.path.join(simparams.data_dir, filename), pce_valid_score)


if __name__ == '__main__':
    main()
