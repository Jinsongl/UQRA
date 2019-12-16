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
from museuq.utilities import helpers as museuq_helpers 
from museuq.utilities import metrics_collections as museuq_metrics
from museuq.utilities import dataIO 
from museuq.environment import Kvitebjorn
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()

def main():
    ## ------------------------ Parameters set-up ----------------------- ###
    model_name  = 'ishigami'
    ## 1. Choose Wiener-Askey scheme random variable
    # dist_zeta = cp.Uniform(-1,1)
    # dist_zeta = cp.Gamma(4,1)
    dist_x      = cp.Iid(cp.Uniform(-np.pi, np.pi),3) 
    dist_zeta   = cp.Iid(cp.Uniform(-1, 1),3) 

    ## 2. If transformation needed, like Rosenblatt, need to be done here
    ## Perform Rosenblatt etc

    ## 3. Define independent random variable in physical problems
    # dist_x = cp.Uniform(-np.pi, np.pi)

    simparams = museuq.simParameters(model_name, dist_zeta)
    # simparams.set_error()  # no observation error for sdof
    simparams.info()


    #### --------------------------------------------------------------------- ###
    #### ------------------------ Validation Data Set   ---------------------- ###
    #### --------------------------------------------------------------------- ###
    # doe = museuq.RandomDesign('MCS', 1e6, ['uniform']*3,ndim=3, params=[(-np.pi, np.pi)]*3)
    # doe.samples()
    # print(doe)
    # print(doe.x.shape)
    # print(doe.filename)
    # dataIO.save_data(doe.x, doe.filename, simparams.data_dir)
    # solver = museuq.Solver(model_name)
    # solver.run(doe.x)
    # np.save(os.path.join(simparams.data_dir, doe.filename+'_y'), solver.y)
    valid_x = np.load(os.path.join(simparams.data_dir, 'DoE_McsE6.npy')) 
    valid_u = (valid_x + np.pi) / np.pi - 1
    valid_y = np.load(os.path.join(simparams.data_dir, 'DoE_McsE6_y.npy')) 

    #### --------------------------------------------------------------------- ###
    #### ------------------------ Define DoE parameters ---------------------- ###
    #### --------------------------------------------------------------------- ###
    # doe_method, doe_rule, doe_orders = 'MC', 'R', sorted([1e3]*3)

    #### --------------------------------------------------------------------- ###
    #### ----------------- Define surrogate model parameters ----------------- ###
    #### --------------------------------------------------------------------- ###
    # metamodel_params= {'cal_coeffs': 'Galerkin', 'dist_zeta': dist_zeta}
    metamodel_class = 'PCE'
    pce_fit_method  = 'GLK'

    #### ------------------------------ Run DOE ----------------------------- ###
    for idoe_order in range(3,20):
        doe = museuq.QuadratureDesign('leg', idoe_order, len(dist_zeta))
        doe.samples()
        print(' > {:<15s} : {}'.format('Experimental Design', doe))
        doe.x = -np.pi + np.pi * (doe.u + 1.0 ) 
        doe_data = np.concatenate((doe.u, doe.x, doe.w.reshape(1,-1)), axis=0)
        dataIO.save_data(doe_data, doe.filename, simparams.data_dir)

    #### ----------------------------- Run Solver -------------------------- ###
        solver = museuq.Solver(model_name)
        # ###>>> option 1: run with input data
        # ###>>> option 2: run with input file names
        solver.run(doe.x)
        np.save(os.path.join(simparams.data_dir, doe.filename+'_y'), solver.y)
    #### ----------------------- Build PCE Surrogate Model -------------------- ###

        ### ============ Get training points ============
        train_w     = doe.w
        train_u     = doe.u
        train_y     = solver.y
        # print('Train x: {}'.format(train_x.shape))
        # print('Train Y: {}'.format(train_y.shape))
        # print('Train w: {}'.format(train_w.shape))

        ### ============ Get Surrogate Model for each QoI============
        pce_model = museuq.PCE(idoe_order-1, dist_zeta)
        print(len(pce_model.basis[0]))
        pce_model.fit(train_u, train_y, w=train_w, fit_method=pce_fit_method)
        # pce_model.fit(train_u, train_y,fit_method=pce_fit_method)
        # print(pce_model.poly_coeffs)
        # print(pce_model.basis_coeffs)

        ### ============ Validating surrogate models at training points ============
        metrics = [ 'explained_variance_score',
                    'mean_absolute_error',
                    'mean_squared_error',
                    'median_absolute_error',
                    'r2_score', 'moment', 'mquantiles']
        upper_tail_probs = [0.99,0.999,0.9999]
        moment = [1,2,3,4]

        pce_valid_y, pce_valid_score = pce_model.predict(valid_u, valid_y, metrics=metrics,prob=upper_tail_probs,moment=moment)
        np.save(os.path.join(simparams.data_dir, doe.filename+'_{:s}_{:s}_valid_y'.format(metamodel_class, pce_fit_method)), pce_valid_y)
        np.save(os.path.join(simparams.data_dir, doe.filename+'_{:s}_{:s}_valid_score'.format(metamodel_class, pce_fit_method)), pce_valid_score)

        pce_train_y, pce_train_score = pce_model.predict(train_u, train_y, metrics=metrics,prob=upper_tail_probs,moment=moment)
        np.save(os.path.join(simparams.data_dir, doe.filename+'_{:s}_{:s}_y'.format(metamodel_class, pce_fit_method)), pce_train_y)
        np.save(os.path.join(simparams.data_dir, doe.filename+'_{:s}_{:s}_score'.format(metamodel_class, pce_fit_method)), pce_train_score)

        ### ============ Make prediction at specified points (hs, tp) ============
        # hstp_grid   = np.load(os.path.join(simparams.data_dir, 'HsTp_grid118.npy'))
        # hstp_grid_u = Kvitebjorn.cdf(hstp_grid)
        # hstp_grid_z = np.array([cp.Normal().inv(hstp_grid_u[0,:]), cp.Normal().inv(hstp_grid_u[1,:])])
        # eta_grid    = pce_model.predict(hstp_grid_z)


        # valid_scores= np.array([eta_valid_score, y_valid_score])
        # filename_   = doe.filename+'{:s}_y_{:s}_score.npy'.format(itag, metamodel_class)
        # np.save(os.path.join(simparams.data_dir, filename_),valid_scores)

        # valid_data  = np.array([eta_valid, y_valid])
        # filename_   = doe.filename+'{:s}_y_{:s}_valid.npy'.format(itag, metamodel_class)
        # np.save(os.path.join(simparams.data_dir, filename_),valid_data)

        # grid_data   = np.array([hstp_grid_z, hstp_grid, eta_grid, y_grid])
        # filename_   = doe.filename+'{:s}_y_{:s}_grid.npy'.format(itag, metamodel_class)
        # np.save(os.path.join(simparams.data_dir, filename_),grid_data)

        # ============ Make prediction with monte carlo samples ============
        # print('>>> Prediction with surrogate models... ') 

        # test_doe_method, test_doe_rule, test_doe_orders, test_repeat = 'MC', 'R', 1e6, 10
        # # print( [test_doe_orders]*test_repeat)
        # test_doe = museuq.DoE(test_doe_method, test_doe_rule, [test_doe_orders]*test_repeat)
        # # pbar = tqdm(range(data_test_params[1]), ascii=True, desc="   - ")

        # mcs_filenames = [ 'DoE_McRE6R{:d}.npy'.format(r) for r in range(10)] 
        # pbar = tqdm(mcs_filenames, ascii=True, desc="   - ")

        # for itest_tag, ifilename in zip(test_doe.filename_tags, pbar):
            # museuq_helpers.blockPrint()
            # ### >>> option1: regenerate MCS samples randomly
            # # dist_zeta   = y_pce_model.kwparams['dist_zeta']
            # # zeta_mcs    = dist_zeta.sample(test_doe_orders, rule=test_doe_rule).reshape(1,-1)

            # ### >>> option2: load MCS samples from files
            # data_set    = np.load(os.path.join(simparams.data_dir, ifilename))
            # zeta_mcs    = data_set[:2,:]
            # eta_pred_mcs= pce_model.predict(zeta_mcs)
            # y_pred_mcs  = y_pce_model.predict(zeta_mcs)
            # data_pred   = np.array([eta_pred_mcs, y_pred_mcs])
            # filename_   = doe.filename + '{:s}_{:s}_pred_{:s}.npy'.format(itag, metamodel_class, itest_tag)
            # np.save(os.path.join(simparams.data_dir, filename_), data_pred)
            # museuq_helpers.enablePrint()

            # # # museuq_helpers.upload2gdrive(fname_test_path+r'{:d}'.format(r),  y_pred_mcs, simparam.data_dir_id)
            # # # print(' > Calculating ECDF of MCS data and retrieve data to plot...')
            # # # eta_pred_mcs_ecdf = museuq_helpers.get_exceedance_data(np.array(eta_pred_mcs), prob=simparams.prob_fails)
            # # # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:s}_pred_r{:d}_ecdf_pf{}_eta.npy'.format(itag,r,str(prob_fails)[-1])),eta_pred_mcs_ecdf)
            # # # y_pred_mcs_ecdf = museuq_helpers.get_exceedance_data(np.array(y_pred_mcs), prob=simparams.prob_fails)
            # # # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:s}_pred_r{:d}_ecdf_pf{}_y.npy'.format(itag,r,str(prob_fails)[-1])),y_pred_mcs_ecdf)


if __name__ == '__main__':
    main()
