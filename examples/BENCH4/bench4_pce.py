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
import uqra
import numpy as np, chaospy as cp, os, sys
import warnings
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel,WhiteKernel)
from tqdm import tqdm
from uqra.utilities import helpers as uqra_helpers 
from uqra.utilities import metrics_collections as uqra_metrics
from uqra.utilities import dataIO 
from uqra.environment import Kvitebjorn
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()

def main():
    ## ------------------------ Parameters set-up ----------------------- ###
    model_name  = 'bench4'
    ## 1. Choose Wiener-Askey scheme random variable
    dist_x    = cp.Normal(5,5)
    dist_zeta = cp.Normal()

    ## 2. If transformation needed, like Rosenblatt, need to be done here
    ## Perform Rosenblatt etc

    ## 3. Define independent random variable in physical problems
    # dist_x = cp.Uniform(-np.pi, np.pi)

    simparams = uqra.simParameters(model_name, dist_zeta)
    # simparams.set_error()  # no observation error for sdof
    simparams.info()


    #### --------------------------------------------------------------------- ###
    #### ------------------------ Validation Data Set   ---------------------- ###
    #### --------------------------------------------------------------------- ###
    # doe = uqra.RandomDesign('MCS', 1e6, ['normal'],ndim=1)
    # doe.samples()
    # print(doe)
    # print(doe.x.shape)
    # print(doe.filename)
    # dataIO.save_data(doe.x, doe.filename, simparams.data_dir)
    # solver = uqra.Solver(model_name)
    # solver.run(doe.x)
    # np.save(os.path.join(simparams.data_dir, doe.filename+'_y'), solver.y)
    valid_x = np.load(os.path.join(simparams.data_dir, 'DoE_McsE6.npy')) 
    valid_y = np.load(os.path.join(simparams.data_dir, 'DoE_McsE6_y.npy')) 
    plot_x  = np.linspace(-10,20, 1000).reshape(1,-1)
    plot_u  = (plot_x - 5.0)/5.0 

    solver  = uqra.Solver(model_name)
    solver.run(plot_x)
    plot_y  = solver.y 
    np.save(os.path.join(simparams.data_dir, 'DoE_plot_x'), plot_x)
    np.save(os.path.join(simparams.data_dir, 'DoE_plot_u'), plot_u)
    np.save(os.path.join(simparams.data_dir, 'DoE_plot_y'), plot_y)

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
    for idoe_order in range(2,6):
        doe = uqra.QuadratureDesign('hem', idoe_order, len(dist_zeta))
        doe.samples()
        doe.x = doe.u * 5 + 5
        print(doe)
        doe_data = np.concatenate((doe.u, doe.x, doe.w.reshape(1,-1)), axis=0)
        dataIO.save_data(doe_data, doe.filename, simparams.data_dir)

    #### ----------------------------- Run Solver -------------------------- ###
        solver = uqra.Solver(model_name)
        solver.run(doe.x)
        np.save(os.path.join(simparams.data_dir, doe.filename+'_y'), solver.y)
    #### ----------------------- Build PCE Surrogate Model -------------------- ###

        ### ============ Get training points ============
        train_w     = doe.w
        train_u     = doe.u
        train_y     = solver.y
        # print('Train u: {}'.format(train_u.shape))
        # print('Train Y: {}'.format(train_y.shape))
        # print('Train w: {}'.format(train_w.shape))

        ### ============ Get Surrogate Model for each QoI============
        pce_model = uqra.PCE(idoe_order-1, dist_zeta)
        pce_model.fit(train_u, train_y, w=train_w, fit_method=pce_fit_method)

        # pce_model.fit(train_x, train_y,fit_method=pce_fit_method)
        # print(pce_model.poly_coeffs)
        # print(pce_model.basis_coeffs)

        ### ============ Validating surrogate models at training points ============
        metrics = ['explained_variance_score',
                'mean_absolute_error',
                'mean_squared_error',
                'median_absolute_error',
                'r2_score',
                'moment',
                'mquantiles']
        upper_tail_probs = [0.99,0.999,0.9999]
        moment = [1,2,3,4]

        # pce_valid_y, pce_valid_score = pce_model.predict(valid_x, valid_y, metrics=metrics,prob=upper_tail_probs,moment=moment)
        # np.save(os.path.join(simparams.data_dir, doe.filename+'_{:s}_{:s}_valid_y'.format(metamodel_class, pce_fit_method)), pce_valid_y)
        # np.save(os.path.join(simparams.data_dir, doe.filename+'_{:s}_{:s}_valid_score'.format(metamodel_class, pce_fit_method)), pce_valid_score)

        pce_train_y, pce_train_score = pce_model.predict(train_u, train_y, metrics=metrics,prob=upper_tail_probs,moment=moment)
        np.save(os.path.join(simparams.data_dir, doe.filename+'_{:s}_{:s}_y'.format(metamodel_class, pce_fit_method)), pce_train_y)
        np.save(os.path.join(simparams.data_dir, doe.filename+'_{:s}_{:s}_score'.format(metamodel_class, pce_fit_method)), pce_train_score)

        pce_plot_y, pce_plot_score = pce_model.predict(plot_u, plot_y, metrics=metrics,prob=upper_tail_probs,moment=moment)
        np.save(os.path.join(simparams.data_dir, doe.filename+'_{:s}_{:s}_plot_y'.format(metamodel_class, pce_fit_method)), pce_plot_y)
        np.save(os.path.join(simparams.data_dir, doe.filename+'_{:s}_{:s}_plot_score'.format(metamodel_class, pce_fit_method)), pce_plot_score)
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
        # test_doe = uqra.DoE(test_doe_method, test_doe_rule, [test_doe_orders]*test_repeat)
        # # pbar = tqdm(range(data_test_params[1]), ascii=True, desc="   - ")

        # mcs_filenames = [ 'DoE_McRE6R{:d}.npy'.format(r) for r in range(10)] 
        # pbar = tqdm(mcs_filenames, ascii=True, desc="   - ")

        # for itest_tag, ifilename in zip(test_doe.filename_tags, pbar):
            # uqra_helpers.blockPrint()
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
            # uqra_helpers.enablePrint()

            # # # uqra_helpers.upload2gdrive(fname_test_path+r'{:d}'.format(r),  y_pred_mcs, simparam.data_dir_id)
            # # # print(' > Calculating ECDF of MCS data and retrieve data to plot...')
            # # # eta_pred_mcs_ecdf = uqra_helpers.get_exceedance_data(np.array(eta_pred_mcs), prob=simparams.prob_fails)
            # # # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:s}_pred_r{:d}_ecdf_pf{}_eta.npy'.format(itag,r,str(prob_fails)[-1])),eta_pred_mcs_ecdf)
            # # # y_pred_mcs_ecdf = uqra_helpers.get_exceedance_data(np.array(y_pred_mcs), prob=simparams.prob_fails)
            # # # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:s}_pred_r{:d}_ecdf_pf{}_y.npy'.format(itag,r,str(prob_fails)[-1])),y_pred_mcs_ecdf)


if __name__ == '__main__':
    main()
