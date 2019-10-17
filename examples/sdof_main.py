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
from museuq.utilities import dataIO as museuq_dataio 
from museuq.environment import Kvitebjorn
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()

def main():
    ## ------------------------ Parameters set-up ----------------------- ###
    prob_fails  = 1e-5              # failure probabilities
    model_name  = 'linear_oscillator'
    ## 1. Choose Wiener-Askey scheme random variable
    # dist_zeta = cp.Uniform(-1,1)
    # dist_zeta = cp.Gamma(4,1)
    dist_normal = cp.Normal()
    dist_zeta = cp.Iid(cp.Normal(),2) 

    ## 2. If transformation needed, like Rosenblatt, need to be done here
    ## Perform Rosenblatt etc

    ## 3. Define independent random variable in physical problems
    # dist_x = cp.Uniform(-np.pi, np.pi)

    simparams = museuq.simParameters(model_name, dist_zeta, prob_fails=prob_fails)
    # simparams.set_error()  # no observation error for sdof
    simparams.info()

    ## ------------------------ Define DoE parameters ---------------------- ###
    # doe_method, doe_rule, doe_orders = 'QUAD', 'hem', sorted([5,6,7,8,9,10])
    doe_method, doe_rule, doe_orders = 'MC', 'R', [1e6]*10 
    doe    = museuq.DoE(doe_method, doe_rule, doe_orders, dist_zeta)
    print(doe.filename_tags)

    #  >>> comment below out to skip DoE process 
    train_zeta= doe.get_samples()
    train_x   = [Kvitebjorn.samples(np.array([dist_normal.cdf(itrain_zeta[0,:]), dist_normal.cdf(itrain_zeta[1,:])])) for itrain_zeta in train_zeta] 
    doe.set_samples(env=train_x)
    doe.info()
    doe.save_data(simparams.data_dir)
    # doe.save_data(os.getcwd())
    assert len(train_x) == len(train_zeta)

    #### --------------------------------------------------------------------------- ###
    #### -------------------------------- Run Solver ------------------------------- ###
    #### --------------------------------------------------------------------------- ###

    #### >>> option 1: Run solver directly after DoE 

    solver      = museuq.Solver(model_name, train_x)
    ## for sdof, if want to return full time series, specify return_all=True. 
    ## Otherwise only statistics of time sereis will be returned and time series data will be discarded
    train_y     = solver.run() 
    filename_tags = [itag+'_stats' for itag in doe.filename_tags]
    museuq_dataio.save_data(train_y, doe.filename, simparams.data_dir, filename_tags)  

    #### >>> option 2: Run solver with samples from data files

    # for itag in doe.filename_tags:
        # data_set    = np.load(os.path.join(simparams.data_dir, doe.filename + '{:s}.npy'.format(itag)))
        # train_zeta  = data_set[:2,:]
        # train_x     = data_set[2:,:]
        # solver      = museuq.Solver(model_name, train_x)
        # train_y     = solver.run(doe_method=doe_method)
        # museuq_dataio.save_data(train_y, doe.filename, simparams.data_dir, itag+'_stats')

    ###------------------------ Define surrogate model parameters ---------------------- ###

    ### -------------------------------- PCE Surrogate Model -------------------- ###

    # metamodel_params= {'cal_coeffs': 'Galerkin', 'dist_zeta': dist_zeta}
    # metamodel_class = 'PCE'
    # metamodel_basis_setting = doe_orders
    
    # for itag, iquad_order in zip(doe.filename_tags, metamodel_basis_setting):
        # ### ============ Get training points ============
        # data_set    = np.load(os.path.join(simparams.data_dir, doe.filename + '{:s}.npy'.format(itag)))
        # train_zeta  = data_set[:2,:]
        # train_w     = data_set[2 ,:]
        # train_x     = data_set[3:5,:]
        # data_set    = np.load(os.path.join(simparams.data_dir, doe.filename + '{:s}_stats.npy'.format(itag)))
        # train_eta   = np.squeeze(data_set[:, 4, 0])
        # train_y     = np.squeeze(data_set[:, 4, 1])

        # ### ============ Get Surrogate Model for each QoI============
        # print('Surrogate Model for eta: ') 
        # eta_pce_model = museuq.SurrogateModel(metamodel_class, [iquad_order-1], **metamodel_params)
        # eta_pce_model.fit(train_zeta, train_eta, weight=train_w)

        # ### ============ Validating surrogate models at training points ============
        # print('>>> Validating surrogate model...')
        # eta_valid, eta_valid_score = eta_pce_model.predict(train_zeta, train_eta)


        # ### ============ Make prediction at specified points (hs, tp) ============
        # hstp_grid   = np.load(os.path.join(simparams.data_dir, 'HsTp_grid118.npy'))
        # hstp_grid_u = Kvitebjorn.cdf(hstp_grid)
        # hstp_grid_z = np.array([cp.Normal().inv(hstp_grid_u[0,:]), cp.Normal().inv(hstp_grid_u[1,:])])
        # eta_grid    = eta_pce_model.predict(hstp_grid_z)

        # ### ============ Get Surrogate Model for each QoI============
        # print('Surrogate Model for SDOF response: ') 
        # y_pce_model = museuq.SurrogateModel(metamodel_class, [iquad_order-1], **metamodel_params)
        # y_pce_model.fit(train_zeta, train_y, weight=train_w)

        # ### ============ Validating surrogate models at training points ============
        # print('>>> Validating surrogate model...')
        # y_valid, y_valid_score = y_pce_model.predict(train_zeta, train_y)

        # ### ============ Make prediction at specified points (hs, tp) ============
        # y_grid  = y_pce_model.predict(hstp_grid_z)
        # ### ============ Save data  ============
        # # data_scores = np.array([eta_valid_score, y_valid_score]).T
        # # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{}_{}_scores.npy'.format(itag, metamodel_class)),data_scores)

        # # data_valid = np.array([eta_valid,y_valid]).T

        # valid_scores= np.array([eta_valid_score, y_valid_score])
        # filename_   = doe.filename+'{:s}_y_{:s}_score.npy'.format(itag, metamodel_class)
        # np.save(os.path.join(simparams.data_dir, filename_),valid_scores)

        # valid_data  = np.array([eta_valid, y_valid])
        # filename_   = doe.filename+'{:s}_y_{:s}_valid.npy'.format(itag, metamodel_class)
        # np.save(os.path.join(simparams.data_dir, filename_),valid_data)

        # grid_data   = np.array([hstp_grid_z, hstp_grid, eta_grid, y_grid])
        # filename_   = doe.filename+'{:s}_{:s}_grid.npy'.format(itag, metamodel_class)
        # np.save(os.path.join(simparams.data_dir, filename_),grid_data)

        # ## ============ Make prediction with monte carlo samples ============
        # print('>>> Prediction with surrogate models... ') 
        # data_test_params= [1e6, 10, 'R'] ##[nsamples, repeat, sampling rule]
        # # pbar = tqdm(range(data_test_params[1]), ascii=True, desc="   - ")
        # mcs_filenames = [ 'DoE_McRE6R{:d}.npy'.format(r) for r in range(10)] 
        # pbar = tqdm(mcs_filenames, ascii=True, desc="   - ")
        # for r in pbar:
            # museuq_helpers.blockPrint()
            # ### >>> option1: regenerate MCS samples randomly
            # # dist_zeta   = y_pce_model.kwparams['dist_zeta']
            # # zeta_mcs    = dist_zeta.sample(data_test_params[0], rule=data_test_params[2]).reshape(1,-1)

            # ### >>> option2: load MCS samples from files
            # zeta_mcs, x_mcs = np.load(os.path.join(simparams.data_dir, ifilename))
            # zeta_mcs = zeta_mcs.reshape(1,-1)
            # ????
            # train_zeta  = data_set[:2,:]

            # eta_pred_mcs= eta_pce_model.predict(zeta_mcs)
            # y_pred_mcs  = y_pce_model.predict(zeta_mcs)
            # data_pred   = np.array([eta_pred_mcs, y_pred_mcs])
            # filename_   = doe.filename + '{:s}_{:s}_pred_r{:d}.npy'.format(itag, metamodel_class, r)
            # np.save(os.path.join(simparams.data_dir, filename_), data_pred)
            # museuq_helpers.enablePrint()

            # museuq_helpers.upload2gdrive(fname_test_path+r'{:d}'.format(r),  y_pred_mcs, simparam.data_dir_id)
            # print(' > Calculating ECDF of MCS data and retrieve data to plot...')
            # eta_pred_mcs_ecdf = museuq_helpers.get_exceedance_data(np.array(eta_pred_mcs), prob=simparams.prob_fails)
            # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:s}_pred_r{:d}_ecdf_pf{}_eta.npy'.format(itag,r,str(prob_fails)[-1])),eta_pred_mcs_ecdf)
            # y_pred_mcs_ecdf = museuq_helpers.get_exceedance_data(np.array(y_pred_mcs), prob=simparams.prob_fails)
            # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:s}_pred_r{:d}_ecdf_pf{}_y.npy'.format(itag,r,str(prob_fails)[-1])),y_pred_mcs_ecdf)



    ##### -------------------------------- mPCE Surrogate Model -------------------- ###
    # metamodel_params= {'cal_coeffs': 'Galerkin', 'dist_zeta': dist_zeta}
    # metamodel_class = 'mPCE'
    # metamodel_basis_setting = doe_orders
    
    # for iquad_order in np.unique(metamodel_basis_setting):
        # ### ============ Get training points ============
        # tags        = [i for i in doe.filename_tags if i.startswith(str(iquad_order))]
        # train_zeta  = []
        # train_w     = [] 
        # train_x     = [] 
        # train_eta   = []
        # train_y     = [] 

        # for itag in tags:
            # data_set    = np.load(os.path.join(simparams.data_dir, 'DoE_QuadHem{:s}.npy'.format(itag)))
            # train_zeta.append(data_set[:2,:])
            # train_w.append(data_set[2 ,:])
            # train_x.append(data_set[3:5,:])
            # data_set    = np.load(os.path.join(simparams.data_dir, 'DoE_QuadHem{:s}_stats.npy'.format(itag)))
            # train_eta.append(np.squeeze(data_set[:, 4, 0]))
            # train_y.append(np.squeeze(data_set[:, 4, 1]))

        # ### ============ Get Surrogate Model for each QoI============
        # print('Surrogate Model for eta: ') 
        # eta_pce_model = museuq.SurrogateModel(metamodel_class, [iquad_order-1], **metamodel_params)
        # eta_pce_model.fit(train_zeta, train_eta, weight=train_w)

        # ### ============ Validating surrogate models at training points ============
        # print('>>> Validating surrogate model...')
        # # eta_valid, eta_valid_score = eta_pce_model.predict(train_zeta, train_eta)


        # ### ============ Make prediction at specified points (hs, tp) ============
        # hstp_grid   = np.load(os.path.join(simparams.data_dir, 'HsTp_grid118.npy'))
        # hstp_grid_u = Kvitebjorn.cdf(hstp_grid)
        # hstp_grid_z = np.array([cp.Normal().inv(hstp_grid_u[0,:]), cp.Normal().inv(hstp_grid_u[1,:])])
        # eta_grid = eta_pce_model.predict(hstp_grid_z)

        # ### ============ Get Surrogate Model for each QoI============
        # print('Surrogate Model for SDOF response: ') 
        # y_pce_model = museuq.SurrogateModel(metamodel_class, [iquad_order-1], **metamodel_params)
        # y_pce_model.fit(train_zeta, train_y, weight=train_w)

        # ### ============ Validating surrogate models at training points ============
        # print('>>> Validating surrogate model...')
        # # y_valid, y_valid_score = y_pce_model.predict(train_zeta, train_y)

        # ### ============ Make prediction at specified points (hs, tp) ============
        # y_grid = y_pce_model.predict(hstp_grid_z)

        # ### ============ Save data  ============
        # # data_scores = np.array([eta_valid_score, y_valid_score]).T
        # # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{}_{}_scores.npy'.format(itag, metamodel_class)),data_scores)

        # # data_valid = np.array([eta_valid,y_valid]).T
        # data_valid = np.array([eta_grid,y_grid]).T
        # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{}_grid.npy'.format(iquad_order, metamodel_class)),data_valid)

        # ## ============ Make prediction with monte carlo samples ============
        # print('>>> Prediction with surrogate models... ') 
        # data_test_params= [1e6, 10, 'R'] ##[nsamples, repeat, sampling rule]
        # pbar = tqdm(range(data_test_params[1]), ascii=True, desc="   - ")
        # for r in pbar:
            # museuq_helpers.blockPrint()
            # dist_zeta   = eta_pce_model.kwparams['dist_zeta']
            # zeta_mcs    = dist_zeta.sample(data_test_params[0], rule=data_test_params[2])
            # eta_pred_mcs= eta_pce_model.predict(zeta_mcs)
            # y_pred_mcs  = y_pce_model.predict(zeta_mcs)
            # data_pred   = np.array([eta_pred_mcs, y_pred_mcs])
            # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{:s}_pred_r{:d}.npy'.format(iquad_order, metamodel_class, r)),data_pred)
            # museuq_helpers.enablePrint()

            # # museuq_helpers.upload2gdrive(fname_test_path+r'{:d}'.format(r),  y_pred_mcs, simparam.data_dir_id)
            # # print(' > Calculating ECDF of MCS data and retrieve data to plot...')
            # eta_pred_mcs_ecdf = museuq_helpers.get_exceedance_data(np.array(eta_pred_mcs), prob=simparams.prob_fails)
            # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{:s}_pred_r{:d}_ecdf_pf{}_eta.npy'.format(iquad_order,metamodel_class,r,str(prob_fails)[-1])),eta_pred_mcs_ecdf)
            # y_pred_mcs_ecdf = museuq_helpers.get_exceedance_data(np.array(y_pred_mcs), prob=simparams.prob_fails)
            # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{:s}_pred_r{:d}_ecdf_pf{}_y.npy'.format(iquad_order,metamodel_class,r,str(prob_fails)[-1])),y_pred_mcs_ecdf)







    ### -------------------------------- GPR Surrogate Model -------------------- ###
    # metamodel_params= {'n_restarts_optimizer': 10}

    # metamodel_class = 'GPR'
    # metamodel_basis_setting = [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e4)) + 
            # WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1)) ]
    
    # for iquad_order in doe_orders:
        # data_set    = np.load(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}.npy'.format(iquad_order)))
        # train_zeta  = data_set[:2,:]
        # train_w     = data_set[2,:]
        # train_x     = data_set[3:5,:]
        # data_set    = np.load(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_stats.npy'.format(iquad_order)))
        # train_eta   = np.squeeze(data_set[:, 4, 1])
        # train_y     = np.squeeze(data_set[:, 4, 2])

        # ### ============ Get Surrogate Model for each QoI============
        # print('Surrogate Model for eta: ') 
        # eta_gpr_model = museuq.SurrogateModel(metamodel_class, metamodel_basis_setting, **metamodel_params)
        # eta_gpr_model.fit(train_x, train_eta)

        # #### ============ Validating surrogate models at training points ============
        # print('>>> Validating surrogate model...')
        # eta_valid, eta_valid_score = eta_gpr_model.predict(train_x, train_eta)

        # ### ============ Make prediction at specified points (hs, tp) ============
        # hstp_grid   = np.load(os.path.join(simparams.data_dir, 'HsTp_grid118.npy'))
        # eta_grid = eta_gpr_model.predict(hstp_grid)




        # #### ============ Get Surrogate Model for each QoI============
        # print('Surrogate Model for SDOF response: ') 
        # y_gpr_model = museuq.SurrogateModel(metamodel_class, metamodel_basis_setting, **metamodel_params)
        # y_gpr_model.fit(train_x, train_y)


        # #### ============ Validating surrogate models at training points ============
        # print('>>> Validating surrogate model...')
        # y_valid, y_valid_score = y_gpr_model.predict(train_x, train_y)

        # ### ============ Make prediction at specified points (hs, tp) ============
        # y_grid = y_gpr_model.predict(hstp_grid)


        # #### ============ Save data  ============
        # # data_valid = np.array([eta_valid,y_valid]).T
        # # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{}x_valid.npy'.format(iquad_order, metamodel_class)),data_valid)

        # # data_scores = np.array([eta_valid_score, y_valid_score]).T
        # # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{}x_scores.npy'.format(iquad_order, metamodel_class)),data_scores)

        # data_valid = np.array([eta_grid,y_grid]).T
        # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{}_{}_grid.npy'.format(iquad_order, metamodel_class)),data_valid)



        # print('>>> Prediction with surrogate models... ') 
        # data_test_params= [1e7, 10, 'R'] ##[nsamples, repeat, sampling rule]
        # pbar = tqdm(range(data_test_params[1]), ascii=True, desc="   - ")
        # for r in pbar:
            # museuq_helpers.blockPrint()
            # zeta_mcs    = dist_zeta.sample(data_test_params[0], rule=data_test_params[2])
            # x_mcs       = Kvitebjorn.samples(np.array([dist_normal.cdf(zeta_mcs[0,:]), dist_normal.cdf(zeta_mcs[1,:])]))
            # eta_pred_mcs= eta_gpr_model.predict(x_mcs)
            # y_pred_mcs  =   y_gpr_model.predict(x_mcs)
            # data_pred   = np.array([eta_pred_mcs, y_pred_mcs])
            # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{}x_pred_r{:d}.npy'.format(iquad_order,metamodel_class, r)),data_pred)
            # museuq_helpers.enablePrint()

            # # museuq_helpers.upload2gdrive(fname_test_path+r'{:d}'.format(r),  y_pred_mcs, simparam.data_dir_id)
            # # print(' > Calculating ECDF of MCS data and retrieve data to plot...')
            # eta_pred_mcs_ecdf = museuq_helpers.get_exceedance_data(np.array(eta_pred_mcs), prob=simparams.prob_fails)
            # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{}x_pred_r{:d}_ecdf_pf{}_eta.npy'.format(iquad_order,metamodel_class,r,str(prob_fails)[-1])),eta_pred_mcs_ecdf)
            # y_pred_mcs_ecdf = museuq_helpers.get_exceedance_data(np.array(y_pred_mcs), prob=simparams.prob_fails)
            # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{}x_pred_r{:d}_ecdf_pf{}_y.npy'.format(iquad_order,metamodel_class,r,str(prob_fails)[-1])),y_pred_mcs_ecdf)
            # rfname_mcs  = fname_test_path + '{:d}_ecdf'.format(r) 
            # # np.save(rfname_mcs, y_pred_mcs_ecdf)








if __name__ == '__main__':
    main()
