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
    model_name  = 'BENCH4'
    ## 1. Choose Wiener-Askey scheme random variable
    # dist_zeta = cp.Uniform(-1,1)
    # dist_zeta = cp.Gamma(4,1)
    dist_normal = cp.Normal()
    dist_zeta   = cp.Normal()
    dist_x      = cp.Normal(5,5)  ## (mu, sigma)

    ## 2. If transformation needed, like Rosenblatt, need to be done here
    ## Perform Rosenblatt etc

    ## 3. Define independent random variable in physical problems
    # dist_x = cp.Uniform(-np.pi, np.pi)

    simparams = museuq.simParameters(model_name, dist_zeta, prob_fails=prob_fails)
    # simparams.set_error()
    # simparams.set_error('normal',loc=0, scale=50)
    # simparams.set_error('normal',cov=0.15)
    simparams.set_error('Gumbel',cov=0.15)
    simparams.info()

    ## ------------------------ Define DoE parameters ---------------------- ###
    doe_method, doe_rule, doe_orders = 'QUAD', 'hem', [5]*25
    # doe_method, doe_rule, doe_orders = 'MC', 'R', [1e7]*10
    doe    = museuq.DoE(doe_method, doe_rule, doe_orders, dist_zeta)

    ### >>> comment below out to skip DoE process 
    samples_zeta= doe.get_samples()
    samples_x = doe.mappingto(dist_x)
    doe.info()
    doe.save_data(simparams.data_dir)
    assert len(samples_x) == len(samples_zeta)

    #### --------------------------------------------------------------------------- ###
    #### -------------------------------- Run Solver ------------------------------- ###
    #### --------------------------------------------------------------------------- ###

    #### >>> option 1: Run solver directly after DoE 

    solver   = museuq.Solver(model_name, samples_x)
    solver        = museuq.Solver(model_name, samples_x, error=simparams.error)
    samples_y     = solver.run()
    filename_tags = [itag+'_y_{:s}'.format(simparams.error.name) for itag in doe.filename_tags]
    museuq_dataio.save_data(samples_y, doe.filename, simparams.data_dir, filename_tags)

    ##>> for QoI statistics
    # samples_stats = solver.get_stats()
    # filename_tags = [itag+'_stats' for itag in doe.filename_tags]
    # museuq_dataio.save_data(samples_stats, doe.filename, simparams.data_dir, filename_tags)

    #### >>> option 2: Run solver with samples from data files

    # for itag in doe.filename_tags:
        # data_set    = np.load(os.path.join(simparams.data_dir, doe.filename + '{:s}.npy'.format(itag)))
        # samples_zeta= data_set[0,:].reshape(1,-1)
        # samples_x   = data_set[1,:].reshape(1,-1)
        # solver      = museuq.Solver(model_name, samples_x, error=simparams.error)
        # samples_y   = solver.run()
        # museuq_dataio.save_data(samples_y, doe.filename, simparams.data_dir, itag+'_y_{:s}'.format(simparams.error.name))

    ###------------------------ Define surrogate model parameters ---------------------- ###

    ### -------------------------------- PCE Surrogate Model -------------------- ###
    # metamodel_params= {'cal_coeffs': 'Galerkin', 'dist_zeta': dist_zeta}
    # metamodel_class = 'PCE'
    # metamodel_basis_setting = doe_orders
    
    # for itag, iquad_order in zip(doe.filename_tags, metamodel_basis_setting):
        # ### ============ Get training points ============
        # data_set    = np.load(os.path.join(simparams.data_dir, doe.filename + '{:s}.npy'.format(itag)))
        # train_zeta  = data_set[0,:].reshape(1,-1)
        # train_w     = np.squeeze(data_set[1,:])
        # train_x     = data_set[2,:].reshape(1,-1)
        # data_set    = np.load(os.path.join(simparams.data_dir, doe.filename + '{:s}_y_{:s}.npy'.format(itag,simparams.error.name)))
        # train_y     = np.squeeze(data_set)

        # ### ============ Get Surrogate Model for each QoI============
        # print('Surrogate Model for {:s}: '.format(model_name.upper())) 
        # y_pce_model = museuq.SurrogateModel(metamodel_class, [iquad_order-1], **metamodel_params)
        # y_pce_model.fit(train_zeta, train_y, weight=train_w)

        # ### ============ Validating surrogate models at training points ============
        # print('>>> Validating surrogate model...')
        # y_valid, y_valid_scores = y_pce_model.predict(train_zeta, train_y)



        # ### ============ Make prediction at specified points (hs, tp) ============
        # zeta_grid   = np.linspace(-3,3,600)
        # x_grid      = zeta_grid * 5 + 5
        # y_grid      = y_pce_model.predict(zeta_grid.reshape(1,-1))
        # res_grid    = np.array([zeta_grid, x_grid, y_grid])

        # ### ============ Save data  ============
        # # data_scores = np.array([eta_validate_scores, y_valid_scores]).T
        # filename_ = doe.filename+'{:s}_y_{:s}_{:s}_score.npy'.format(itag, metamodel_class, simparams.error.name)
        # np.save(os.path.join(simparams.data_dir, filename_),y_valid_scores)
        # filename_ = doe.filename+'{:s}_y_{:s}_{:s}_valid.npy'.format(itag, metamodel_class, simparams.error.name)
        # np.save(os.path.join(simparams.data_dir, filename_),y_valid)
        # filename_ = doe.filename+'{:s}_{:s}_{:s}_grid.npy'.format(itag, metamodel_class, simparams.error.name)
        # np.save(os.path.join(simparams.data_dir, filename_),res_grid)

        # #### ============ Make prediction with monte carlo samples ============
        # print('>>> Prediction with surrogate models... ') 
        # data_test_params= [1e7, 10, 'R'] ##[nsamples, repeat, sampling rule]
        # # pbar = tqdm(range(data_test_params[1]), ascii=True, desc="   - ")

        # mcs_filenames = [ 'DoE_McRE7R{:d}.npy'.format(r) for r in range(10)] 
        # pbar = tqdm(mcs_filenames, ascii=True, desc="   - ")

        # for r, ifilename in zip(range(data_test_params[1]), pbar):
            # museuq_helpers.blockPrint()

            # ### >>> option1: regenerate MCS samples randomly
            # # dist_zeta   = y_pce_model.kwparams['dist_zeta']
            # # zeta_mcs    = dist_zeta.sample(data_test_params[0], rule=data_test_params[2]).reshape(1,-1)

            # ### >>> option2: load MCS samples from files
            # zeta_mcs, x_mcs = np.load(os.path.join(simparams.data_dir, ifilename))
            # zeta_mcs = zeta_mcs.reshape(1,-1)

            # y_pred_mcs  = y_pce_model.predict(zeta_mcs)
            # data_pred   = np.array(y_pred_mcs)
            # filename_   = doe.filename + '{:s}_{:s}_{:s}_pred_r{:d}.npy'.format(itag, metamodel_class, simparams.error.name, r)
            # np.save(os.path.join(simparams.data_dir, filename_),data_pred)
            # museuq_helpers.enablePrint()

            #### museuq_helpers.upload2gdrive(fname_test_path+r'{:d}'.format(r),  y_pred_mcs, simparam.data_dir_id)
            ###>>>>> Calculating ECDF of MCS data and retrieve data to plot...')
            # y_pred_mcs_ecdf = museuq_helpers.get_exceedance_data(np.array(y_pred_mcs), prob=simparams.prob_fails)
            # filename_   = doe.filename + '{:s}_{:s}_{:s}_pred_r{:d}_ecdf_pf{:s}_y.npy'.format(itag, metamodel_class, simparams.error.name, r, str(prob_fails)[-1])
            # np.save(os.path.join(simparams.data_dir, filename_),y_pred_mcs_ecdf)



    ##### -------------------------------- mPCE Surrogate Model -------------------- ###
    metamodel_params= {'cal_coeffs': 'Galerkin', 'dist_zeta': dist_zeta}
    metamodel_class = 'mPCE'
    metamodel_basis_setting = doe_orders
    
    for iquad_order in np.unique(metamodel_basis_setting):
        ### ============ Get training points ============
        tags        = [i for i in doe.filename_tags if i.startswith(str(iquad_order))]
        train_zeta  = []
        train_w     = [] 
        train_x     = [] 
        train_eta   = []
        train_y     = [] 

        for itag in tags:
            data_set    = np.load(os.path.join(simparams.data_dir, doe.filename + '{:s}.npy'.format(itag)))
            train_zeta.append(data_set[0,:].reshape(1,-1))
            train_w.append(np.squeeze(data_set[1,:]))
            train_x.append(data_set[2,:].reshape(1,-1))
            data_set    = np.load(os.path.join(simparams.data_dir, doe.filename + '{:s}_y_{:s}.npy'.format(itag,simparams.error.name)))
            train_y.append(np.squeeze(data_set))

        ### ============ Get Surrogate Model for each QoI============
        y_pce_model = museuq.SurrogateModel(metamodel_class, [iquad_order-1], **metamodel_params)
        y_pce_model.fit(train_zeta, train_y, weight=train_w)

        ### ============ Validating surrogate models at training points ============
        print('>>> Validating surrogate model...')
        y_valid, y_valid_scores = y_pce_model.predict(train_zeta[0], np.mean(train_y, axis=0))


        # ### ============ Make prediction at specified points ============
        zeta_grid   = np.linspace(-3,3,600)
        x_grid      = zeta_grid * 5 + 5
        y_grid      = y_pce_model.predict(zeta_grid.reshape(1,-1))
        res_grid    = np.array([zeta_grid, x_grid, y_grid])
        ### ============ Save data  ============

        # data_scores = np.array([eta_validate_scores, y_valid_scores]).T
        filename_ = doe.filename+'{:s}_y_{:s}_{:s}_score.npy'.format(itag, metamodel_class, simparams.error.name)
        np.save(os.path.join(simparams.data_dir, filename_),y_valid_scores)
        filename_ = doe.filename+'{:s}_y_{:s}_{:s}_valid.npy'.format(itag, metamodel_class, simparams.error.name)
        np.save(os.path.join(simparams.data_dir, filename_),y_valid)
        filename_ = doe.filename+'{:s}_{:s}_{:s}_grid.npy'.format(itag, metamodel_class, simparams.error.name)
        np.save(os.path.join(simparams.data_dir, filename_),res_grid)

        ## ============ Make prediction with monte carlo samples ============
        print('>>> Prediction with surrogate models... ') 
        data_test_params= [1e6, 10, 'R'] ##[nsamples, repeat, sampling rule]

        mcs_filenames = [ 'DoE_McRE7R{:d}.npy'.format(r) for r in range(10)] 
        pbar = tqdm(mcs_filenames, ascii=True, desc="   - ")
        for r, ifilename in zip(range(data_test_params[1]), pbar):
            museuq_helpers.blockPrint()
            # ### >>> option1: regenerate MCS samples randomly
            # # dist_zeta   = y_pce_model.kwparams['dist_zeta']
            # # zeta_mcs    = dist_zeta.sample(data_test_params[0], rule=data_test_params[2]).reshape(1,-1)

            ### >>> option2: load MCS samples from files
            zeta_mcs, x_mcs = np.load(os.path.join(simparams.data_dir, ifilename))
            zeta_mcs = zeta_mcs.reshape(1,-1)

            y_pred_mcs  = y_pce_model.predict(zeta_mcs)
            data_pred   = np.array(y_pred_mcs)
            filename_   = doe.filename + '{:s}_{:s}_{:s}_pred_r{:d}.npy'.format(itag, metamodel_class, simparams.error.name, r)
            np.save(os.path.join(simparams.data_dir, filename_), data_pred)
            museuq_helpers.enablePrint()

            # museuq_helpers.upload2gdrive(fname_test_path+r'{:d}'.format(r),  y_pred_mcs, simparam.data_dir_id)
            # print(' > Calculating ECDF of MCS data and retrieve data to plot...')
            # eta_pred_mcs_ecdf = museuq_helpers.get_exceedance_data(np.array(eta_pred_mcs), prob=simparams.prob_fails)
            # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{:s}_pred_r{:d}_ecdf_pf{}_eta.npy'.format(iquad_order,metamodel_class,r,str(prob_fails)[-1])),eta_pred_mcs_ecdf)
            # y_pred_mcs_ecdf = museuq_helpers.get_exceedance_data(np.array(y_pred_mcs), prob=simparams.prob_fails)
            # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{:s}_pred_r{:d}_ecdf_pf{}_y.npy'.format(iquad_order,metamodel_class,r,str(prob_fails)[-1])),y_pred_mcs_ecdf)







    # # ### -------------------------------- GPR Surrogate Model -------------------- ###
    # metamodel_params= {'n_restarts_optimizer': 10}

    # metamodel_class = 'GPR'
    # metamodel_basis_setting = [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e4)) + 
            # WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1)) ]
    
    # for iquad_order in doe_orders:
        # data_set    = np.load(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}.npy'.format(iquad_order)))
        # train_zeta  = data_set[0,:].reshape(1,-1)
        # train_w     = np.squeeze(data_set[1,:])
        # train_x     = data_set[2,:].reshape(1,-1)
        # data_set    = np.load(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_y_{:s}.npy'.format(iquad_order,simparams.error.name)))
        # train_y     = np.squeeze(data_set)

        # #### ============ Get Surrogate Model for each QoI============
        # print('Surrogate Model for SDOF response: ') 
        # y_gpr_model = museuq.SurrogateModel(metamodel_class, metamodel_basis_setting, **metamodel_params)
        # y_gpr_model.fit(train_x, train_y)


        # #### ============ Validating surrogate models at training points ============
        # print('>>> Validating surrogate model...')
        # y_valid, y_valid_scores = y_gpr_model.predict(train_x, train_y)

        # ### ============ Make prediction at specified points (hs, tp) ============
        # zeta_grid   = np.linspace(-3,3,600)
        # x_grid      = zeta_grid * 5 + 5
        # y_grid      = y_gpr_model.predict(x_grid.reshape(1,-1))
        # res_grid    = np.array([zeta_grid, x_grid, y_grid])


        # ### ============ Save data  ============

        # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{:s}_{:s}_score.npy'.format(iquad_order, metamodel_class, simparams.error.name)),y_valid_scores)
        # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{:s}_{:s}_valid.npy'.format(iquad_order, metamodel_class, simparams.error.name)),y_valid)
        # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{:s}_{:s}_grid.npy'.format(iquad_order, metamodel_class, simparams.error.name)),res_grid)

        # print('>>> Prediction with surrogate models... ') 
        # data_test_params= [1e7, 10, 'R'] ##[nsamples, repeat, sampling rule]
        # pbar = tqdm(range(data_test_params[1]), ascii=True, desc="   - ")
        # for r in pbar:
            # museuq_helpers.blockPrint()
            # # dist_zeta   = y_gpr_model.kwparams['dist_zeta']
            # zeta_mcs    = dist_zeta.sample(data_test_params[0], rule=data_test_params[2]).reshape(1,-1)
            # x_mcs       = zeta_mcs * 5 + 5
            # y_pred_mcs  = y_gpr_model.predict(x_mcs)
            # data_pred   = np.array(y_pred_mcs)
            # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{:s}_{:s}_pred_r{:d}.npy'.format(iquad_order, metamodel_class, simparams.error.name, r)),data_pred)
            # museuq_helpers.enablePrint()

            # # museuq_helpers.upload2gdrive(fname_test_path+r'{:d}'.format(r),  y_pred_mcs, simparam.data_dir_id)
            # # print(' > Calculating ECDF of MCS data and retrieve data to plot...')
            # y_pred_mcs_ecdf = museuq_helpers.get_exceedance_data(np.array(y_pred_mcs), prob=simparams.prob_fails)
            # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{:s}_{:s}_pred_r{:d}_ecdf_pf{:s}_y.npy'.format(iquad_order, metamodel_class, simparams.error.name,r,str(prob_fails)[-1])),y_pred_mcs_ecdf)
            # # rfname_mcs  = fname_test_path + '{:d}_ecdf'.format(r) 
            # # np.save(rfname_mcs, y_pred_mcs_ecdf)


if __name__ == '__main__':
    main()
