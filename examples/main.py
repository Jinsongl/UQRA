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
from museuq.utilities import helpers as museuq_helpers 
from museuq.utilities import metrics_collections as museuq_metrics
from museuq.utilities import dataIO as museuq_dataio 
from museuq.environment import Kvitebjorn
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()

def main():
    ## ------------------------ Parameters set-up ----------------------- ###
    prob_fails  = 1e-3              # failure probabilities
    model_name  = 'linear_oscillator'
    ## 1. Choose Wiener-Askey scheme random variable
    # dist_zeta = cp.Uniform(-1,1)
    # dist_zeta = cp.Gamma(4,1)
    dist_zeta = cp.Iid(cp.Normal(),2) 

    ## 2. If transformation needed, like Rosenblatt, need to be done here
    ## Perform Rosenblatt etc

    ## 3. Define independent random variable in physical problems
    # dist_x = cp.Uniform(-np.pi, np.pi)

    error_params=None
    simparams = museuq.setup(model_name, dist_zeta, prob_fails)
    simparams.set_error(error_params)
    simparams.disp()

    ## ------------------------ Define DoE parameters ---------------------- ###
    doe_method, doe_rule, doe_orders = 'QUAD', 'hem', [5,6,7,8]
    # doe_method, doe_rule, doe_orders = 'MC', 'R', [1e2]*3
    quad_doe    = museuq.DoE(doe_method, doe_rule, doe_orders, dist_zeta)
    samples_zeta= quad_doe.get_samples()
    # print(samples_zeta.shape)
    shapes = [isamples_zeta.shape for isamples_zeta in samples_zeta]
    print(shapes)
    samples_x   = [Kvitebjorn.samples(dist_zeta.cdf(isamples_zeta[:2,:])) for isamples_zeta in samples_zeta] 
    quad_doe.set_samples(env=samples_x)
    # print(quad_doe.samples_env)
    quad_doe.save_data(simparams.data_dir)
    quad_doe.disp()
    assert len(samples_x) == len(samples_zeta)

    ## ---------------------- Define Solver parameters ---------------------- ###
    solver      = museuq.Solver(model_name, samples_x)
    samples_y   = solver.run(doe_method=doe_method)
    filename_tags = [itag+'_y' for itag in quad_doe.filename_tags]
    museuq_dataio.save_data(samples_y, quad_doe.filename, simparams.data_dir, filename_tags)
    samples_y_stats = solver.get_stats(simparams.qoi2analysis, simparams.stats)
    filename_tags = [itag+'_y_stats' for itag in quad_doe.filename_tags]
    museuq_dataio.save_data(samples_y_stats, quad_doe.filename, simparams.data_dir, filename_tags)


    # ------------------------ Define surrogate model parameters ---------------------- ###

    # metamodel_params= {'cal_coeffs': 'Galerkin', 'dist_zeta': dist_zeta}

    # samples_zeta = np.load(os.path.join(simparams.data_dir, 'DoE_QuadHem5.npy')

    # for idoe_sample_zeta, idoe_samplex, idoe_sampley_stats, ipoly_order in zip(samples_zeta, samples_x, samples_y_stats, [4,5,6]):
        # metamodel_class, metamodel_basis_setting = 'PCE', ipoly_order 
        # pce_model  = museuq.SurrogateModel(metamodel_class, metamodel_basis_setting, **metamodel_params)
        # x_train    = np.squeeze(idoe_samplex[:-1,:])
        # x_weight   = np.squeeze(idoe_samplex[-1,:])
        # zeta_weight= x_weight 
        # y_train    = np.squeeze(idoe_sampley_stats[:, 4, 2])
        # zeta_train = np.squeeze(idoe_sample_zeta[:-1,:])
        # pce_model.fit(zeta_train, y_train, weight=zeta_weight)
        # y_validate = pce_model.predict(zeta_train)
        # pce_model_scores = pce_model.score(zeta_train, y_train)
    # train_data  = [ x_train, x_weight , y_train, zeta_train, np.array(y_validate)]
        # np.save(os.path.join(simparams.data_dir, fname_train_out), train_data)

        # data_test_params= [1e2, 10, 'R'] ##[nsamples, repeat, sampling rule]












        # for r in range(data_test_params[1]):
            # dist_zeta = pce_model.kwparams['dist_zeta']
            # zeta_mcs  = dist_zeta.sample(data_test_params[0], data_test_params[2])
            # y_pred_mcs= pce_model.predict(zeta_mcs)

            # museuq_helpers.upload2gdrive(fname_test_path+r'{:d}'.format(r),  y_pred_mcs, simparam.data_dir_id)
            # print(' ► Calculating ECDF of MCS data and retrieve data to plot...')
            # y_pred_mcs_ecdf = museuq_helpers.get_exceedance_data(np.array(y_pred_mcs), prob=simparam.prob_fails)
            # # rfname_mcs  = fname_test_path + '{:d}_ecdf'.format(r) 
            # # np.save(rfname_mcs, y_pred_mcs_ecdf)
if __name__ == '__main__':
    main()
