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
from museuq.utilities import helpers as uqhelpers 
from museuq.utilities import metrics as uqmetrics
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()

def main():
    ## ------------------------ Parameters set-up ----------------------- ###
    prob_fails  = 1e-1              # failure probabilities
    model_name  = 'linear_oscillator'
    ## 1. Choose Wiener-Askey scheme random variable
    # dist_zeta = cp.Uniform(-1,1)
    # dist_zeta = cp.Gamma(4,1)
    dist_zeta = cp.Iid(cp.Uniform(0,1),2) 

    ## 2. If transformation needed, like Rosenblatt, need to be done here
    ## Perform Rosenblatt etc

    ## 3. Define independent random variable in physical problems
    # dist_x = cp.Uniform(-np.pi, np.pi)
    dist_x = cp.Iid(cp.Uniform(-np.pi, np.pi),2) 
    error_params=None
    simparams = museuq.setup(model_name, dist_zeta, dist_x, prob_fails)
    simparams.set_error(error_params)
    simparams.disp()

    ## ------------------------ Define DoE parameters ---------------------- ###
    doe_method, doe_rule, doe_orders = 'QUAD', 'hem', [3,4,5,6]
    quad_doe = museuq.DoE(doe_method, doe_rule, doe_orders, dist_zeta)
    samples_zeta= quad_doe.get_samples()
    quad_doe.disp()

    # print(*samples_zeta, sep='\n')
    samples_x   = quad_doe.mappingto(dist_x)
    assert len(samples_x) == len(samples_zeta)

    ## ------------------------ Define Solver parameters ---------------------- ###
    solver = museuq.Solver(model_name, samples_x)
    samples_y = solver.run(quad_doe)
    print(samples_y)

    # ## ------------------------ Define surrogate model parameters ---------------------- ###
    # x_train    = np.squeeze(samples_x[0][0])
    # x_weight   = np.squeeze(samples_x[0][1])
    # zeta_weight= x_weight 
    # y_train    = np.squeeze(samples_y[0])
    # zeta_train = np.squeeze(samples_zeta[0][0])

    # metamodel_class, metamodel_basis_setting = 'PCE', [11,15] 
    # metamodel_params= {'cal_coeffs': 'GQ', 'dist_zeta': dist_zeta}
    # pce_model   = museuq.SurrogateModel(metamodel_class, metamodel_basis_setting, **metamodel_params)

    # pce_model.fit_model(zeta_train, y_train, weight=zeta_weight)
    # y_validate  = pce_model.predict(zeta_train)
    # train_data  = [ x_train, x_weight , y_train, zeta_train, np.array(y_validate)]
    # np.save(os.path.join(simparam.data_dir, fname_train_out), train_data)

    # data_test_params= [1e2, 10, 'R'] ##[nsamples, repeat, sampling rule]

    # for r in range(data_test_params[1]):
        # dist_zeta = pce_model.kwparams['dist_zeta']
        # zeta_mcs  = dist_zeta.sample(data_test_params[0], data_test_params[2])
        # y_pred_mcs= pce_model.predict(zeta_mcs)

        # uqhelpers.upload2gdrive(fname_test_path+r'{:d}'.format(r),  y_pred_mcs, simparam.data_dir_id)
        # print(' ► Calculating ECDF of MCS data and retrieve data to plot...')
        # y_pred_mcs_ecdf = uqhelpers.get_exceedance_data(np.array(y_pred_mcs), prob=simparam.prob_fails)
        # # rfname_mcs  = fname_test_path + '{:d}_ecdf'.format(r) 
        # # np.save(rfname_mcs, y_pred_mcs_ecdf)
if __name__ == '__main__':
    main()
