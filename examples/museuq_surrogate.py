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
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()

def main():

    data_dir = '/Users/jinsongliu/External/MUSE_UQ_DATA/linear_oscillator/Data'
    filename = 'DoE_QuadHem'
    metrics  = ['max_error', 'mean_absolute_error', 'mean_squared_error','moments', 'upper_tails']
    metamodel_params={'cal_coeffs': 'Galerkin', 'dist_zeta': cp.Iid(cp.Normal(),2)}
    doe_orders, poly_orders = [5,6,7,8], [4,5,6,7]
    for idoe_order, ipoly_order in zip(doe_orders, poly_orders):
        metamodel_class, metamodel_basis_setting = 'PCE', ipoly_order 
        pce_model  = museuq.SurrogateModel(metamodel_class, metamodel_basis_setting, **metamodel_params)
        samples_doe= np.load(os.path.join(data_dir, filename+'{:d}.npy'.format(idoe_order)))
        samples_doe_qoi= np.load(os.path.join(data_dir, filename+'{:d}_y_stats.npy'.format(idoe_order)))
        zeta_train  = np.squeeze(samples_doe[:2,:])
        zeta_weight = np.squeeze(samples_doe[2,:])
        y_train     = np.squeeze(samples_doe_qoi[:,4, 2]) #t,x_t,y_t, f,x_pxx,y_pxx

        pce_model.fit(zeta_train, y_train, weight=zeta_weight)
        y_validate = pce_model.predict(zeta_train)
        pce_model_scores = pce_model.score(zeta_train, y_train, metrics=metrics, moment=np.arange(1,5))
        # train_data  = [ x_train, x_weight , y_train, zeta_train, np.array(y_validate)]
        # np.save(os.path.join(simparams.data_dir, fname_train_out), train_data)

        # data_test_params= [1e2, 10, 'R'] ##[nsamples, repeat, sampling rule]


if __name__ == '__main__':
    main()

