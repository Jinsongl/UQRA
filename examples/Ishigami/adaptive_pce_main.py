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
    ndim        = 3
    model_name  = 'ishigami'
    ## 1. Choose Wiener-Askey scheme random variable
    dist_x      = cp.Iid(cp.Uniform(-np.pi, np.pi),ndim) 
    dist_zeta   = cp.Iid(cp.Uniform(-1, 1),ndim) 
    ## 2. If transformation needed, like Rosenblatt, need to be done here
    ## Perform Rosenblatt etc
    simparams = museuq.simParameters(model_name, dist_zeta)
    # simparams.set_error()  # no observation error for sdof
    simparams.info()


    #### ----------------------- Build PCE Surrogate Model -------------------- ###

    metamodel_class = 'PCE'
    quad_orders     = range(3,16)
    upper_tail_probs= [0.999, 0.9999,0.99999]
    moment2cal      = [1,2,3,4]
    metrics2cal     = [ 'explained_variance_score', 'mean_absolute_error', 'mean_squared_error',
                'median_absolute_error', 'r2_score', 'r2_score_adj', 'moment', 'mquantiles']

    quad_orders= range(3,12)
    solver  = museuq.Solver(model_name)
    opt_cri = 'D'
    for p in quad_orders:
        for r in range(1):
            basis = cp.orth_ttr(p,dist_zeta)
            for ia in alpha:
                ### ============ Get training points ============
                num_basis= min(int(len(basis)*ia), 10000)
                filename = 'DoE_McsE4R{:d}_p{:d}_Opt{:s}{:d}.npy'.format(r, p,opt_cri, num_basis)
                data_set = np.load(os.path.join(simparams.data_dir,'DoE_McsE4_PCE_OLS_OPT', filename))
                print('  > {:<10s}: {:s}'.format('filename', filename))
                print('    {:<10s}: {}'.format('data shape', data_set.shape))
                train_u  = data_set[1:4,:] 
                train_x  = data_set[4:7,:] 
                train_y  = data_set[7  ,:] 
                # # print('Train x: {}'.format(train_x.shape))
                # # print('Train Y: {}'.format(train_y.shape))
                # # print('Train w: {}'.format(train_w.shape))

                ### ============ Get Surrogate Model for each QoI============
                pce_model = museuq.PCE(p, dist_zeta)
                # print(len(pce_model.basis[0]))
                # pce_model.fit(train_u, train_y, w=train_w, fit_method=pce_fit_method)
                pce_model.fit(train_u, train_y, fit_method=pce_fit_method)
                # print(pce_model.poly_coeffs)
                # print(pce_model.basis_coeffs)

                ### ============ Validating surrogate models at training points ============
                metrics = [ 'explained_variance_score',
                            'mean_absolute_error',
                            'mean_squared_error',
                            'median_absolute_error',
                            'r2_score', 'moment', 'mquantiles']
                upper_tail_probs = [0.99,0.999]
                moment = [1,2,3,4]

                filename = 'DoE_McsE4R{:d}.npy'.format(r)
                data_set = np.load(os.path.join(simparams.data_dir, filename))
                valid_u  = data_set[0:3,:]
                valid_x  = data_set[3:6,:]
                valid_y  = data_set[6  ,:]

                pce_valid_y, pce_valid_score = pce_model.predict(valid_u, valid_y, \
                        metrics=metrics,prob=upper_tail_probs,moment=moment)
                filename = 'DoE_McsE4R{:d}_p{:d}_Opt{:s}{:d}_{:s}_{:s}_E4R{:d}_y.npy'.format(r, p, opt_cri, num_basis,metamodel_class, pce_fit_method, r)
                np.save(os.path.join(simparams.data_dir, filename), pce_valid_y)

                filename = 'DoE_McsE4R{:d}_p{:d}_Opt{:s}{:d}_{:s}_{:s}_E4R{:d}_score.npy'.format(r, p, opt_cri, num_basis,metamodel_class, pce_fit_method, r)
                np.save(os.path.join(simparams.data_dir, filename), pce_valid_score)

                # pce_valid_y_ecdf = uqhelpers.get_exceedance_data(pce_valid_y, 1e-5)
                # filename = os.path.join(simparams.data_dir, doe.filename+'_{:s}_{:s}_valid_y_ecdf'.format(metamodel_class, pce_fit_method))
                # np.save(filename, pce_valid_y_ecdf)

                # pce_train_y, pce_train_score = pce_model.predict(train_u, train_y, metrics=metrics,prob=upper_tail_probs,moment=moment)
                # np.save(os.path.join(simparams.data_dir, doe.filename+'_{:s}_{:s}_y'.format(metamodel_class, pce_fit_method)), pce_train_y)
            # np.save(os.path.join(simparams.data_dir, doe.filename+'_{:s}_{:s}_score'.format(metamodel_class, pce_fit_method)), pce_train_score)


if __name__ == '__main__':
    main()
