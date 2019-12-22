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
    model_name  = 'ishigami'
    ## 1. Choose Wiener-Askey scheme random variable
    dist_x      = cp.Iid(cp.Uniform(-np.pi, np.pi),3) 
    dist_zeta   = cp.Iid(cp.Uniform(-1, 1),3) 
    ## 2. If transformation needed, like Rosenblatt, need to be done here
    ## Perform Rosenblatt etc
    simparams = museuq.simParameters(model_name, dist_zeta)
    # simparams.set_error()  # no observation error for sdof
    simparams.info()

    # #### --------------------------------------------------------------------- ###
    # #### ------------------------ Validation Data Set   ---------------------- ###
    # #### --------------------------------------------------------------------- ###
    # # doe = museuq.LHS(1e5, ['uniform']*3,ndim=3, params=[(-np.pi, np.pi)]*3)
    # ### distribution parameters are specified by (loc, scale) as scipy.stats.
    # doe = museuq.RandomDesign('MCS', n_samples=1e6, ndim=3, dist_names='uniform', dist_theta=[(-np.pi, 2*np.pi),]*3)
    # # doe = museuq.LHS(1e4, 3, dist_names='uniform', dist_theta=[(-np.pi, 2*np.pi),]*3)

    # for r in range(10):
        # np.random.seed()
        # doe.samples()
        # # print(doe)
        # # print(doe.x[:,:4])
        # # print(doe.u[:,:4])
        # # print(doe.filename)
        # dataIO.save_data(doe.x, doe.filename+'R{:d}'.format(r), simparams.data_dir)
        # solver = museuq.Solver(model_name)
        # solver.run(doe.x)
        # np.save(os.path.join(simparams.data_dir, doe.filename+'R{:d}_y'.format(r) ), solver.y)

        # valid_x = np.load(os.path.join(simparams.data_dir, doe.filename+'R{:d}.npy'.format(r))) 
        # valid_u = (valid_x + np.pi) / np.pi - 1
        # valid_y = np.load(os.path.join(simparams.data_dir, doe.filename+'R{:d}_y.npy'.format(r))) 
        # valid_data = np.vstack((valid_y, valid_x, valid_u))
        # valid_data_ecdf = uqhelpers.get_exceedance_data(valid_data, 1e-3, isExpand=True)
        # filename = os.path.join(simparams.data_dir, doe.filename+'R{:d}_y_ecdf.npy'.format(r))
        # np.save(filename, valid_data_ecdf)

    # valid_x = np.load(os.path.join(simparams.data_dir, 'DoE_McsE4.npy' )) 
    # valid_u = (valid_x + np.pi) / np.pi - 1
    # valid_y = np.load(os.path.join(simparams.data_dir, 'DoE_McsE4_y.npy')) 
    #### --------------------------------------------------------------------- ###
    #### ------------------------ Define DoE parameters ---------------------- ###
    #### --------------------------------------------------------------------- ###
    # doe_method, doe_rule, doe_orders = 'MC', 'R', sorted([1e3]*3)

    # #### --------------------------------------------------------------------- ###
    # #### ----------------- Define surrogate model parameters ----------------- ###
    # #### --------------------------------------------------------------------- ###
    # metamodel_params= {'cal_coeffs': 'Galerkin', 'dist_zeta': dist_zeta}
    metamodel_class = 'PCE'
    pce_fit_method  = 'OLS'

    # ### ------------------------------ Run DOE ----------------------------- ###
    # for idoe_order in range(3,20):
        # doe = museuq.QuadratureDesign('leg', idoe_order, len(dist_zeta))
        # doe.samples()
        # print(' > {:<15s} : {}'.format('Experimental Design', doe))
        # doe.x = -np.pi + np.pi * (doe.u + 1.0 ) 
        # doe_data = np.concatenate((doe.u, doe.x, doe.w.reshape(1,-1)), axis=0)
        # dataIO.save_data(doe_data, doe.filename, simparams.data_dir)

    #### ----------------------------- Run Solver -------------------------- ###
    # solver = museuq.Solver(model_name)
    # for r in range(10):
        # for num_basis in [72,85,99]:
            # filename = 'DoE_McsE4R{:d}_OptD{:d}_p10_x.npy'.format(r, num_basis)
            # samples_x = np.load(os.path.join(simparams.data_dir, filename))
            # print('input file: {}'.format(filename))
            # print(samples_x.shape)
            # # ###>>> option 1: run with input data
            # # ###>>> option 2: run with input file names
            # solver.run(samples_x)
            # np.save(os.path.join(simparams.data_dir, filename[:-6]+'_y'), solver.y)

    #### ----------------------- Build PCE Surrogate Model -------------------- ###
    p_orders= range(3,12)
    solver  = museuq.Solver(model_name)
    alpha   = [1.0, 1.1, 1.3, 1.5]
    dist_u  = cp.Iid(cp.Uniform(-1,1),3)
    opt_cri = 'D'
    for p in p_orders:
        for r in range(1):
            basis = cp.orth_ttr(p,dist_u)
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
