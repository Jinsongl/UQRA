#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import uqra, warnings, random, math
import numpy as np, os, sys
import collections
import scipy.stats as stats
import scipy
import scipy.io
from tqdm import tqdm
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()
def main():

    ## ------------------------ Displaying set up ------------------- ###
    np.random.seed(100)
    np.set_printoptions(precision=4)
    np.set_printoptions(threshold=8)
    np.set_printoptions(suppress=True)
    pf       = 0.5/(50*365.25*24)
    data_dir = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/FPSO_SDOF/Data' 
    samples_dir = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples'
    ## ------------------------ Environmental Contour ----------------- ###
    site_env_dist = uqra.environment.Kvitebjorn()
    print('------------------------------------------------------------')
    print('>>> Environmental Contour for Model: FPSO                   ')
    print('------------------------------------------------------------')
    EC2D_data   = np.load('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/FPSO_SDOF/Data/FPSO_DoE_EC2D_T50.npy') 
    EC2D_median = EC2D_data[-1]
    y50_EC_idx  = np.argmax(EC2D_median)
    y50_EC      = EC2D_data[:,y50_EC_idx]
    print(' > Extreme reponse from EC:')
    print('   - y: {:.2f}'.format(y50_EC[-1]))
    print('   - u: {}, x: {}'.format(y50_EC[:2], y50_EC[2:4]))

    y50_EC_boots = np.load('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/FPSO_SDOF/Data/FPSO_DoE_EC2D_T50_bootstrap.npy')
    u_origin = np.array([np.min(y50_EC_boots[0]), np.max(y50_EC_boots[1])]).reshape(-1,1)
    x_origin = np.array([np.min(y50_EC_boots[2]), np.max(y50_EC_boots[3])]).reshape(-1,1)
    print(' > Extreme reponse from EC (Bootstrapping):')
    print('   - y: [mean, std]= [{:.2f}, {:.2f}]'.format(np.mean(y50_EC_boots[-1]), np.std(y50_EC_boots[-1])))
    print('   - u: min(u1): {:.2f}, max(u2): {:.2f}'.format(u_origin[0,0], u_origin[1,0] ))
    print('   - x: min(Hs): {:.2f}, max(Tp): {:.2f}'.format(x_origin[0,0], x_origin[1,0] ))

    ## ------------------------ Simulation Parameters ----------------- ###
    solver = uqra.FPSO()
    simparams = uqra.Parameters()
    simparams.solver     = solver
    simparams.pce_degs   = np.array(range(2,11))
    simparams.n_cand     = int(1e5)
    simparams.n_test     = -1
    simparams.doe_method = 'MCS' ### 'mcs', 'D', 'S', 'reference'
    simparams.optimality = 'D'#'D', 'S', None
    # simparams.hem_type   = 'physicists'
    simparams.hem_type   = 'probabilists'
    simparams.fit_method = 'LASSOLARS'
    simparams.n_splits   = 50
    # repeats              = 50 if simparams.optimality is None else 1
    # alphas               = np.arange(3,11)/10 
    alphas               = 1.2
    # alphas               = [1.2]
    # # simparams.num_samples=np.arange(21+1, 130, 5)
    simparams.update()
    simparams.info()
    print('\n================================================================================')
    print('   - Sampling and Fitting:')
    print('     - {:<23s} : {}'.format('Sampling method'  , simparams.doe_method))
    print('     - {:<23s} : {}'.format('Optimality '      , simparams.optimality))
    print('     - {:<23s} : {}'.format('Fitting method'   , simparams.fit_method))

    ## ----------- Candidate and testing data set for DoE ----------- ###

    print(' > Getting testing data set...')
    filename  = 'FPSO_DoE_McsE5R0.npy' 
    data_test = np.load(os.path.join(data_dir, filename)) 
    u_test, x_test, y_test = data_test[:2], data_test[2:4], np.squeeze(data_test[-1])
    print('   - Test data: U: {}, X:{}, Y:{}'.format(u_test.shape, x_test.shape, y_test.shape))
    print(np.mean(u_test, axis=1))
    print(np.std(u_test, axis=1))
    # print('     - U (mean, std) : {:.2f}, {:.2f}'.format(np.mean(u_test), np.std(u_test)))
    # print('     - X (mean, std) : {:.2f}, {:.2f}'.format(np.mean(x_test), np.std(x_test)))
    # print('     - Y (mean, std) : {:.2f}, {:.2f}'.format(np.mean(y_test), np.std(y_test)))
    # print(uqra.metrics.mquantiles(y_test, 1-np.array(pf)))

    print(' > Getting candiate data set...')
    data_cand = data_test

    print(' > Getting prediction data set...')
    filename = 'DoE_McsE7R0.npy'
    u_pred = np.load(os.path.join(samples_dir, simparams.doe_method, 'Norm', filename))[:solver.ndim,:]
    x_pred = site_env_dist.ppf(stats.norm().cdf(u_pred))
    y_pred = -np.inf * np.ones((u_pred.shape[1],))
    data_pred = np.concatenate((u_pred, x_pred, y_pred.reshape(1,-1)))
    idx_in_circle = np.arange(u_pred.shape[1])[np.linalg.norm(u_pred, axis=0)**2 < 2]
    u_pred_in_circle = u_pred[:,idx_in_circle]


    ### ============ Initial Values ============
    data_poly_deg = []
    for pce_deg in simparams.pce_degs:
        print(' > Building surrogate model ...')
        ## ----------- Define PCE  ----------- ###
        # orth_poly= uqra.Legendre(d=solver.ndim, deg=pce_deg)
        orth_poly= uqra.Hermite(d=solver.ndim, deg=pce_deg, hem_type=simparams.hem_type)
        pce_model= uqra.PCE(orth_poly)
        pce_model.info()
        modeling = uqra.Modeling(solver, pce_model, simparams)

        ### ============ Build Surrogate Model ============
        n_train  = int(alphas * pce_model.num_basis)  # + n_ec_samples
        filename = 'DoE_McsE5R0_2Heme{:d}_D.npy'.format(pce_deg)
        doe_idx  = np.load(os.path.join(samples_dir, 'OED', filename))
        u_train  = data_cand[:solver.ndim, doe_idx[:n_train]] - u_origin
        x_train  = data_cand[solver.ndim:2*solver.ndim, doe_idx[:n_train]]
        y_train  = data_cand[-1, doe_idx[:n_train]].reshape(1,-1)
        # u_train  = u_train[:,:n_train]
        # x_train  = x_train[:,:n_train]
        # y_train  = y_train[:,:n_train]

        # n_size   = int(u_train.shape[1]/2.0)
        # u_train  = np.concatenate((u_train[:,:n_train], u_train[:,n_size:n_size+n_train]), axis=1)
        # x_train  = np.concatenate((x_train[:,:n_train], x_train[:,n_size:n_size+n_train]), axis=1)
        # y_train  = np.concatenate((y_train[:,:n_train], y_train[:,n_size:n_size+n_train]), axis=1)

        # n_size   = int(u_train.shape[1]/2.0)
        # u_train  = np.concatenate((EC_u, u_train[:,:n_train], u_train[:,n_size:n_size+n_train]), axis=1)
        # x_train  = np.concatenate((EC_x, x_train[:,:n_train], x_train[:,n_size:n_size+n_train]), axis=1)
        # y_train  = np.concatenate((EC_y[0].reshape(1,-1), y_train[:,:n_train], y_train[:,n_size:n_size+n_train]), axis=1)

        # u_train  = np.concatenate((EC_u, u_train[:,:n_train]), axis=1)
        # x_train  = np.concatenate((EC_x, x_train[:,:n_train]), axis=1)
        # y_train  = np.concatenate((EC_y[0].reshape(1,-1), y_train[:,:n_train]), axis=1)
        # print('     - Train Dataset: U: {}, X: {}, Y: {}'.format(u_train.shape, x_train.shape, y_train.shape))
        U_train  = pce_model.basis.vandermonde(u_train)

        if simparams.doe_method.lower().startswith('cls'):
            w_train = modeling.cal_cls_weight(u_train, pce_model.basis, pce_model.active_index)
            U_train = modeling.rescale_data(U_train, w_train) 
        else:
            w_train = None
            U_train = U_train[:, pce_model.active_index]

        for iy_train in y_train:
            pce_model.fit(simparams.fit_method, u_train, iy_train, w_train, n_splits=simparams.n_splits)
            y_train_hat = pce_model.predict(u_train)
            # y_test_hat  = pce_model.predict(u_test - y50_EC[1:3].reshape(-1,1))
            y_test_hat  = pce_model.predict(u_test-u_origin)
            test_mse    = uqra.metrics.mean_squared_error(y_test, y_test_hat)


            y_pred_in_circle = pce_model.predict(u_pred_in_circle - u_origin)
            y_pred[idx_in_circle] = y_pred_in_circle
            y50_pce = uqra.metrics.mquantiles(y_pred, 1-np.array(pf))
            y50_pce_idx = (np.abs(y_pred - y50_pce)).argmin()
            data_ = np.array([pce_deg, u_train.shape[1], pce_model.score, pce_model.cv_error, test_mse])
            data_ = np.append(data_, y50_pce)
            data_poly_deg.append(data_)

            ### ============ calculating & updating metrics ============
            tqdm.write(' > Summary')
            with np.printoptions(precision=4):
                tqdm.write('     - {:<15s} : {}'.format( 'y50_pce'       , y50_pce))
                tqdm.write('     - {:<15s} : {:.4f}'.format( 'Test MSE ' , test_mse))
                tqdm.write('     - {:<15s} : {:.4f}'.format( 'CV MSE'    , pce_model.cv_error))
                tqdm.write('     - {:<15s} : {:.4f}'.format( 'Score '    , pce_model.score))
                tqdm.write('     - {:<15s} : {}'.format( 'Design state', x_pred[:,y50_pce_idx]))
                # tqdm.write('     - {:<15s} : {:.4f}'.format( 'kappa '    , kappa))
                tqdm.write('     ----------------------------------------')

    filename = '{:s}_{:s}_{:s}'.format(solver.nickname, pce_model.tag, simparams.tag)
    # try:
        # np.save(os.path.join(simparams.data_dir_result, filename), np.array(data_poly_deg))
    # except:
    np.save(os.path.join(os.getcwd(), filename), np.array(data_poly_deg))


if __name__ == '__main__':
    main()
