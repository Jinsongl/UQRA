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
import scipy.io
from tqdm import tqdm
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()

def main():

    ## ------------------------ Displaying set up ------------------- ###
    np.set_printoptions(precision=4)
    np.set_printoptions(threshold=8)
    np.set_printoptions(suppress=True)
    pf = [1e-4, 1e-5, 1e-6]
    np.random.seed(100)
    filename_cand = 'DoE_McsE7R0.npy'
    data_dir = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/FPSO_SDOF/Data'
    data_dir_test = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/FPSO_SDOF/TestData'
    fig_dir = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/FPSO_SDOF/Figures'

    ## ------------------------ Environmental Contour ----------------- ###
    EC_filename = 'FPSO_EC_T50_R20.mat'
    data = scipy.io.loadmat(os.path.join(data_dir, EC_filename))
    EC_u = data['u']
    EC_x = data['x']
    EC_y = data['y']
    EC_y_median = np.median(EC_y, axis=0)
    EC_data = np.concatenate((EC_y_median.reshape(1,-1),EC_u,EC_x), axis=0)

    y50_EC_idx = np.argmax(EC_y_median)
    y50_EC     = EC_data[:,y50_EC_idx]
    print('Extreme reponse from EC:')
    print('   {}'.format(y50_EC))

    ## ------------------------ Simulation Parameters ----------------- ###
    solver = uqra.FPSO()
    simparams = uqra.Parameters()
    simparams.solver     = solver
    simparams.pce_degs   = np.array(range(2,16))
    simparams.n_cand     = int(1e5)
    simparams.n_test     = -1
    simparams.doe_method = 'MCS' ### 'mcs', 'D', 'S', 'reference'
    simparams.optimality = 'D' #'D', 'S', None
    # simparams.hem_type   = 'physicists'
    simparams.hem_type   = 'probabilists'
    simparams.fit_method = 'LASSOLARS'
    simparams.n_splits   = 50
    doe_repeats          = 1 if simparams.optimality == 'D' else 50 ## to reduce the randomness from DoE samples 
    alphas               = [1.2] 
    # alphas               = [-1]
    # simparams.num_samples=np.arange(21+1, 130, 5)
    simparams.update()
    simparams.info()

    ### ============ Initial Values ============
    ## ----------- Candidate and testing data set for DoE ----------- ###
    print(' > Getting candidate data set...')
    # u_cand = modeling.get_candidate_data(filename= 'DoE_McsE7R0.npy')
    data_cand = np.load('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/MCS/Norm/DoE_McsE5R0.npy')
    u_cand = data_cand[:solver.ndim, :simparams.n_cand]
    print('     - Candidate data: U: {}'.format(u_cand.shape))
    print(' > Getting test data set...')
    data_test = scipy.io.loadmat('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/FPSO_SDOF/Data/DoE_McsE7R0_y.mat')
    u_test = data_test['u']
    x_test = data_test['x']
    y_test = np.squeeze(data_test['y'])
    print('     - Test data: U: {}, X: {}, Y: {}'.format(u_test.shape, x_test.shape, y_test.shape))

    print(' > Initial samples from Latin Hypercube: ')
    doe = uqra.LHS([stats.norm(0,1),]*solver.ndim)
    u_train, x_train = doe.get_samples(36)
    # y_train =
    print('     - Samples: U: {}, X: {}'.format(u_train.shape, x_train.shape))
    u_train = [u_train, ] * doe_repeats
    x_train = [x_train, ] * doe_repeats
    data_poly_deg = []

    print(' > Starting simulation...')
    for p in simparams.pce_degs:
        print('\n================================================================================')
        simparams.info()
        print('   - Sampling and Fitting:')
        print('     - {:<23s} : {}'.format('Sampling method'  , simparams.doe_method))
        print('     - {:<23s} : {}'.format('Optimality '      , simparams.optimality))
        print('     - {:<23s} : {}'.format('Fitting method'   , simparams.fit_method))
        ## ----------- Define PCE  ----------- ###
        # orth_poly= uqra.Legendre(d=solver.ndim, deg=p)
        orth_poly= uqra.Hermite(d=solver.ndim, deg=p, hem_type=simparams.hem_type)
        pce_model= uqra.PCE(orth_poly)
        pce_model.info()

        modeling = uqra.Modeling(solver, pce_model, simparams)

        # u_test, x_test, y_test = modeling.get_test_data(solver, pce_model) 
        print(uqra.metrics.mquantiles(y_test, 1-np.array(pf)))
        u_cand_p = p ** 0.5 * u_cand if modeling.is_cls_unbounded() else u_cand
        # assert np.array_equal(u_test, x_test)
        with np.printoptions(precision=2):
            u_cand_mean_std = np.array((np.mean(u_cand[0]), np.std(u_cand[0])))
            u_test_mean_std = np.array((np.mean(u_test[0]), np.std(u_test[0])))
            x_test_mean_std = np.array((np.mean(x_test[0]), np.std(x_test[0])))
            u_cand_ref = np.array(modeling.candidate_data_reference())
            u_test_ref = np.array(modeling.test_data_reference())
            # x_test_ref = np.array((solver.distributions[0].mean(), solver.distributions[0].std()))
            print('    - {:<25s} : {}'.format('Candidate Data ', u_cand.shape))
            print('    - {:<25s} : {}'.format('Test Data ', u_test.shape))
            print('    > {:<25s}'.format('Validate data set '))
            print('    - {:<25s} : {} {} '.format('u cand (mean, std)', u_cand_mean_std, u_cand_ref))
            print('    - {:<25s} : {} {} '.format('u test (mean, std)', u_test_mean_std, u_test_ref))
            # print('    - {:<25s} : {} {} '.format('x test (mean, std)', x_test_mean_std, x_test_ref))

        ## ----------- Oversampling ratio ----------- ###
        simparams.update_num_samples(pce_model.num_basis, alphas=alphas)
        print(' > Starting with oversampling ratio {}'.format(np.around(simparams.alphas,2)))

        ## random samples 
        # random_idx = np.random.randint(0, u_cand_p.shape[1], size=(doe_repeats, simparams.num_samples[-1]))
        # u_train = [u_cand_p[:, idx] for idx in random_idx] 
        # x_train = [solver.map_domain(iu_train, pce_model.basis.dist_u) for iu_train in u_train]
        # y_train = [solver.run(ix_train) for ix_train in x_train]
        ## 
        # for i, n in enumerate(simparams.num_samples):
        data_ndoe  = []
        for i in tqdm(range(doe_repeats), ascii=True, ncols=80):
            tqdm.write('\n------------------------------ Resampling: {:d}/{:d} ------------------------------'.format(i, doe_repeats))
            iu_train = u_train[i]
            ix_train = x_train[i]
            iy_train = y_train[i]
            nsamples = simparams.num_samples[-1]
            while nsamples < pce_model.num_basis:
                ### ============ Estimate sparsity ============
                tqdm.write(' > {:<20s}: alpha = {:.2f}, # samples = {:d}'.format(
                    'Sparsity estimating',nsamples/pce_model.num_basis, nsamples))
                if simparams.doe_method.lower().startswith('cls'):
                    w_train = modeling.cal_cls_weight(iu_train, pce_model.basis)
                else:
                    w_train = None
                pce_model.fit('LASSOLARS', iu_train, iy_train, w_train, n_splits=simparams.n_splits)
                # pce_model.var(0.9)  ### returns significant basis which count for 90% variance

                ### ============ adding samples based on sparsity ============
                ### number of new samples cannot large than num_basis
                n = min(len(pce_model.var_pct_basis), pce_model.num_basis - nsamples) 
                tqdm.write(' > {:<20s}: Optimality-> {}; Basis-> {}/{}; # new samples = {:d}'.format(
                    'New samples', simparams.optimality, len(pce_model.active_basis), pce_model.num_basis, n))
                u_train_new, _ = modeling.get_train_data(n, u_cand_p, u_train=iu_train, 
                        basis=pce_model.basis, active_basis=pce_model.active_basis)
                # u_train_new, _ = modeling.get_train_data(n, u_cand_p, u_train=iu_train, basis=pce_model.basis)
                x_train_new = solver.map_domain(u_train_new, pce_model.basis.dist_u)
                y_train_new = solver.run(x_train_new)
                iu_train = np.hstack((iu_train, u_train_new)) 
                ix_train = np.hstack((ix_train, x_train_new)) 
                iy_train = np.hstack((iy_train, y_train_new)) 
                u_train[i] = iu_train
                x_train[i] = ix_train
                y_train[i] = iy_train

                ### ============ Build Surrogate Model ============
                U_train = pce_model.basis.vandermonde(iu_train)
                if simparams.doe_method.lower().startswith('cls'):
                    w_train = modeling.cal_cls_weight(iu_train, pce_model.basis, pce_model.active_index)
                    U_train = modeling.rescale_data(U_train, w_train) 
                else:
                    w_train = None
                    U_train = U_train[:, pce_model.active_index]

                nsamples = iu_train.shape[-1]
                tqdm.write(' > {:<20s}: alpha = {:.2f}, # samples = {:d}'.format(
                    'Fitting sparse model', nsamples/pce_model.num_basis, nsamples))
                pce_model.fit('ols', iu_train, iy_train, w_train, n_splits=simparams.n_splits, active_basis=pce_model.active_basis)
                # pce_model.fit('ols', iu_train, iy_train, w_train, n_splits=simparams.n_splits)
                y_train_hat = pce_model.predict(iu_train)
                y_test_hat  = pce_model.predict(u_test)
                ## condition number, kappa = max(svd)/min(svd)
                _, sig_value, _ = np.linalg.svd(U_train)
                kappa = max(abs(sig_value)) / min(abs(sig_value)) 

                
                test_mse = uqra.metrics.mean_squared_error(y_test, y_test_hat)
                QoI   = uqra.metrics.mquantiles(y_test_hat, 1-np.array(pf))
                data_ = np.array([p, nsamples, kappa, pce_model.score, pce_model.cv_error, test_mse])
                data_ = np.append(data_, QoI)
                data_poly_deg.append(data_)

                ### ============ calculating & updating metrics ============
                tqdm.write(' > Summary')
                with np.printoptions(precision=4):
                    tqdm.write('     - {:<15s} : {}'.format( 'QoI'       , QoI))
                    tqdm.write('     - {:<15s} : {:.4f}'.format( 'Test MSE ' , test_mse))
                    tqdm.write('     - {:<15s} : {:.4f}'.format( 'CV MSE'    , pce_model.cv_error))
                    tqdm.write('     - {:<15s} : {:.4f}'.format( 'Score '    , pce_model.score))
                    tqdm.write('     - {:<15s} : {:.4f}'.format( 'kappa '    , kappa))
                    tqdm.write('     ----------------------------------------')

    filename = '{:s}_{:s}_{:s}'.format(solver.nickname, pce_model.tag, simparams.tag)
    try:
        np.save(os.path.join(simparams.data_dir_result, filename), np.array(data_poly_deg))
    except:
        np.save(os.path.join(os.getcwd(), filename), np.array(data_poly_deg))

if __name__ == '__main__':
    main()
