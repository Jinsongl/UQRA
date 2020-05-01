#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import museuq, warnings, random, math
import numpy as np, os, sys
import collections
import scipy.stats as stats
from scipy import sparse
from tqdm import tqdm
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()

def main():

    ## ------------------------ Displaying set up ------------------- ###
    np.set_printoptions(precision=4)
    np.set_printoptions(threshold=8)
    np.set_printoptions(suppress=True)
    pf          = [1e-4, 1e-5, 1e-6]
    np.random.seed(100)
    ## ------------------------ Define solver ----------------------- ###
    # orth_poly   = museuq.Legendre(d=ndim, deg=p)
    # orth_poly   = museuq.Hermite(d=ndim, deg=p, hem_type='physicists')
    # orth_poly   = museuq.Hermite(d=ndim, deg=p, hem_type='probabilists')
    # solver      = museuq.SparsePoly(orth_poly, sparsity='full', seed=100)

    # solver      = museuq.ExpAbsSum(stats.uniform(-1,2),d=2,c=[-2,1],w=[0.25,-0.75])
    # solver      = museuq.ExpSquareSum(stats.uniform(-1,2),d=2,c=[1,1],w=[1,0.5])
    # solver      = museuq.CornerPeak(stats.uniform(-1,2), d=2)
    # solver      = museuq.ProductPeak(stats.uniform(-1,2), d=2,c=[-3,2],w=[0.5,0.5])
    # solver      = museuq.Franke()
    # solver      = museuq.Ishigami()

    solver      = museuq.ExpAbsSum(stats.norm(0,1),d=2,c=[-2,1],w=[0.25,-0.75])
    # solver      = museuq.ExpSquareSum(stats.norm(0,1),d=2,c=[1,1],w=[1,0.5])
    # solver      = museuq.CornerPeak(stats.norm(0,1), d=3, c=np.array([1,2,3]), w=[0.5,]*3)
    # solver      = museuq.ProductPeak(stats.norm(0,1), d=2, c=[-3,2], w=[0.5,]*2)
    # solver      = museuq.ExpSum(stats.norm(0,1), d=3)
    # solver      = museuq.FourBranchSystem()
    ## ------------------------ Simulation Parameters ----------------- ###
    simparams = museuq.Parameters(solver)
    simparams.pce_degs   = np.array(range(2,7))
    simparams.n_cand     = int(1e5)
    simparams.n_test     = -1
    simparams.doe_method = 'MCS' ### 'mcs', 'D', 'S', 'reference'
    simparams.optimality = 'S'#'D', 'S', None
    # simparams.hem_type   = 'physicists'
    simparams.hem_type   = 'probabilists'
    simparams.fit_method = 'LASSOLARS'
    simparams.n_splits   = 50
    repeats              = 50 if simparams.optimality is None else 1
    alphas               = np.arange(3,11)/10 
    # alphas               = [-1]
    # simparams.num_samples=np.arange(21+1, 130, 5)
    simparams.update()
    simparams.info()

    ### ============ Initial Values ============
    print(' > Starting simulation...')
    data_poly_deg = []
    for p in simparams.pce_degs:
        print('\n================================================================================')
        simparams.info()
        print('   - Sampling and Fitting:')
        print('     - {:<23s} : {}'.format('Sampling method'  , simparams.doe_method))
        print('     - {:<23s} : {}'.format('Optimality '      , simparams.optimality))
        print('     - {:<23s} : {}'.format('Fitting method'   , simparams.fit_method))
        ## ----------- Define PCE  ----------- ###
        # orth_poly= museuq.Legendre(d=solver.ndim, deg=p)
        orth_poly= museuq.Hermite(d=solver.ndim, deg=p, hem_type=simparams.hem_type)
        pce_model= museuq.PCE(orth_poly)
        pce_model.info()

        modeling = museuq.Modeling(solver, pce_model, simparams)
        # modeling.sample_selected=[]

        ## ----------- Candidate and testing data set for DoE ----------- ###
        print(' > Getting candidate data set...')
        u_cand = modeling.get_candidate_data()
        u_test, x_test, y_test = modeling.get_test_data(solver, pce_model) 
        # assert np.array_equal(u_test, x_test)
        with np.printoptions(precision=2):
            u_cand_mean_std = np.array((np.mean(u_cand[0]), np.std(u_cand[0])))
            u_test_mean_std = np.array((np.mean(u_test[0]), np.std(u_test[0])))
            x_test_mean_std = np.array((np.mean(x_test[0]), np.std(x_test[0])))
            u_cand_ref = np.array(modeling.candidate_data_reference())
            u_test_ref = np.array(modeling.test_data_reference())
            x_test_ref = np.array((solver.distributions[0].mean(), solver.distributions[0].std()))
            print('    - {:<25s} : {}'.format('Candidate Data ', u_cand.shape))
            print('    - {:<25s} : {}'.format('Test Data ', u_test.shape))
            print('    > {:<25s}'.format('Validate data set '))
            print('    - {:<25s} : {} {} '.format('u cand (mean, std)', u_cand_mean_std, u_cand_ref))
            print('    - {:<25s} : {} {} '.format('u test (mean, std)', u_test_mean_std, u_test_ref))
            print('    - {:<25s} : {} {} '.format('x test (mean, std)', x_test_mean_std, x_test_ref))

        ## ----------- Oversampling ratio ----------- ###
        simparams.update_num_samples(pce_model.num_basis, alphas=alphas)
        print(' > Oversampling ratio: {}'.format(np.around(simparams.alphas,2)))

        QoI_nsample     = []
        score_nsample   = []
        cv_err_nsample  = []
        test_err_nsample= []
        coef_err_nsample= []
        cond_num_nsample= []
        u_train = [None,] * repeats
        for i, n in enumerate(simparams.num_samples):
            ### ============ Initialize pce_model for each n ============
            pce_model= museuq.PCE(orth_poly)
            ### ============ Get training points ============
            u_cand_p = p ** 0.5 * u_cand if modeling.is_cls_unbounded() else u_cand
            # n = n - len(modeling.sample_selected)
            u_train = u_train[0] if repeats == 1 else u_train
            if simparams.optimality is not None:
                n1 = max(2,int(round(n/2)))
            else:
                n1 = n
            _, u_train = modeling.get_train_data((repeats,n1), u_cand_p, u_train=None, basis=pce_model.basis)
            # print(modeling.sample_selected)
            QoI_repeat     = []
            score_repeat   = []
            cv_err_repeat  = []
            test_err_repeat= []
            cond_num_repeat= []
            coef_err_repeat= []
            u_train  = [u_train,] if repeats == 1 else u_train
            for iu_train in tqdm(u_train, ascii=True, ncols=80,
                    desc='   [alpha={:.2f}, {:d}/{:d}, n={:d}]'.format(simparams.alphas[i], i+1, len(simparams.alphas),n)):

                ix_train = solver.map_domain(iu_train, pce_model.basis.dist_u)
                # assert np.array_equal(iu_train, ix_train)
                iy_train = solver.run(ix_train)
                if simparams.optimality is not None:
                    ### ============ Build Full PCE model ============
                    U_train = pce_model.basis.vandermonde(iu_train)[:, modeling.active_index] 
                    if simparams.doe_method.lower().startswith('cls'):
                        w_train = modeling.cal_cls_weight(iu_train, pce_model.basis)
                        U_train = modeling.rescale_data(U_train, w_train) 
                    else:
                        w_train = None
                        U_train = U_train

                    pce_model.fit(simparams.fit_method, iu_train, iy_train, w_train, n_splits=simparams.n_splits)
                    pce_model.estimate_sparsity_var(0.95)

                    ### update candidate data set for this p degree, cls unbuounded
                    u_train_new, _ = modeling.get_train_data(n-n1, u_cand_p, u_train=iu_train, basis=pce_model.basis, active_basis=pce_model.var_basis)
                    x_train_new = solver.map_domain(u_train_new, pce_model.basis.dist_u)
                    y_train_new = solver.run(x_train_new)
                    iu_train = np.hstack((iu_train, u_train_new)) 
                    ix_train = np.hstack((ix_train, x_train_new)) 
                    iy_train = np.hstack((iy_train, y_train_new)) 

                ### ============ Build Surrogate Model ============
                U_train = pce_model.basis.vandermonde(iu_train)[:, modeling.active_index]
                if simparams.doe_method.lower().startswith('cls'):
                    w_train = modeling.cal_cls_weight(iu_train, pce_model.basis)
                    U_train = modeling.rescale_data(U_train, w_train) 
                else:
                    w_train = None
                    U_train = U_train

                pce_model.fit('ols', iu_train, iy_train, w_train, n_splits=simparams.n_splits, active_basis=pce_model.var_basis)
                # pce_model.fit(simparams.fit_method, iu_train, y_train, w_train)
                # pce_model.fit_ols(iu_train, y_train, w_train)
                y_train_hat = pce_model.predict(iu_train)
                y_test_hat  = pce_model.predict(u_test)

                ## condition number, kappa = max(svd)/min(svd)
                _, sig_value, _ = np.linalg.svd(U_train)
                kappa = max(abs(sig_value)) / min(abs(sig_value)) 


                # QoI_repeat.append(np.linalg.norm(solver.coef- pce_model.coef, np.inf) < 1e-2)
                QoI_repeat.append(museuq.metrics.mquantiles(y_test_hat, 1-np.array(pf)))
                test_err_repeat.append(museuq.metrics.mean_squared_error(y_test, y_test_hat))
                cond_num_repeat.append(kappa)
                score_repeat.append(pce_model.score)
                cv_err_repeat.append(pce_model.cv_error)

            ### ============ calculating & updating metrics ============
            with np.printoptions(precision=4):
                print('     - {:<15s} : {}'.format( 'QoI'       , np.mean(QoI_repeat, axis=0)))
                print('     - {:<15s} : {:.4f}'.format( 'Test MSE ' , np.mean(test_err_repeat)))
                print('     - {:<15s} : {:.4f}'.format( 'CV MSE'    , np.mean(cv_err_repeat)))
                print('     - {:<15s} : {:.4f}'.format( 'Score '    , np.mean(score_repeat)))
                print('     - {:<15s} : {:.4f}'.format( 'kappa '    , np.mean(cond_num_repeat)))
                print('     ----------------------------------------')

            QoI_nsample.append(np.array(QoI_repeat))
            test_err_nsample.append(test_err_repeat)
            cv_err_nsample.append(cv_err_repeat)
            score_nsample.append(score_repeat)
            cond_num_nsample.append(cond_num_repeat)
        QoI_nsample      = np.moveaxis(np.array(QoI_nsample), -1, 0)
        QoI_nsample      = [iqoi for iqoi in QoI_nsample]
        score_nsample    = np.array(score_nsample)
        test_err_nsample = np.array(test_err_nsample)
        cond_num_nsample = np.array(cond_num_nsample)
        poly_deg = score_nsample/score_nsample*p
        nsamples = np.repeat(simparams.num_samples.reshape(-1,1), score_nsample.shape[1], axis=1)

        data_alpha = np.array([poly_deg, nsamples, *QoI_nsample, cond_num_nsample, score_nsample, test_err_nsample])
        data_alpha = np.moveaxis(data_alpha, 1, 0)
        data_poly_deg.append(data_alpha)
    filename = '{:s}_{:s}_{:s}'.format(solver.nickname, pce_model.tag, simparams.tag)
    try:
        np.save(os.path.join(simparams.data_dir_result, filename), np.array(data_poly_deg))
    except:
        np.save(os.path.join(os.getcwd(), filename), np.array(data_poly_deg))


if __name__ == '__main__':
    main()
