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
    np.set_printoptions(precision=8)
    np.set_printoptions(threshold=1000)
    np.set_printoptions(suppress=True)
    pf          = [1e-4, 1e-5, 1e-6]
    np.random.seed(100)
    ## ------------------------ Define solver ----------------------- ###
    solver      = museuq.ExpAbsSum(stats.uniform(-1,2),d=2,c=[-2,1],w=[0.25,-0.75])
    # solver      = museuq.ExpSquareSum(stats.uniform(-1,2),d=2,c=[1,1],w=[1,0.5])
    # solver      = museuq.CornerPeak(stats.uniform(-1,2), d=2)
    # solver      = museuq.ProductPeak(stats.uniform(-1,2), d=2,c=[-3,2],w=[0.5,0.5])
    # solver      = museuq.Franke()
    # solver      = museuq.Ishigami()

    # solver      = museuq.ExpAbsSum(stats.norm(0,1),d=2,c=[-2,1],w=[0.25,-0.75])
    # solver      = museuq.ExpSquareSum(stats.norm(0,1),d=2,c=[1,1],w=[1,0.5])
    # solver      = museuq.CornerPeak(stats.norm(0,1), d=3, c=np.array([1,2,3]), w=[0.5,]*3)
    # solver      = museuq.ProductPeak(stats.norm(0,1), d=2, c=[-3,2], w=[0.5,]*2)
    # solver      = museuq.ExpSum(stats.norm(0,1), d=3)
    # solver      = museuq.FourBranchSystem()
    ## ------------------------ Simulation Parameters ----------------- ###
    simparams = museuq.Parameters()
    simparams.solver     = solver
    simparams.pce_degs   = np.array([5])
    simparams.n_cand     = int(1e5)
    simparams.n_test     = -1
    simparams.doe_method = 'CLS' ### 'mcs', 'D', 'S', 'reference'
    simparams.optimality = None #'D', 'S', None
    # simparams.hem_type   = 'physicists'
    # simparams.hem_type   = 'probabilists'
    simparams.fit_method = 'LASSOLARS'
    simparams.n_splits   = 50
    repeats              = 50 if simparams.optimality is None else 1
    alphas               = [0.5] 
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
        orth_poly= museuq.Legendre(d=solver.ndim, deg=p)
        # orth_poly= museuq.Hermite(d=solver.ndim, deg=p, hem_type=simparams.hem_type)
        pce_model= museuq.PCE(orth_poly)
        pce_model.info()

        modeling = museuq.Modeling(solver, pce_model, simparams)
        # modeling.sample_selected=[]

        ## ----------- Candidate and testing data set for DoE ----------- ###
        print(' > Getting candidate data set...')
        u_cand = modeling.get_candidate_data()
        u_test, x_test, y_test = modeling.get_test_data(solver, pce_model) 
        print(museuq.metrics.mquantiles(y_test, 1-np.array(pf)))
        u_cand_p = p ** 0.5 * u_cand if modeling.is_cls_unbounded() else u_cand
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
        print(' > Starting with oversampling ratio {}'.format(np.around(simparams.alphas,2)))

        random_idx = np.random.randint(0, u_cand_p.shape[1], size=(repeats, simparams.num_samples[-1]))
        u_train = [u_cand_p[:, idx] for idx in random_idx] 
        x_train = [solver.map_domain(iu_train, pce_model.basis.dist_u) for iu_train in u_train]
        y_train = [solver.run(ix_train) for ix_train in x_train]
        nsamples= [[iu_train.shape[1],] for iu_train in u_train]
        # for i, n in enumerate(simparams.num_samples):
        QoI_repeat     = []
        score_repeat   = []
        cv_err_repeat  = []
        test_err_repeat= []
        cond_num_repeat= []
        coef_err_repeat= []
        for i in tqdm(range(repeats), ascii=True, ncols=80):
            tqdm.write('------------------------------ Resampling: {:d}/{:d} ------------------------------'.format(i, repeats))
            iu_train = u_train[i]
            ix_train = x_train[i]
            iy_train = y_train[i]
            QoI_nsample     = []
            score_nsample   = []
            cv_err_nsample  = []
            test_err_nsample= []
            coef_err_nsample= []
            cond_num_nsample= []
            while nsamples[i][-1] < math.ceil(1.1*pce_model.num_basis):
                ### ============ Build Surrogate Model ============
                if simparams.doe_method.lower().startswith('cls'):
                    w_train = modeling.cal_cls_weight(iu_train, pce_model.basis)
                else:
                    w_train = None
                tqdm.write('     [Sparsity: alpha={:.2f}, n={:d}]'.format(nsamples[i][-1]/pce_model.num_basis, nsamples[i][-1]))
                pce_model.fit('LASSOLARS', iu_train, iy_train, w_train, n_splits=simparams.n_splits)
                ### update candidate data set for this p degree, cls unbuounded
                # n = min(pce_model.sparsity, math.ceil(1.1*pce_model.num_basis) - iu_train.shape[1])
                # if n == 0:
                    # break
                pce_model.var(0.95)
                n = min(len(pce_model.var_pct_basis), math.ceil(1.1*pce_model.num_basis) - nsamples[i][-1])
                u_train_new, _ = modeling.get_train_data(n, u_cand_p, u_train=iu_train, basis=pce_model.basis, active_basis=pce_model.active_basis)
                # u_train_new, _ = modeling.get_train_data(n, u_cand_p, u_train=iu_train, basis=pce_model.basis)
                x_train_new = solver.map_domain(u_train_new, pce_model.basis.dist_u)
                y_train_new = solver.run(x_train_new)
                iu_train = np.hstack((iu_train, u_train_new)) 
                ix_train = np.hstack((ix_train, x_train_new)) 
                iy_train = np.hstack((iy_train, y_train_new)) 
                u_train[i] = iu_train
                x_train[i] = ix_train
                y_train[i] = iy_train
                nsamples[i].append(iu_train.shape[-1])

                U_train = pce_model.basis.vandermonde(iu_train)[:, modeling.active_index] 
                if simparams.doe_method.lower().startswith('cls'):
                    w_train = modeling.cal_cls_weight(iu_train, pce_model.basis)
                    U_train = modeling.rescale_data(U_train, w_train) 
                else:
                    w_train = None
                    U_train = U_train
                tqdm.write('     [Fit model: alpha={:.2f}, n={:d}]'.format(nsamples[i][-1]/pce_model.num_basis, nsamples[i][-1]))
                pce_model.fit('ols', iu_train, iy_train, w_train, n_splits=simparams.n_splits, active_basis=pce_model.active_basis)
                # pce_model.fit('ols', iu_train, iy_train, w_train, n_splits=simparams.n_splits)
                y_train_hat = pce_model.predict(iu_train)
                y_test_hat  = pce_model.predict(u_test)
                ## condition number, kappa = max(svd)/min(svd)
                _, sig_value, _ = np.linalg.svd(U_train)
                kappa = max(abs(sig_value)) / min(abs(sig_value)) 

                QoI_nsample.append(museuq.metrics.mquantiles(y_test_hat, 1-np.array(pf)))
                test_err_nsample.append(museuq.metrics.mean_squared_error(y_test, y_test_hat))
                cond_num_nsample.append(kappa)
                score_nsample.append(pce_model.score)
                cv_err_nsample.append(pce_model.cv_error)

                ### ============ calculating & updating metrics ============
                with np.printoptions(precision=4):
                    tqdm.write('     - {:<15s} : {}'.format( 'QoI'       , QoI_nsample[-1]))
                    tqdm.write('     - {:<15s} : {:.4f}'.format( 'Test MSE ' , test_err_nsample[-1]))
                    tqdm.write('     - {:<15s} : {:.4f}'.format( 'CV MSE'    , cv_err_nsample[-1]))
                    tqdm.write('     - {:<15s} : {:.4f}'.format( 'Score '    , score_nsample[-1]))
                    tqdm.write('     - {:<15s} : {:.4f}'.format( 'kappa '    , cond_num_nsample[-1]))
                    tqdm.write('     ----------------------------------------')

            # QoI_repeat.append(np.linalg.norm(solver.coef- pce_model.coef, np.inf) < 1e-2)
            QoI_repeat.append(np.array(QoI_nsample).T)
            test_err_repeat.append(test_err_nsample)
            cond_num_repeat.append(cond_num_nsample)
            score_repeat.append(score_nsample)
            cv_err_repeat.append(cv_err_nsample)

            # print(QoI_repeat.shape)
            # print(test_err_repeat.shape)
            # print(cond_num_repeat.shape)
            # print(score_repeat.shape)
            # print(cv_err_repeat.shape)
        # QoI_nsample      = np.moveaxis(np.array(QoI_nsample), -1, 0)
        # QoI_nsample      = [iqoi for iqoi in QoI_nsample]
        # score_nsample    = np.array(score_nsample)
        # test_err_nsample = np.array(test_err_nsample)
        # cond_num_nsample = np.array(cond_num_nsample)
        # poly_deg         = score_nsample/score_nsample*p
        # nsamples         = np.array(nsamples) 
        # data_alpha       = np.array([poly_deg, nsamples, *QoI_nsample, cond_num_nsample, score_nsample, test_err_nsample])
        # data_alpha       = np.moveaxis(data_alpha, 1, 0)


        QoI = np.moveaxis(np.array(QoI_repeat), 0, -1)
        QoI = [iqoi for iqoi in QoI]
        score    = np.array(score_repeat).T
        test_err = np.array(test_err_repeat).T
        cond_num = np.array(cond_num_repeat).T
        poly_deg = score/score*p
        nsamples = np.array(nsamples).T
        data_alpha = np.array([poly_deg, nsamples, *QoI, cond_num, score, test_err])
        # data_alpha       = np.moveaxis(data_alpha, 1, 0)

        data_poly_deg.append(data_alpha)
    filename = '{:s}_{:s}_{:s}_pct{:d}'.format(
            solver.nickname, pce_model.tag, simparams.tag, int(simparams.alphas[0]*100))
    try:
        np.save(os.path.join(simparams.data_dir_result, filename), np.array(data_poly_deg))
    except:
        np.save(os.path.join(os.getcwd(), filename), np.array(data_poly_deg))


if __name__ == '__main__':
    main()
