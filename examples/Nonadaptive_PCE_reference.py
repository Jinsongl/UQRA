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

    # solver      = museuq.ExpSquareSum(stats.uniform(-1,2),d=2,c=[1,1],w=[1,0.5])
    solver      = museuq.ExpAbsSum(stats.uniform(-1,2),d=2,c=[-2,1],w=[0.25,-0.75])
    # solver      = museuq.CornerPeak(stats.uniform(-1,2), d=2)
    # solver      = museuq.ProductPeak(stats.uniform(-1,2), d=2,c=[-3,2],w=[0.5,0.5])
    # solver      = museuq.Franke()

    # solver      = museuq.ExpSquareSum(stats.norm(0,1),d=2,c=[1,1],w=[1,0.5])
    # solver      = museuq.ExpAbsSum(stats.norm(0,1),d=2,c=[-2,1],w=[0.25,-0.75])
    # solver      = museuq.CornerPeak(stats.norm(0,1), d=3, c=np.array([1,2,3]), w=[0.5,]*3)
    # solver      = museuq.ProductPeak(stats.norm(0,1), d=3, c=1.0/np.array([1,2,3])**2, w=[0.5,]*3)
    # solver      = museuq.ExpSum(stats.norm(0,1), d=3)
    # solver      = museuq.FourBranchSystem()
    ## ------------------------ Simulation Parameters ----------------- ###
    simparams = museuq.Parameters(solver)
    simparams.pce_degs   = np.array(range(13, 16))
    simparams.n_cand     = int(1e5)
    simparams.n_test     = -1
    simparams.doe_method = 'CLS' ### 'mcs', 'D', 'S', 'reference'
    simparams.optimality = None #'D', 'S', None
    # simparams.hem_type   = 'physicists'
    # simparams.hem_type   = 'probabilists'
    simparams.fit_method = 'OLS'
    simparams.n_splits   = 5
    repeat               = 10
    simparams.update()
    simparams.info()


    ## ----------- Oversampling ratio ----------- ###
    # alpha =[]
    # alpha.append([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    # alpha.append([1.1,2.0])
    # alpha.append(np.linspace(1, 2, 6))
    # alpha.append(np.linspace(2, 4, 6))
    # alpha.append(np.linspace(4,10,11))
    # alpha.append(np.linspace(3, 4, 5))
    # alpha.append(np.linspace(4,10, 7))
    # alphas = np.unique(np.hstack(alpha))



    ### ============ Initial Values ============
    print(' > Starting simulation...')
    data_p = []
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

        nsamples = []
        # nsamples.append(np.arange(pce_model.num_basis+1, 500, 20))
        nsamples.append(2*math.ceil(np.log(pce_model.num_basis) * pce_model.num_basis))
        nsamples = np.unique(np.hstack(nsamples))

        modeling = museuq.Modeling(solver, pce_model, simparams)
        modeling.sample_selected=[]

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

        try:
            nsamples = np.array([math.ceil(pce_model.num_basis*ialpha) for ialpha in alphas])
            alphas = nsamples/pce_model.num_basis
        except NameError:
            try:
                alphas = nsamples/pce_model.num_basis
                nsamples = np.array([math.ceil(pce_model.num_basis*ialpha) for ialpha in alphas])
            except NameError:
                raise ValueError('Either alphas or nsamples should be defined')
        print(' > Oversampling ratio: {}'.format(np.around(alphas,2)))

        QoI     = []
        score   = []
        cv_err  = []
        test_err= []
        coef_err= []
        cond_num= []
        test_l2 = []
        u_train = [None,] * repeat
        ### ============ Start adaptive iteration ============
        for i, n in enumerate(nsamples):
            ### ============ Get training points ============
            u_cand_p = p ** 0.5 * u_cand if modeling.is_cls_unbounded() else u_cand
            # print(u_cand_p[:,:3])
            # print(u_cand[:,:3])
            n = n - len(modeling.sample_selected)
            u_train = u_train[0] if repeat == 1 else u_train
            _, u_train = modeling.get_train_data((repeat,n), u_cand_p, u_train=u_train, basis=pce_model.basis)
            # print(modeling.sample_selected)
            QoI_     = []
            score_   = []
            cv_err_  = []
            test_err_= []
            cond_num_= []
            coef_err_= []
            test_l2_ = []
            u_train = [u_train,] if repeat == 1 else u_train
            for iu_train in tqdm(u_train, ascii=True, desc='   [alpha={:.2f}, {:d}/{:d}, n={:d}]'.format(alphas[i], i+1, len(alphas),n)):
                ix_train = solver.map_domain(iu_train, pce_model.basis.dist_u)
                # assert np.array_equal(iu_train, ix_train)
                y_train = solver.run(ix_train)

                ### ============ Build Surrogate Model ============
                U_train = pce_model.basis.vandermonde(iu_train) 
                if simparams.doe_method.lower().startswith('cls'):
                    w_train = modeling.cal_cls_weight(iu_train, pce_model.basis)
                    U_train = modeling.rescale_data(U_train, w_train) 
                else:
                    w_train = None
                    U_train = U_train

                pce_model.fit(simparams.fit_method, iu_train, y_train, w_train, n_splits=min(simparams.n_splits, iu_train.shape[1]))
                # pce_model.fit_ols(iu_train, y_train, w_train)
                y_train_hat = pce_model.predict(iu_train)
                y_test_hat  = pce_model.predict(u_test)


                ## condition number, kappa = max(svd)/min(svd)
                _, s, _ = np.linalg.svd(U_train)
                kappa = max(abs(s)) / min(abs(s)) 


                # coef_err_.append(np.linalg.norm(solver.coef- pce_model.coef, np.inf))
                # QoI_.append(np.linalg.norm(solver.coef- pce_model.coef, np.inf) < 1e-2)
                QoI_.append(museuq.metrics.mquantiles(y_test_hat, 1-np.array(pf)))
                test_err_.append(museuq.metrics.mean_squared_error(y_test, y_test_hat))
                test_l2_.append(np.linalg.norm(y_test-y_test_hat,2))
                cond_num_.append(kappa)
                score_.append(pce_model.score)
                cv_err_.append(pce_model.cv_error)

            ### ============ calculating & updating metrics ============
            
            # coef_err.append(np.mean(coef_err_))
            QoI.append(np.mean(QoI_, axis=0))
            test_err.append(np.mean(test_err_))
            cv_err.append(np.mean(cv_err_))
            score.append(np.mean(score_))
            cond_num.append(np.mean(np.array(cond_num_)))
            test_l2.append(np.mean(test_l2_))


            np.set_printoptions(precision=4)
            print('     - {:<15s} : {}'.format( 'QoI', np.array(QoI)))
            # print('     - {:<15s} : {}'.format( 'Coef Error', np.array(coef_err)))
            print('     - {:<15s} : {}'.format( 'Test MSE ', np.array(test_err)))
            print('     - {:<15s} : {}'.format( 'Test L2 ', np.array(test_l2)))
            print('     - {:<15s} : {}'.format( 'CV MSE ', np.array(cv_err)))
            print('     - {:<15s} : {}'.format( 'Score ', np.array(score)))
            print('     - {:<15s} : {}'.format( 'Condition Num:', np.array(cond_num)))

        QoI      = np.array(QoI).T
        score    = np.array(score)
        nsamples = np.array(nsamples)
        test_err = np.array(test_err)
        cond_num = np.array(cond_num)
        # coef_err = np.array(coef_err)

        # data_alpha = np.vstack((nsamples,  QoI, cond_num, score, test_err, coef_err))
        data_alpha = np.vstack((np.ones(nsamples.shape)*p, nsamples,  QoI, cond_num, score, test_err)).T
        # data_alpha = np.vstack((nsamples,  QoI, cond_num, score, coef_err))
        # data_alpha = np.vstack((nsamples,  cond_num))

        # filename = 'Fix_{:s}_{:s}_reference'.format(solver.nickname, simparams.get_tag())
        # np.save(os.path.join(simparams.data_dir_result, filename), data_alpha)
        data_p.append(data_alpha)
    filename = '{:s}_{:s}_{:s}_reference'.format(solver.nickname, pce_model.tag, simparams.tag)
    np.save(os.path.join(simparams.data_dir_result, filename), np.array(data_p))

if __name__ == '__main__':
    main()
