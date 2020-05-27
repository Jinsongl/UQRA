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
from tqdm import tqdm
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()

def main():

    ## ------------------------ Displaying set up ------------------- ###
    np.set_printoptions(precision=4)
    np.set_printoptions(threshold=1000)
    np.set_printoptions(suppress=True)
    pf = [1e-4, 1e-5, 1e-6]
    test_filename = r'DoE_McsE6R8.npy'
    ## ------------------------ Define solver ----------------------- ###
    random_seed = 100
    out_responses = [2]
    out_stats = ['absmax']
    n_short_term = 1
    m=1
    c=0.1/np.pi
    k=1.0/np.pi/np.pi
    m,c,k  = [stats.norm(m, 0.05*m), stats.norm(c, 0.2*c), stats.norm(k, 0.1*k)]
    # env    = museuq.Environment([stats.uniform, stats.norm])
    # env    = museuq.environment.Kvitebjorn.Kvitebjorn()
    env    = museuq.Environment([2,])
    solver = museuq.linear_oscillator(m=m,c=c,k=k,excitation='spec_test1', environment=env,
            t=1000,t_transit=10, dt=0.1, out_responses=out_responses, out_stats=out_stats, phase=range(n_short_term))
    ## ------------------------ Simulation Parameters ----------------- ###
    simparams = museuq.Parameters()
    simparams.solver     = solver
    simparams.pce_degs   = np.array(range(10,11))
    simparams.n_cand     = int(1e5)
    simparams.n_test     = -1
    simparams.doe_method = 'CLS' ### 'mcs', 'D', 'S', 'reference'
    simparams.optimality = 'S'#'D', 'S', None
    simparams.hem_type   = 'physicists'
    # simparams.hem_type   = 'probabilists'
    simparams.fit_method = 'OLS'
    simparams.n_splits   = 50
    repeats              = 1 if simparams.optimality is None else 1
    # alphas               = np.arange(3,11)/10 
    alphas               = [1.2]
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
        u_test, x_test, y_test = modeling.get_test_data(solver, pce_model, filename=test_filename, qoi=out_responses,n=1e6) 
        y_test_mean = np.mean(y_test, axis=0)
        y_test_std  = np.std(y_test, axis=0)
        print(museuq.metrics.mquantiles(y_test.T, 1-np.array(pf), multioutput='raw_values'))
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
        print(' > Oversampling ratio: {}'.format(np.around(simparams.alphas,2)))
        data_nsample = []
        for i, n in enumerate(simparams.num_samples):
            ### ============ Initialize pce_model for each n ============
            pce_model= museuq.PCE(orth_poly)
            ### ============ Get training points ============
            _, u_train = modeling.get_train_data((repeats,n), u_cand_p, u_train=None, basis=pce_model.basis)
            # print(modeling.sample_selected)
            data_repeat = []
            for iu_train in tqdm(u_train, ascii=True, ncols=80,
                    desc='   [alpha={:.2f}, {:d}/{:d}, n={:d}]'.format(simparams.alphas[i], i+1, len(simparams.alphas),n)):

                ix_train = solver.map_domain(iu_train, pce_model.basis.dist_u)
                # assert np.array_equal(iu_train, ix_train)
                iy_train = solver.run(ix_train)
                iy_train_mean = np.mean(iy_train, axis=0)
                iy_train_std  = np.std(iy_train, axis=0)
                ### Full model checking
                assert len(pce_model.active_index) == pce_model.num_basis
                ### ============ Build Surrogate Model ============
                ### surrogate model for each short term simulation
                U_train = pce_model.basis.vandermonde(iu_train)
                if simparams.doe_method.lower().startswith('cls'):
                    w_train = modeling.cal_cls_weight(iu_train, pce_model.basis, pce_model.active_index)
                    U_train = modeling.rescale_data(U_train, w_train) 
                else:
                    w_train = None
                    U_train = U_train[:, pce_model.active_index]
                ## condition number, kappa = max(svd)/min(svd)
                _, sig_value, _ = np.linalg.svd(U_train)
                kappa = max(abs(sig_value)) / min(abs(sig_value)) 

                pce_model.fit(simparams.fit_method,iu_train,iy_train.T,w_train,n_splits=simparams.n_splits)
                y_test_hat  = pce_model.predict(u_test)
                y_pred = np.vstack((u_test, x_test, y_test_hat.reshape(1,-1)))
                print(u_test.shape)
                print(x_test.shape)
                print(y_test_hat.shape)
                np.save(os.path.join(simparams.data_dir_result,'TestData', modeling.filename_test[:-4]+'_y2_pred.npy'), y_pred)
                
                # pce_model.fit(simparams.fit_method,iu_train,iy_train_mean,w_train,n_splits=simparams.n_splits)
                # y_test_mean_hat  = pce_model.predict(u_test)

                # pce_model.fit(simparams.fit_method,iu_train,iy_train_std,w_train,n_splits=simparams.n_splits)
                # y_test_std_hat  = pce_model.predict(u_test)

                # # y_pred = np.vstack((y_test_hat.reshape(-1,n_test), y_test_mean_hat.reshape(-1,n_test), y_test_std.reshape(-1,n_test)))

                # test_mse = museuq.metrics.mean_squared_error(y_test.T, y_test_hat,multioutput='raw_values')
                # test_mean_mse = museuq.metrics.mean_squared_error(y_test_mean, y_test_mean_hat)
                # test_std_mse = museuq.metrics.mean_squared_error(y_test_std, y_test_std_hat)
                # excd_vals = museuq.metrics.mquantiles(y_test_hat, 1-np.array(pf), multioutput='raw_values').reshape(len(pf), -1)
                # excd_mean_val = museuq.metrics.mquantiles(y_test_mean_hat, 1-np.array(pf), multioutput='raw_values')
                # data = np.array([p, n, kappa, pce_model.score, pce_model.cv_error])
                # data = np.append(data, test_mse)
                # data = np.append(data, test_mean_mse)
                # data = np.append(data, test_std_mse)
                # for iexcd_val, iexcd_mean_val in zip(excd_vals, excd_mean_val):
                    # data = np.append(data, iexcd_val)
                    # data = np.append(data, iexcd_mean_val)
                # data_poly_deg.append(data)

                # outlist = ['polydegree', 'nsamples', 'kappa', 'score', 'cv_error']
                # for i in range(len(test_mse)):
                    # outlist.append('y_nst{:d}'.format(i))

                # outlist.append('y_mean')
                # outlist.append('y_std')
                # npf, nst = excd_vals.shape
                # for ipf in [4,5,6]:
                    # for i in range(nst):
                        # outlist.append('y_pf{:d}_nst{:d}'.format(ipf, i))
                        # if i == nst-1:
                            # outlist.append('y_mean_pf{:d}'.format(ipf))
                # outlist = [': '.join([str(str1),str2]) for str1, str2 in zip(range(np.size(data)), outlist)]

                # ### ============ calculating & updating metrics ============
                # tqdm.write(' > Summary')
                # with np.printoptions(precision=4):
                    # tqdm.write('     - {:<15s} : {}'.format( 'excd_vals'       , excd_mean_val))
                    # tqdm.write('     - {:<15s} : {:.4f}'.format( 'Test MSE ' , test_mean_mse))
                    # tqdm.write('     - {:<15s} : {:.4f}'.format( 'CV MSE'    , pce_model.cv_error))
                    # tqdm.write('     - {:<15s} : {:.4f}'.format( 'Score '    , pce_model.score))
                    # tqdm.write('     - {:<15s} : {:.4f}'.format( 'kappa '    , kappa))
                    # tqdm.write('     ----------------------------------------')

    # filename = '{:s}_{:s}_{:s}'.format(solver.nickname, pce_model.tag, simparams.tag)
    # try:
        # # np.save(os.path.join(simparams.data_dir_result, filename), np.array(data_poly_deg))
        # # np.savetxt(os.path.join(simparams.data_dir_result, 'outlist.txt'), outlist, delimiter=' ', fmt='%s') 
    # except:
        # # np.save(os.path.join(os.getcwd(), filename), np.array(data_poly_deg))
        # # np.savetxt(os.path.join(simparams.data_dir_result, 'outlist.txt'), outlist, delimiter=' ', fmt='%s') 
        # np.save(os.path.join(os.getcwd(), filename+'_pred'), y_test_hat)


if __name__ == '__main__':
    main()
