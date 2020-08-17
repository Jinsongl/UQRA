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
from tqdm import tqdm
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()

def get_basis(deg, simparams, solver):
    if simparams.doe_method.lower().startswith('mcs'):
        if simparams.poly_type.lower() == 'leg':
            print(' Legendre polynomial')
            basis = uqra.Legendre(d=solver.ndim, deg=deg)

        elif simparams.poly_type.lower().startswith('hem'):
            print(' Probabilists Hermite polynomial')
            basis = uqra.Hermite(d=solver.ndim,deg=deg, hem_type='probabilists')
        else:
            raise ValueError 

    elif simparams.doe_method.lower().startswith('cls'):
        if simparams.poly_type.lower() == 'leg':
            print(' Legendre polynomial')
            basis = uqra.Legendre(d=solver.ndim,deg=deg)

        elif simparams.poly_type.lower().startswith('hem'):
            print(' Probabilists Hermite polynomial')
            basis = uqra.Hermite(d=solver.ndim,deg=deg, hem_type='physicists')
        else:
            raise ValueError
    else:
        raise ValueError

    return basis 

def main():

    ## ------------------------ Displaying set up ------------------- ###
    np.set_printoptions(precision=4)
    np.set_printoptions(threshold=8)
    np.set_printoptions(suppress=True)
    pf          = [1e-4, 1e-5, 1e-6]
    np.random.seed(100)
    ## ------------------------ Define solver ----------------------- ###
    # solver      = uqra.ExpAbsSum(stats.uniform(-1,2),d=2,c=[-2,1],w=[0.25,-0.75])
    # solver      = uqra.ExpSquareSum(stats.uniform(-1,2),d=2,c=[1,1],w=[1,0.5])
    # solver      = uqra.CornerPeak(stats.uniform(-1,2), d=2)
    # solver      = uqra.ProductPeak(stats.uniform(-1,2), d=2,c=[-3,2],w=[0.5,0.5])
    solver      = uqra.Franke()
    # solver      = uqra.Ishigami()

    # solver      = uqra.ExpAbsSum(stats.norm(0,1),d=2,c=[-2,1],w=[0.25,-0.75])
    # solver      = uqra.ExpSquareSum(stats.norm(0,1),d=2,c=[1,1],w=[1,0.5])
    # solver      = uqra.CornerPeak(stats.norm(0,1), d=3, c=np.array([1,2,3]), w=[0.5,]*3)
    # solver      = uqra.ProductPeak(stats.norm(0,1), d=2, c=[-3,2], w=[0.5,]*2)
    # solver      = uqra.ExpSum(stats.norm(0,1), d=3)
    # solver      = uqra.FourBranchSystem()


    ## ------------------------ Simulation Parameters ----------------- ###
    simparams = uqra.Parameters()
    simparams.solver     = solver
    simparams.pce_degs   = np.array(range(2,21))
    simparams.n_cand     = int(1e5)
    simparams.n_test     = int(2e5)
    simparams.doe_method = 'MCS' ### 'mcs', 'D', 'S', 'reference'
    simparams.optimality = 'S'#'D', 'S', None
    simparams.fit_method = 'OLS'
    simparams.poly_type  = 'leg'
    simparams.n_splits   = 50
    repeats              = 50 if simparams.optimality is None else 1
    alphas               = [1.2]
    simparams.update()
    simparams.info()

    ## ----------- Test data set ----------- ###
    ## ----- Testing data set centered around u_center, first 100000
    print(' > Getting Test data set...')
    filename    = 'Franke_2Leg_DoE_McsE6R0.npy'
    data_test   = np.load(os.path.join(simparams.data_dir_result,'TestData', filename))
    u_test      = data_test[            :  solver.ndim, :simparams.n_test]
    x_test      = data_test[solver.ndim :2*solver.ndim, :simparams.n_test]
    y_test      = data_test[-1, :simparams.n_test]

    print(' > Test data: {:s}'.format(filename))
    print('   - {:<25s} : {}, {}, {}'.format('Test Dataset (U,X,Y)',u_test.shape, x_test.shape, y_test.shape ))
    print('   - {:<25s} : [{}, {}]'.format('Test U[mean, std]',np.mean(u_test, axis=1),np.std (u_test, axis=1)))
    print('   - {:<25s} : [{}]'.format('Test max(U)[U1, U2]',np.amax(abs(u_test), axis=1)))
    print('   - {:<25s} : [{}]'.format('Test [min(Y), max(Y)]',np.array([np.amin(y_test),np.amax(y_test)])))

    ## ----------- Predict data set ----------- ###
    ## ----- Prediction data set centered around u_center, all  
    # filename= 'DoE_McsE7R0.npy'
    # mcs_data= np.load(os.path.join(simparams.data_dir_sample,'MCS', 'Uniform', filename))
    # u_pred  = mcs_data[:solver.ndim, :simparams.n_pred] 


    ### ============ Initial Values ============
    print(' > Starting simulation...')
    metrics_each_deg = []
    data_test = []
    for deg in simparams.pce_degs:
        print('\n================================================================================')
        simparams.info()
        print('   - Sampling and Fitting:')
        print('     - {:<23s} : {}'.format('Sampling method'  , simparams.doe_method))
        print('     - {:<23s} : {}'.format('Optimality '      , simparams.optimality))
        print('     - {:<23s} : {}'.format('Fitting method'   , simparams.fit_method))
        ## ----------- Define PCE  ----------- ###
        basis     = get_basis(deg, simparams, solver)
        pce_model = uqra.PCE(basis)
        modeling  = uqra.Modeling(solver, pce_model, simparams)
        pce_model.info()

        ## ----------- Candidate and testing data set for DoE ----------- ###
        if simparams.doe_method.lower().startswith('cls2'):
            filename = os.path.join(simparams.data_dir_sample, 'CLS', 'DoE_Cls2E7d2R0.npy')
            u_cand = np.load(filename)[:solver.ndim, :simparams.n_cand]
            u_cand = u_cand * radius_surrogate

        elif simparams.doe_method.lower().startswith('cls4'):
            filename = os.path.join(simparams.data_dir_sample, 'CLS', 'DoE_Cls4E7d2R0.npy')
            u_cand = np.load(filename)[:solver.ndim, :simparams.n_cand]

        elif simparams.doe_method.lower().startswith('mcs'):
            filename = os.path.join(simparams.data_dir_sample, 'MCS',pce_model.basis.dist_name,'DoE_McsE7R0.npy')
            u_cand = np.load(filename)[:solver.ndim, :simparams.n_cand]

        u_cand_p = deg ** 0.5 * u_cand if simparams.doe_method.lower() in ['cls4', 'cls5'] else u_cand
        print(' > Candidate data: {:s}'.format(filename))
        print('   - shape: {}'.format(u_cand_p.shape))

        ## ----------- Oversampling ratio ----------- ###
        simparams.update_num_samples(pce_model.num_basis, alphas=alphas)
        print(' > Oversampling ratio: {}'.format(np.around(simparams.alphas,2)))
        for i, n in enumerate(simparams.num_samples):
            ### ============ Initialize pce_model for each n ============
            pce_model= uqra.PCE(basis)
            ### ============ Get training points ============
            _, u_train = modeling.get_train_data((repeats,n), u_cand_p, u_train=None, basis=pce_model.basis)
            # doe = uqra.LHS(pce_model.basis.dist_u)
            # u_train = np.array([doe.samples(size=n, random_state=None) for _ in range(50)])
            # print(u_train.shape)
            # print(modeling.sample_selected)
            data_repeat = []
            for iu_train in tqdm(u_train, ascii=True, ncols=80,
                    desc='   [alpha={:.2f}, {:d}/{:d}, n={:d}]'.format(simparams.alphas[i], i+1, len(simparams.alphas),n)):

                ix_train = solver.map_domain(iu_train, pce_model.basis.dist_u)
                # assert np.array_equal(iu_train, ix_train)
                iy_train = solver.run(ix_train)
                ### Full model checking
                assert len(pce_model.active_index) == pce_model.num_basis
                ### ============ Build Surrogate Model ============
                U_train = pce_model.basis.vandermonde(iu_train)
                if simparams.doe_method.lower().startswith('cls'):
                    w_train = modeling.cal_cls_weight(iu_train, pce_model.basis, pce_model.active_index)
                    U_train = modeling.rescale_data(U_train, w_train) 
                else:
                    w_train = None
                    U_train = U_train[:, pce_model.active_index]

                pce_model.fit(simparams.fit_method, iu_train, iy_train, w_train, n_splits=simparams.n_splits)
                # pce_model.fit(simparams.fit_method, iu_train, y_train, w_train)
                y_train_hat = pce_model.predict(iu_train)
                y_test_hat  = pce_model.predict(u_test)

                ## condition number, kappa = max(svd)/min(svd)
                _, sig_value, _ = np.linalg.svd(U_train)
                kappa = max(abs(sig_value)) / min(abs(sig_value)) 

                
                test_mse = uqra.metrics.mean_squared_error(y_test, y_test_hat)
                # QoI   = uqra.metrics.mquantiles(y_test_hat, 1-np.array(pf))
                data_ = np.array([deg, n, kappa, pce_model.score, pce_model.cv_error, test_mse])
                # data_ = np.append(data_, QoI)
                metrics_each_deg.append(data_)

                data_test.append([deg, n, y_test_hat])
                ### ============ calculating & updating metrics ============
                tqdm.write(' > Summary')
                with np.printoptions(precision=4):
                    # tqdm.write('     - {:<15s} : {}'.format( 'QoI'       , QoI))
                    tqdm.write('     - {:<15s} : {:.4f}'.format( 'Test MSE ' , test_mse))
                    tqdm.write('     - {:<15s} : {:.4f}'.format( 'CV MSE'    , pce_model.cv_error))
                    tqdm.write('     - {:<15s} : {:.4f}'.format( 'Score '    , pce_model.score))
                    tqdm.write('     - {:<15s} : {:.4f}'.format( 'kappa '    , kappa))
                    tqdm.write('     ----------------------------------------')

    ### ============ Saving QoIs ============
    metrics_each_deg = np.array(metrics_each_deg)
    with open(os.path.join(simparams.data_dir_result, 'outlist_name.txt'), "w") as text_file:
        text_file.write('\n'.join(['deg', 'n', 'kappa', 'score','cv_error', 'test_mse' ]))

    filename = '{:s}_{:s}_{:s}'.format(solver.nickname, pce_model.tag, simparams.tag)
    try:
        np.save(os.path.join(simparams.data_dir_result, filename), np.array(metrics_each_deg))
    except:
        np.save(os.path.join(os.getcwd(), filename), np.array(metrics_each_deg))


    ### ============ Saving test ============
    data_test = np.array(data_test, dtype=object)

    filename = '{:s}_{:s}_{:s}_test'.format(solver.nickname, pce_model.tag, simparams.tag)
    try:
        np.save(os.path.join(simparams.data_dir_result, filename), data_test)
    except:
        np.save(os.path.join(os.getcwd(), filename), data_test)


if __name__ == '__main__':
    main()
