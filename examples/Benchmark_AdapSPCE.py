#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import uqra
import numpy as np, os, sys
import scipy.stats as stats
from tqdm import tqdm
# warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()

class Data():
    pass

def main():

    ## ------------------------ Displaying set up ------------------- ###
    print('\n#################################################################################')
    print(' >>>  Start UQRA : {:s}'.format(__file__, theta))
    print('#################################################################################\n')
    np.random.seed(100)
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
    # solver      = uqra.Franke()
    solver      = uqra.Ishigami()

    # solver      = uqra.ExpAbsSum(stats.norm(0,1),d=2,c=[-2,1],w=[0.25,-0.75])
    # solver      = uqra.ExpSquareSum(stats.norm(0,1),d=2,c=[1,1],w=[1,0.5])
    # solver      = uqra.CornerPeak(stats.norm(0,1), d=3, c=np.array([1,2,3]), w=[0.5,]*3)
    # solver      = uqra.ProductPeak(stats.norm(0,1), d=2, c=[-3,2], w=[0.5,]*2)
    # solver      = uqra.ExpSum(stats.norm(0,1), d=3)
    # solver      = uqra.FourBranchSystem()


    ## ------------------------ Simulation Parameters ----------------- ###
    simparams = uqra.Parameters(solver, doe_method=['CLS4', 'D'], fit_method='LASSOLARS')
    simparams.set_udist('norm')
    simparams.x_dist     = solver.distributions 
    simparams.pce_degs   = np.array(range(2,11))
    simparams.n_cand     = int(1e5)
    simparams.n_test     = int(2e5)
    simparams.n_pred     = int(1e7)
    simparams.n_splits   = 50
    simparams.alphas     = 2
    n_initial = 20
    simparams.info()

    ## ----------- Define U distributions ----------- ###
    sampling_domain = -simparams.u_dist[0].ppf(1e-7)
    sampling_space  = 'u'
    dist_support    = None        ## always in x space

    # sampling_domain = [[-1,1],] * solver.ndim 
    # sampling_space  = 'u'
    # dist_support    = np.array([[0, 18],[0,35]]) ## always in x space

    ## ----------- Predict data set ----------- ###
    print(' > Getting predict data set...')
    filename = '{:s}_Cls4E7R{:d}_pred.npy'.format(solver.nickname, theta)
    data = np.load(os.path.join(simparams.data_dir_result, 'TestData', filename))
    idx_inside, idx_outside = uqra.samples_within_ellipsoid(data[:simparams.ndim], c=0,radii=sampling_domain)
    data_pred = data[:, idx_inside ]
    data_pred_outside = data[:, idx_outside]
    u_pred = data_pred[:simparams.ndim]
    x_pred = data_pred[simparams.ndim:2*simparams.ndim]
    u_pred_outside = data_pred_outside[:simparams.ndim]
    x_pred_outside = data_pred_outside[simparams.ndim:2*simparams.ndim]
    print('   - {:<25s} : {}'.format(' Sample domain', sampling_domain))
    print('   - {:<25s} : {}'.format(' dist Support ', dist_support if dist_support is None else dist_support.reshape(1,-1)))
    print('   - {:<25s} '.format(' Samples inside domain...'))
    print('   - {:<25s} : {}, {}'.format(' U,X', u_pred.shape, x_pred.shape))
    print('   - {:<25s} : {}, {}'.format(' U support', np.amin(u_pred, axis=1), np.amax(u_pred, axis=1)))
    print('   - {:<25s} : {}, {}'.format(' X support', np.amin(x_pred, axis=1), np.amax(x_pred, axis=1)))
    print('   - {:<25s} '.format(' Samples outside domain...'))
    print('   - {:<25s} : {}'.format('X ', x_pred_outside.shape))

    y_pred_outside = np.array(solver.run(x_pred_outside), ndmin=1)
    print('   - {:<25s} : {}, {}'.format('(X,Y) ', x_pred_outside.shape, y_pred_outside.shape))

    ## ----------- Test data set ----------- ###
    ## ----- Testing data set centered around u_center, first 100000
    print(' > Getting Test data set...')
    # filename    = 'Franke_2Leg_DoE_McsE6R0.npy'
    filename    = 'Ishigami_3Leg_DoE_McsE6R0.npy'
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
    # data_test = []
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
        cand_fname, u_cand = get_candidate_data(simparams, solver, pce_model)
        print(' > Candidate data: {:s}'.format(cand_fname))
        print('   - shape: {}'.format(u_cand.shape if u_cand is not None else 'None'))

        ## ----------- Oversampling ratio ----------- ###
        # simparams.update_num_samples(pce_model.num_basis, alphas=alphas)
        simparams.update_num_samples(pce_model.num_basis, num_samples=num_samples)
        print(' > Oversampling ratio: {}'.format(np.around(simparams.alphas,2)))
        for i, n in enumerate(simparams.num_samples):
            ### ============ Initialize pce_model for each n ============
            pce_model= uqra.PCE(basis)
            ### ============ Get training points ============
            size = (repeats, n)
            u_train = get_train_data(simparams, modeling, size, u_cand=u_cand, u_train=None, basis=pce_model.basis)

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

                # data_test.append([deg, n, y_test_hat])
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

    # filename = '{:s}_{:s}_{:s}_Alpha{:s}'.format(solver.nickname, pce_model.tag, simparams.tag,
            # str(alphas).replace('.', 'pt'))
    filename = '{:s}_{:s}_{:s}_NumSamples'.format(solver.nickname, pce_model.tag, simparams.tag)
    # filename = '{:s}_{:s}_{:s}_reference'.format(solver.nickname, pce_model.tag, simparams.tag)
    try:
        np.save(os.path.join(simparams.data_dir_result, filename), np.array(metrics_each_deg))
    except:
        np.save(os.path.join(os.getcwd(), filename), np.array(metrics_each_deg))


    ### ============ Saving test ============
    # data_test = np.array(data_test, dtype=object)

    # filename = '{:s}_{:s}_{:s}_Alpha{:s}_test'.format(solver.nickname, pce_model.tag, simparams.tag,
            # str(alphas).replace('.', 'pt'))
    # # filename = '{:s}_{:s}_{:s}__NumSamples_test'.format(solver.nickname, pce_model.tag, simparams.tag)
    # try:
        # np.save(os.path.join(simparams.data_dir_result, filename), data_test)
    # except:
        # np.save(os.path.join(os.getcwd(), filename), data_test)


if __name__ == '__main__':
    main()
