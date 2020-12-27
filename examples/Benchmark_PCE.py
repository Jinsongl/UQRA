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
def main(s=0):

    ## ------------------------ Displaying set up ------------------- ###
    print('\n#################################################################################')
    print(' >>>  Start UQRA : {:d}'.format(s), __file__)
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


    ## ------------------------ UQRA DOE Parameters ----------------- ###
    doe_params = uqra.Parameters()
    doe_params.doe_sampling = 'MCS' 
    doe_params.optimality   = [None, 'S', 'D'] 
    doe_params.dist_name    = 'uniform'
    doe_params.num_cand     = int(1e5)

    ## ------------------------ UQRA Modeling Parameters ----------------- ###
    model_params = uqra.Modeling()
    model_params.name    = 'PCE'
    model_params.degs    = [2,6,10]#np.arange(3,11)
    model_params.ndim    = solver.ndim
    model_params.basis   = 'Leg'
    model_params.fitting = 'OLS' 
    model_params.n_splits= 50
    model_params.alpha   = 2
    model_params.num_test= int(2e5)
    model_params.num_pred= int(1e7)
    model_params.info()


    for deg in model_params.degs:

        ## ------------------------ UQRA Surrogate model----------------- ###
        orth_poly = uqra.poly.orthogonal(model_params.ndim, deg, model_params.basis)
        pce_model = uqra.PCE(orth_poly)
        pce_model.info()
        ## ------------------------ UQRA Simulation Parameters ----------------- ###
        sim_params = uqra.Simulation(solver, pce_model, doe_params)
        sim_params.update_filenames(s)
        filename_test = sim_params.fname_test(solver.nickname)

        print('   > {:<25s}'.format('Input/Output Directories:'))
        print('     - {:<23s} : {:s}'.format(' Working location'  , sim_params.pwd))
        print('     - {:<23s} : {:s}'.format(' Random Samples'    , sim_params.data_dir_random))
        print('     - {:<23s} : {:s}'.format(' UQRA DoE data '    , sim_params.data_dir_doe))
        print('     - {:<23s} : {:s}'.format(' Output data '      , sim_params.data_dir_result))
        print('     - {:<23s} : {:s}'.format(' Output figures'    , sim_params.figure_dir))

        filename_cand   = sim_params.fname_cand
        print('   > {:<25s}'.format('Input/Output files'))
        print('     - {:<23s} : {}'.format(' Cadidate samples'  , filename_cand  ))
        print('     - {:<23s} : {}'.format(' Test data filename', filename_test  ))
        ### 1. Get candidate data set
        if filename_cand:
            print('     - {:<23s} : {:s}'.format(' Candidate samples', filename_cand))
            data_cand = np.load(os.path.join(sim_params.data_dir_random, filename_cand))[:model_params.ndim, :]
            print('     ..{:<30s} shape: {}'.format(' Candidate samples loaded,', data_cand.shape))

        ### 2. Get test data set
        try:
            data_test = np.load(os.path.join(sim_params.data_dir_test, filename_test), allow_pickle=True).tolist()
            
            if isinstance(data_test,  uqra.Data):
                pass
            else:
                data_test = data_test[0]
                assert isinstance(data_test, Data)
        except FileNotFoundError:
            print('     - Preparing Test data (UQRA.Solver: {:s})... '.format(solver.nickname))
            fname_test_in = os.path.join(sim_params.data_dir_random, sim_params.fname_test_in)
            print('     .. Input test data:', fname_test_in)

            data_test   = uqra.Data()
            data_test.x = np.load(fname_test_in)[:model_params.ndim, :model_params.num_test]
            data_test.y = solver.run(data_test.x)
            np.save(os.path.join(sim_params.data_dir_test, filename_test), data_test, allow_pickle=True)
            print('     .. Saving test data to {:s}, shape: x={}, y={}... '.format(filename_test, 
                data_test.x.shape, data_test.y.shape))
        print('     ..{:<30s} shape: {} {}'.format(' Test data loaded,', data_test.x.shape, data_test.y.shape))

        output_indim_ideg = uqra.Data()
        if doe_params.doe_sampling.lower() == 'lhs':
            all_doe_cases = [(doe_params.doe_sampling, None)]
        else:
            all_doe_cases = [(doe_params.doe_sampling, ioptimality) for ioptimality in doe_params.optimality] 

        for doe_params.doe_sampling, ioptimality in all_doe_cases:
            doe_nickname = uqra.Experiment().nickname(doe_params.doe_sampling, ioptimality)
            n_samples    = model_params.alpha * pce_model.num_basis
            print('     ----------------------------------------')
            print('     UQRA Training with Experimental Design {} '.format(doe_nickname))
            print('     ----------------------------------------')
            print('   -> Training with (n={:d}, alpha={:.2f}) samples'.format(n_samples, model_params.alpha))

            if doe_params.doe_sampling.lower() == 'lhs':
                filename_design = sim_params.fname_design(n_samples)
            else:
                filename_design = sim_params.fname_design
            print('     - {:<23s} : {}'.format(' UQRA DoE filename' , filename_design))
            data_design  = np.load(os.path.join(sim_params.data_dir_doe, filename_design), allow_pickle=True)
            print('     ..{:<30s} shape: {}'.format(' Deisgn samples loaded,', data_design.shape))
            data = uqra.Data()
            ### data_deisgn has more than one (50) set of optimal samples, choose the first one
            data_design  = data_design[0]
            if doe_params.doe_sampling.lower() == 'lhs':
                assert data_design.shape == (model_params.ndim, n_samples) 
                x_train = data_design 
            else:
                assert data_design.deg == deg and data_design.ndim == model_params.ndim
                optimal_samples_idx = getattr(data_design, doe_nickname)
                x_train = data_cand[:model_params.ndim, optimal_samples_idx[:n_samples]]
                if pce_model.orth_poly.nickname.lower()=='hem' and doe_params.doe_sampling.lower()=='cls4':
                    x_train = x_train * deg **0.5

            ### 3. train model 
            y_train = solver.run(x_train) 
            y_train = y_train #+ observation_error(y_train)
            X_train = pce_model.orth_poly.vandermonde(x_train)
            # X_train = orth_poly.vandermonde(x_train)
            if doe_params.doe_sampling.startswith('cls'):
                ### reproducing kernel
                WX_train = pce_model.orth_poly.num_basis**0.5*(X_train.T / np.linalg.norm(X_train, axis=1)).T
                w = pce_model.christoffel_weight(x_train, pce_model.orth_poly, active=None)
            else:
                WX_train = X_train
                w = None
            ## condition number, kappa = max(svd)/min(svd)
            _, sigular_values, _ = np.linalg.svd(WX_train)
            # pce_model.fit_lassolars(x_train, y_train, w=w)
            pce_model.fit(model_params.fitting, x_train, y_train, w=w)
            y_test = pce_model.predict(data_test.x)
            data.test_error = np.linalg.norm(y_test - data_test.y) / np.linalg.norm(data_test.y)
            data.model      = pce_model
            data.beta_hat   = pce_model.coef
            data.kappa      = max(abs(sigular_values)) / min(abs(sigular_values))
            data.filename_test = os.path.join(sim_params.data_dir_test, filename_test)

            tqdm.write(' > Summary')
            with np.printoptions(precision=4):
                # tqdm.write('     - {:<15s} : {}'.format( 'QoI'       , QoI))
                tqdm.write('     - {:<15s} : {:.4e}'.format( 'Test MSE ' , data.test_error))
                tqdm.write('     - {:<15s} : {:.4f}'.format( 'CV MSE'    , pce_model.cv_error))
                tqdm.write('     - {:<15s} : {:.4f}'.format( 'Score '    , pce_model.score))
                tqdm.write('     - {:<15s} : {:.4f}'.format( 'kappa '    , data.kappa))
                tqdm.write('     ----------------------------------------')

    ### ============ Saving QoIs ============
    # metrics_each_deg = np.array(metrics_each_deg)
    # with open(os.path.join(sim_params.data_dir_result, 'outlist_name.txt'), "w") as text_file:
        # text_file.write('\n'.join(['deg', 'n', 'kappa', 'score','cv_error', 'test_mse' ]))

    # # filename = '{:s}_{:s}_{:s}_Alpha{:s}'.format(solver.nickname, pce_model.tag, sim_params.tag,
            # # str(alphas).replace('.', 'pt'))
    # filename = '{:s}_{:s}_{:s}_NumSamples'.format(solver.nickname, pce_model.tag, sim_params.tag)
    # # filename = '{:s}_{:s}_{:s}_reference'.format(solver.nickname, pce_model.tag, sim_params.tag)
    # try:
        # np.save(os.path.join(sim_params.data_dir_result, filename), np.array(metrics_each_deg))
    # except:
        # np.save(os.path.join(os.getcwd(), filename), np.array(metrics_each_deg))




if __name__ == '__main__':
    main(0)
