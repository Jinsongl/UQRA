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


def main(theta):
    ## ------------------------ Displaying set up ------------------- ###
    np.random.seed(100)
    np.set_printoptions(precision=4)
    np.set_printoptions(threshold=8)
    np.set_printoptions(suppress=True)
    pf = 1e-5 #0.5/(50*365.25*24)
    radius_surrogate= 5
    Kvitebjorn      = uqra.environment.Kvitebjorn()
    # short_term_seeds_applied = np.setdiff1d(np.arange(10), np.array([]))

    ## ------------------------ Simulation Parameters ----------------- ###
    # theta = np.arange(theta,theta+1)
    # assert theta.size == 1
    solver    = uqra.FPSO(phase=[theta,])
    simparams = uqra.Parameters()
    simparams.solver     = solver
    simparams.pce_degs   = np.array(range(2,11))
    simparams.n_cand     = int(1e5)
    simparams.n_test     = int(1e6)
    simparams.n_pred     = int(1e6)
    simparams.doe_method = 'MCS' ### 'mcs', 'cls1', 'cls2', ..., 'cls5', 'reference'
    simparams.optimality = 'S'# 'D', 'S', None
    simparams.poly_type  = 'hem'
    simparams.fit_method = 'LASSOLARS'
    simparams.n_splits   = 50
    alphas               = 1.2
    simparams.update()
    n_initial = 20
    print('------------------------------------------------------------')
    print('>>> Model: {:s}, Short-term simulation (n={:d})  '.format(solver.nickname, theta))
    print('------------------------------------------------------------')
    simparams.info()


    ## ----------- Test data set ----------- ###
    ## ----- Testing data set centered around u_center, first 100000
    print(' > Getting Test data set...')
    filename    = '{:s}_DoE_McsE6R{:d}.npy'.format(solver.nickname,theta)
    data_test   = np.load(os.path.join(simparams.data_dir_result,'TestData', filename))
    u_test      = data_test[            :  solver.ndim, :]
    x_test      = data_test[solver.ndim :2*solver.ndim, :]
    y_test      = data_test[-1]

    print('   - {:<25s} : {}, {}, {}'.format('Test Dataset (U,X,Y)',u_test.shape, x_test.shape, y_test.shape ))
    print('   - {:<25s} : [{}, {}]'.format('Test U[mean, std]',np.mean(u_test, axis=1),np.std (u_test, axis=1)))
    print('   - {:<25s} : [{}]'.format('Test max(U)[U1, U2]',np.amax(abs(u_test), axis=1)))
    print('   - {:<25s} : [{}]'.format('Test [min(Y), max(Y)]',np.array([np.amin(y_test),np.amax(y_test)])))


    ## ----------- Predict data set ----------- ###
    ## ----- Prediction data set centered around u_center, all  
    filename= 'DoE_McsE7R{:d}.npy'.format(theta)
    mcs_data= np.load(os.path.join(simparams.data_dir_sample,'MCS', 'Norm', filename))
    u_pred  = mcs_data[:solver.ndim, -simparams.n_pred:] 
    x_pred  = Kvitebjorn.ppf(stats.norm.cdf(u_pred))

    ## ----------- Candidate and testing data set for DoE ----------- ###
    print(' > Getting candidate data set...')
    # u_cand = modeling.get_candidate_data()

    if simparams.doe_method.lower().startswith('cls2'):
        filename = os.path.join(simparams.data_dir_sample, 'CLS', 'DoE_Cls2E7d2R{:d}.npy'.format(theta))
        u_cand = np.load(filename)[:solver.ndim, :simparams.n_cand]
        u_cand = u_cand * radius_surrogate

    elif simparams.doe_method.lower().startswith('cls4'):
        filename = os.path.join(simparams.data_dir_sample, 'CLS', 'DoE_Cls4E7d2R{:d}.npy'.format(theta))
        u_cand = np.load(filename)[:solver.ndim, :simparams.n_cand]

    elif simparams.doe_method.lower().startswith('mcs'):
        filename = os.path.join(simparams.data_dir_sample, 'MCS','Norm','DoE_McsE7R0.npy')
        # filename = os.path.join(simparams.data_dir_sample, 'MCS','Norm','DoE_McsE7R{:d}.npy'.format(theta))
        u_cand = np.load(filename)[:solver.ndim, :simparams.n_cand]

    # u_cand    = deg ** 0.5 * u_cand if modeling.is_cls_unbounded() else u_cand
    # u_cand    = 2** 0.5 * u_cand if modeling.is_cls_unbounded() else u_cand


    metrics_each_deg  = []
    pred_uxy_each_deg = []

    for deg in simparams.pce_degs:
        print('\n================================================================================')
        print('   - Sampling and Fitting:')
        print('     - {:<23s} : {}'.format('Sampling method'  , simparams.doe_method))
        print('     - {:<23s} : {}'.format('Optimality '      , simparams.optimality))
        print('     - {:<23s} : {}'.format('Fitting method'   , simparams.fit_method))

        print(' > Building surrogate model ...')
        ## ----------- Define PCE  ----------- ###
        basis     = get_basis(deg, simparams, solver)
        pce_model = uqra.PCE(basis)
        modeling  = uqra.Modeling(solver, pce_model, simparams)
        pce_model.info()

        ### ----------- Oversampling ratio ----------- ###
        simparams.update_num_samples(pce_model.num_basis, alphas=alphas)
        print(' > Oversampling ratio: {}'.format(np.around(simparams.alphas,2)))
        n_train  = int(alphas * pce_model.num_basis)  

        print(' > Getting Train data set...')
        if simparams.optimality is None:
            doe_idx_u_cand = np.arange(simparams.n_cand) 
            filename= 'Random index'
        else:
            # filename = 'DoE_{:s}E6R{:d}_{:s}_{:s}.npy'.format(simparams.doe_method.capitalize(), theta,
            filename = 'DoE_{:s}E5R0_{:s}_{:s}.npy'.format(simparams.doe_method.capitalize(), 
                         pce_model.tag, simparams.optimality)
            doe_idx_u_cand = np.load(os.path.join(simparams.data_dir_sample, 'OED', filename))
            # doe_idx_u_cand = doe_idx_u_cand, 

        ### ============ Build Surrogate Model ============
        u_train = u_cand[:, doe_idx_u_cand[:n_train]]
        x_train = Kvitebjorn.ppf(stats.norm.cdf(u_train)) 
        y_train = solver.run(x_train)
        U_train = pce_model.basis.vandermonde(u_train)
        if simparams.doe_method.lower().startswith('cls'):
            w_train = modeling.cal_cls_weight(u_train, pce_model.basis, active_index=pce_model.active_index)
            U_train = U_train[:, pce_model.active_index]
            U_train = modeling.rescale_data(U_train, w_train) 
        else:
            w_train = None
            U_train = U_train[:, pce_model.active_index]
        _, sig_value, _ = np.linalg.svd(U_train)
        kappa = max(abs(sig_value)) / min(abs(sig_value)) 

        pce_model.fit(simparams.fit_method, u_train, y_train.T, w=w_train,
                n_splits=simparams.n_splits, active_basis=pce_model.active_basis)

        print('   - {:<25s} : {:s}'.format('File', filename))
        print('   - {:<25s} : {}, {}, {}'.format('Train Dataset (U,X,Y)',u_train.shape, x_train.shape, y_train.shape))
        if w_train is None:
            print('   - {:<25s} : {}'.format('Train Dataset W ', 'None'))
        else:
            print('   - {:<25s} : {}'.format('Train Dataset W ', w_train.shape))
        print('   - {:<25s} : [{}, {}]'.format('Train U[mean, std]',np.mean(u_train, axis=1),np.std (u_train, axis=1)))
        print('   - {:<25s} : [{}]'.format('Train max(U)[U1, U2]',np.amax(abs(u_train), axis=1)))
        print('   - {:<25s} : [{}]'.format('Train [min(Y), max(Y)]',np.array([np.amin(y_train),np.amax(y_train)])))

        y_train_hat = pce_model.predict(u_train)
        y_test_hat  = pce_model.predict(u_test)
        train_error = uqra.metrics.mean_squared_error(y_train, y_train_hat, squared=False)
        test_error  = uqra.metrics.mean_squared_error(y_test, y_test_hat, squared=False)


        ### prediction data set, randomly draw or from MCS directory
        # y_pred = pce_model.predict(u_pred - u_center)
        # alpha  = (pf * mcs_data_ux.shape[1]) / y_pred.size
        # y50_pce_y   = uqra.metrics.mquantiles(y_pred, 1-alpha)
        # y50_pce_idx = np.array(abs(y_pred - y50_pce_y)).argmin()
        # y50_pce_uxy = np.concatenate((u_pred[:,y50_pce_idx], x_pred[:, y50_pce_idx], y50_pce_y)) 
        # pred_uxy_each_deg.append([deg, n_train, y_pred])

        # np.random.seed()
        # u_pred = stats.norm.rvs(size=(solver.ndim,simparams.n_pred))
        # x_pred = Kvitebjorn.ppf(stats.norm.cdf(u_pred))
        y_pred = pce_model.predict(u_pred)
        y50_pce_y   = uqra.metrics.mquantiles(y_pred, 1-pf)
        y50_pce_idx = np.array(abs(y_pred - y50_pce_y)).argmin()
        y50_pce_uxy = np.concatenate((u_pred[:,y50_pce_idx], x_pred[:, y50_pce_idx], y50_pce_y)) 
        pred_uxy_each_deg.append([deg, u_train.shape[1], y_pred])

        res = [deg, u_train.shape[1], train_error, pce_model.cv_error, test_error, kappa]
        for item in y50_pce_uxy:
            res.append(item)
        metrics_each_deg.append(res)

        ### ============ calculating & updating metrics ============
        tqdm.write(' > Summary')
        with np.printoptions(precision=4):
            tqdm.write('     - {:<15s} : {}'.format( 'y50_pce_y' , np.array(metrics_each_deg)[-1:,-1]))
            tqdm.write('     - {:<15s} : {}'.format( 'Train MSE' , np.array(metrics_each_deg)[-1:, 2]))
            tqdm.write('     - {:<15s} : {}'.format( 'CV MSE'    , np.array(metrics_each_deg)[-1:, 3]))
            tqdm.write('     - {:<15s} : {}'.format( 'Test MSE ' , np.array(metrics_each_deg)[-1:, 4]))
            tqdm.write('     - {:<15s} : {}'.format( 'kappa '    , np.array(metrics_each_deg)[-1:, 5]))
            tqdm.write('     ----------------------------------------')

    ### ============ Saving QoIs ============
    metrics_each_deg = np.array(metrics_each_deg)
    with open(os.path.join(simparams.data_dir_result, 'outlist_name.txt'), "w") as text_file:
        text_file.write(', '.join(
            ['deg', 'n_train', 'train error', 'cv_error', 'test mse', 'kappa','y50_pce_u', 'y50_pce_x', 'y50_pce_y']))

    filename = '{:s}_{:s}_{:s}_Alpha{}_ST{}'.format(solver.nickname, pce_model.tag, 
            simparams.tag, str(alphas).replace('.', 'pt'), theta)
    try:
        np.save(os.path.join(simparams.data_dir_result, filename), metrics_each_deg)
    except:
        print(' Directory not found: {}, file save locally... '.format(simparams.data_dir_result))
        np.save(os.path.join(os.getcwd(), filename), metrics_each_deg)

    ### ============ Saving Predict data ============
    pred_uxy_each_deg = np.array(pred_uxy_each_deg, dtype=object)
    filename = '{:s}_{:s}_{:s}_Alpha{}_ST{}_pred'.format(solver.nickname, pce_model.tag, 
            simparams.tag, str(alphas).replace('.', 'pt'), theta)
    try:
        np.save(os.path.join(simparams.data_dir_result, filename), pred_uxy_each_deg)
    except:
        print(' Directory not found: {}, file save locally... '.format(simparams.data_dir_result))
        np.save(os.path.join(os.getcwd(), filename), pred_uxy_each_deg)

if __name__ == '__main__':
    for s in range(6,10):
        main(s)
