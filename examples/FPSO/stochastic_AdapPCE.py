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

def inverse_rosenblatt(x_dist, u, x_range=None, u_dist=stats.norm()):
    """
    Map cdf values from [0,1] to truncated domain in X space 

    x_range: boundary
    u: ndarray of shape(solver.ndim, n)

    """
    if x_range is None:
        u_cdf = u_dist.cdf(u)
        x = x_dist.ppf(u_cdf)
    else:
        x_range = np.array(x_range, ndmin=2)
        hs_range, tp_range = x_range
        u_cdf   = u_dist.cdf(u) ## cdf values in truncated domain

        ## Hs not depend on any other x, return the Fa, Fb. the 1st row corresponds to hs
        Fa_hs, Fb_hs = x_dist.cdf(x_range)[0] 
        ## transform cdf values to non-truncated domain
        u_cdf[0] = u_cdf[0] * (Fb_hs - Fa_hs) + Fa_hs
        ## return hs
        hs = x_dist.ppf(u_cdf)[0]  ## only Hs is correct, Tp is wrong

        ## Tp is conditional on Hs, need to find the Fa, Fb for [Tp1, Tp2] condition on each Hs
        Fa_tp     = x_dist.cdf(np.array([hs, np.ones(hs.shape) * tp_range[0] ]))[1]
        Fb_tp     = x_dist.cdf(np.array([hs, np.ones(hs.shape) * tp_range[1] ]))[1]
        u_cdf[1] = u_cdf[1] * (Fb_tp - Fa_tp) + Fa_tp

        x = x_dist.ppf(u_cdf)
    return x 

def rosenblatt(x_dist, x, x_range=None,  u_dist=stats.norm()):
    """
    Map values from truncated domain in X space to uspace 

    x_range: boundary
    u: ndarray of shape(solver.ndim, n)

    """
    hs, tp  = np.array(x, ndmin=2)

    if x_range is None:
        x_cdf   = x_dist.cdf(x)
        u       = u_dist.ppf(x_cdf)
    else:
        x_range = np.array(x_range, ndmin=2)
        hs_range, tp_range = x_range

        ## get the cdf values in non-truncate domain
        x_cdf   = x_dist.cdf(x)
        ## Hs not depend on any other x, return the Fa, Fb. the 1st row corresponds to hs
        Fa_hs, Fb_hs = x_dist.cdf(x_range)[0] 
        ## transform to cdf values in truncated domain
        x_cdf[0] = (x_cdf[0] - Fa_hs)/(Fb_hs  - Fa_hs)

        ## Tp is conditional on Hs, need to find the Fa, Fb for [Tp1, Tp2] condition on each Hs
        Fa_tp     = x_dist.cdf(np.array([hs, np.ones(hs.shape) * tp_range[0] ]))[1]
        Fb_tp     = x_dist.cdf(np.array([hs, np.ones(hs.shape) * tp_range[1] ]))[1]
        x_cdf[1] = (x_cdf[1] - Fa_tp)/(Fb_tp  - Fa_tp)

        u = u_dist.ppf(x_cdf)
    return u 

def get_pts_inside_square(x, center=[0,0], edges=[1,1]):
    """
    return coordinates of points inside the defined square
    """
    x = np.array(x, ndmin=2)
    center = np.squeeze(center)
    edges = np.squeeze(edges)
    m, n = x.shape
    
    center = np.array(center).reshape(m, -1)
    x = x - center
    idx1 = abs(x[0]) <= abs(edges[0]/2)  
    idx2 = abs(x[1]) <= abs(edges[1]/2)
    idx  = np.logical_and(idx1,idx2)
    return idx

def get_basis(deg, simparams, solver):
    if simparams.doe_method.lower().startswith('mcs'):
        if simparams.u_dist.dist.name == 'uniform':
            print(' Legendre polynomial')
            basis = uqra.Legendre(d=solver.ndim, deg=deg)

        elif simparams.u_dist.dist.name == 'norm':
            print(' Probabilists Hermite polynomial')
            basis = uqra.Hermite(d=solver.ndim,deg=deg, hem_type='probabilists')
        else:
            raise ValueError 

    elif simparams.doe_method.lower().startswith('cls'):
        if simparams.u_dist.dist.name == 'uniform':
            print(' Legendre polynomial')
            basis = uqra.Legendre(d=solver.ndim,deg=deg)

        elif simparams.u_dist.dist.name == 'norm':
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
    solver    = uqra.FPSO(random_state =theta)
    simparams = uqra.Parameters()
    simparams.solver     = solver
    simparams.pce_degs   = np.array(range(2,11))
    simparams.n_cand     = int(1e5)
    simparams.n_test     = int(1e6)
    simparams.n_pred     = int(1e6)
    simparams.doe_method = 'MCS' ### 'mcs', 'cls1', 'cls2', ..., 'cls5', 'reference'
    simparams.optimality = 'S'# 'D', 'S', None
    simparams.u_dist     = stats.norm(0,1)
    # simparams.u_dist   = stats.uniform(-1,2)
    simparams.fit_method = 'LASSOLARS'
    simparams.n_splits   = 50
    alphas  = 1.2
    simparams.update()
    n_initial = 20
    # hs_range = [9,16]
    # tp_range = [9,20]
    hs_range = [0,20]
    tp_range = [0,40]
    x_range  = [hs_range, tp_range]
    x_square_center = np.mean([hs_range, tp_range], axis=1)
    x_square_edges  = np.array([hs_range[1] - hs_range[0],tp_range[1] - tp_range[0]])

    print('------------------------------------------------------------')
    print('>>> Model: {:s}, Short-term simulation (n={:d})  '.format(solver.nickname, theta))
    print('------------------------------------------------------------')
    simparams.info()

    ## ----------- Test data set ----------- ###
    ## ----- Testing data set centered around u_center, first 100000
    print(' > Getting Test data set...')
    filename    = '{:s}_DoE_McsE6R{:d}.npy'.format(solver.nickname,theta)
    data_test   = np.load(os.path.join(simparams.data_dir_result,'TestData', filename))
    x_test      = data_test[solver.ndim :2* solver.ndim, :]
    y_test      = data_test[-1]
    x_test_idx  = get_pts_inside_square(x_test, center=x_square_center, edges=x_square_edges)
    x_test      = x_test[:, x_test_idx]
    u_test      = rosenblatt(Kvitebjorn, x_test, x_range=x_range, u_dist=simparams.u_dist)
    y_test      = y_test[   x_test_idx]

    print('   - {:<25s} : {}, {}, {}'.format('Test Dataset (U,X,Y)', u_test.shape, x_test.shape, y_test.shape ))
    print('   - {:<25s} : [{}, {}]'.format('Test U[mean, std]',np.mean(u_test, axis=1),np.std (u_test, axis=1)))
    print('   - {:<25s} : [{}]'.format('Test max(U)[U1, U2]', np.amax(abs(u_test), axis=1)))
    print('   - {:<25s} : [{}]'.format('Test [min(Y), max(Y)]', np.array([np.amin(y_test),np.amax(y_test)])))

    ## ----------- Predict data set ----------- ###
    ## ----- Prediction data set centered around u_center, all  
    filename    = 'DoE_McsE7R{:d}.npy'.format(theta)
    mcs_data    = np.load(os.path.join(simparams.data_dir_sample,'MCS', simparams.u_dist.dist.name.capitalize(), filename))
    u_pred      = mcs_data[:solver.ndim, :simparams.n_pred] 
    x_pred      = Kvitebjorn.ppf(simparams.u_dist.cdf(u_pred))
    x_pred_idx  = get_pts_inside_square(x_pred, center=x_square_center, edges=x_square_edges)
    x_pred      = x_pred[:, x_pred_idx]
    u_pred      = rosenblatt(Kvitebjorn, x_pred, x_range=x_range, u_dist=simparams.u_dist)

    ## ----------- Candidate and testing data set for DoE ----------- ###
    print(' > Getting candidate data set...')

    if simparams.doe_method.lower().startswith('cls1'):
        filename = os.path.join(simparams.data_dir_sample, 'CLS', 'DoE_Cls1E7d2R{:d}.npy'.format(theta))
        u_cand = np.load(filename)[:solver.ndim, :simparams.n_cand]

    elif simparams.doe_method.lower().startswith('cls2'):
        filename = os.path.join(simparams.data_dir_sample, 'CLS', 'DoE_Cls2E7d2R{:d}.npy'.format(theta))
        u_cand = np.load(filename)[:solver.ndim, :simparams.n_cand]
        u_cand = u_cand * radius_surrogate

    elif simparams.doe_method.lower().startswith('cls4'):
        filename = os.path.join(simparams.data_dir_sample, 'CLS', 'DoE_Cls4E7d2R{:d}.npy'.format(theta))
        u_cand = np.load(filename)[:solver.ndim, :simparams.n_cand]

    elif simparams.doe_method.lower().startswith('mcs'):
        filename = os.path.join(simparams.data_dir_sample, 'MCS', simparams.u_dist.dist.name.capitalize(),'DoE_McsE7R0.npy')
        # filename = os.path.join(simparams.data_dir_sample, 'MCS','Norm','DoE_McsE7R{:d}.npy'.format(theta))
        u_cand = np.load(filename)[:solver.ndim, :simparams.n_cand]
    ### ============ Initial Values ============

    metrics_each_deg  = []
    pred_uxy_each_deg = []

    ## Initialize u_train with LHS 
    if simparams.doe_method.lower().startswith('mcs'):
        doe     = uqra.LHS([simparams.u_dist,]*solver.ndim)
        u_train = doe.samples(size=n_initial, random_state=100)  ## u_i ~ [-1, 1]
    elif simparams.doe_method.lower().startswith('cls'):
        u_train = u_cand[:, :n_initial]
    ## mapping points to the square in X space
    x_train = inverse_rosenblatt(Kvitebjorn, u_train, x_range=x_range, u_dist=simparams.u_dist)
    ## mapping points to physical space
    y_train = solver.run(x_train)

    print('   - {:<25s} : {}, {}, {}'.format('Train Dataset (U,X,Y)',u_train.shape, x_train.shape, y_train.shape))
    print('   - {:<25s} : [{}, {}]'.format('Train U[mean, std]',np.mean(u_train, axis=1),np.std (u_train, axis=1)))
    print('   - {:<25s} : [{}]'.format('Train max(U)[U1, U2]',np.amax(abs(u_train), axis=1)))
    print('   - {:<25s} : [{}]'.format('Train [min(Y), max(Y)]',np.array([np.amin(y_train),np.amax(y_train)])))

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

        ### ============ Get training points ============
        print('     > 1. Sparsity estimation ...')
        U_train = pce_model.basis.vandermonde(u_train)
        if simparams.doe_method.lower().startswith('cls'):
            w_train = modeling.cal_cls_weight(u_train, pce_model.basis, pce_model.active_index)
            U_train = U_train[:, pce_model.active_index]
            U_train = modeling.rescale_data(U_train, w_train) 
        else:
            w_train = None
            U_train = U_train[:, pce_model.active_index]

        _, sig_value, _ = np.linalg.svd(U_train)
        kappa0 = max(abs(sig_value)) / min(abs(sig_value)) 

        pce_model.fit('LASSOLARS', u_train, y_train.T, w=w_train, 
                n_splits=simparams.n_splits, epsilon=1e-2)

        print('       Active Index: {}'.format(pce_model.active_index))
        print('     > 2. Getting new training data ...')
        pce_model_sparsity = pce_model.sparsity 
        # n_train_new = int(alphas*pce_model.num_basis)
        n_train_new = min(int(alphas*pce_model.num_basis), pce_model_sparsity)
        tqdm.write('    > {}:{}; Basis: {}/{}; #samples = {:d}'.format(
            'Sampling', simparams.optimality, pce_model_sparsity, pce_model.num_basis, n_train_new ))

        u_train_new, _ = modeling.get_train_data(n_train_new, u_cand, u_train, basis=pce_model.basis, 
                active_basis=pce_model.active_basis)
        # u_train_normal = u_train_new * u_square_vertice + u_square_center
        # x_train_new = Kvitebjorn.ppf(stats.norm.cdf(u_train_normal))
        x_train_new = inverse_rosenblatt(Kvitebjorn, u_train_new, x_range=x_range, u_dist=simparams.u_dist)
        y_train_new = solver.run(x_train_new)
        u_train = np.hstack((u_train, u_train_new)) 
        x_train = np.hstack((x_train, x_train_new)) 
        y_train = np.hstack((y_train, y_train_new)) 
        ### ============ Build 2nd Surrogate Model ============
        # print(bias_weight)
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

        pce_model.fit('OLS', u_train, y_train.T, w_train, 
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
        test_error  = uqra.metrics.mean_squared_error(y_test , y_test_hat , squared=False)

        # np.random.seed()
        # u_pred = stats.norm.rvs(loc=0,scale=1,size=(solver.ndim, simparams.n_pred))
        # x_pred = Kvitebjorn.ppf(stats.norm.cdf(u_pred))
        # u_pred_     = (u_pred-u_square_center)/u_square_vertice
        print('   - {:<25s} : [{}]'.format('Predict min(U)[U1, U2]',np.amin(u_pred, axis=1)))
        print('   - {:<25s} : [{}]'.format('Predict max(U)[U1, U2]',np.amax(u_pred, axis=1)))
        y_pred      = pce_model.predict(u_pred)
        print('   - {:<25s} : {}, {}, {}'.format('Predict Dataset (U,X,Y)',u_pred.shape, x_pred.shape, y_pred.shape))
        alpha       = simparams.n_pred * pf / y_pred.size
        y50_pce_y   = uqra.metrics.mquantiles(y_pred, 1-alpha)
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
            tqdm.write('     - {:<15s} : {:.2f}-> {:.2f}'.format( 'kappa ' , kappa0, kappa))
            tqdm.write('     ----------------------------------------')

    ### ============ Saving train ============
    data_train = np.concatenate((u_train, x_train, y_train.reshape(1,-1)), axis=0)
    filename = '{:s}_Adap{:s}_{:s}_Alpha{}_ST{}_Train'.format(solver.nickname, pce_model.tag, 
            simparams.tag, str(alphas).replace('.', 'pt'), theta)
    try:
        np.save(os.path.join(simparams.data_dir_result, filename), data_train)
    except:
        print(' Directory not found: {}, file save locally... '.format(simparams.data_dir_result))
        np.save(os.path.join(os.getcwd(), filename), data_train)

    ### ============ Saving test ============
    filename = '{:s}_Adap{:s}_{:s}_Alpha{}_ST{}_test'.format(solver.nickname, pce_model.tag, 
            simparams.tag, str(alphas).replace('.', 'pt'), theta)
    try:
        np.save(os.path.join(simparams.data_dir_result, filename), y_test_hat)
    except:
        print(' Directory not found: {}, file save locally... '.format(simparams.data_dir_result))
        np.save(os.path.join(os.getcwd(), filename), y_test_hat)

    ### ============ Saving QoIs ============
    metrics_each_deg = np.array(metrics_each_deg)
    with open(os.path.join(simparams.data_dir_result, 'outlist_name.txt'), "w") as text_file:
        text_file.write(', '.join(
            ['deg', 'n_train', 'train error','cv_error', 'test error', 'kappa', 'y50_pce_u', 'y50_pce_x', 'y50_pce_y']))

    filename = '{:s}_Adap{:s}_{:s}_Alpha{}_ST{}'.format(solver.nickname, pce_model.tag, 
            simparams.tag, str(alphas).replace('.', 'pt'), theta)
    try:
        np.save(os.path.join(simparams.data_dir_result, filename), metrics_each_deg)
    except:
        print(' Directory not found: {}, file save locally... '.format(simparams.data_dir_result))
        np.save(os.path.join(os.getcwd(), filename), metrics_each_deg)

    ### ============ Saving Predict data ============
    pred_uxy_each_deg = np.array(pred_uxy_each_deg, dtype=object)
    filename = '{:s}_Adap{:s}_{:s}_Alpha{}_ST{}_pred'.format(solver.nickname, pce_model.tag, 
            simparams.tag, str(alphas).replace('.', 'pt'), theta)
    try:
        np.save(os.path.join(simparams.data_dir_result, filename), pred_uxy_each_deg)
    except:
        print(' Directory not found: {}, file save locally... '.format(simparams.data_dir_result))
        np.save(os.path.join(os.getcwd(), filename), pred_uxy_each_deg)

if __name__ == '__main__':
    for s in range(10):
        main(s)
