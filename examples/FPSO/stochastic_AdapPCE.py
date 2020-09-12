#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import uqra, warnings
import numpy as np, os, sys
import scipy.stats as stats
import pickle
from tqdm import tqdm
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()

def inverse_rosenblatt(x_dist, u, u_dist, support=None):
    """
    Map cdf values from [0,1] to truncated support in X space 

    support: boundary in x space
    u: ndarray of shape(solver.ndim, n)

    """
    u1, u2 = np.array(u, ndmin=2, copy=False)
    assert len(u_dist) == 2

    if support is None:
        u_cdf = np.array([idist.cdf(iu) for iu, idist in zip(u, u_dist)])
        x = x_dist.ppf(u_cdf)
    else:
        hs_range, tp_range = support
        u_cdf = np.array([idist.cdf(iu) for iu, idist in zip(u, u_dist)]) ## cdf values in truncated support
        x_cdf = np.ones(u_cdf.shape) 

        ## Hs not depend on any other x, return the Fa, Fb. the 1st row corresponds to hs
        if hs_range is None:
            Fa, Fb = 0,1
        else:
            Fa, Fb = x_dist.cdf(support)[0] 
        ## transform cdf values to non-truncated support
        x_cdf[0] = u_cdf[0] * (Fb - Fa) + Fa
        ## return hs
        hs = x_dist.ppf(x_cdf)[0]  ## only Hs is correct, Tp is wrong

        ## Tp is conditional on Hs, need to find the Fa, Fb for [Tp1, Tp2] condition on each Hs
        if tp_range is None:
            Fa, Fb = 0,1
        else:
            Fa = x_dist.cdf(np.array([hs, np.ones(hs.shape) * tp_range[0] ]))[1]
            Fb = x_dist.cdf(np.array([hs, np.ones(hs.shape) * tp_range[1] ]))[1]
        x_cdf[1] = u_cdf[1] * (Fb - Fa) + Fa

        x = x_dist.ppf(x_cdf)
    return x 

def rosenblatt(x_dist, x, u_dist, support=None):
    """
    Map values from truncated support in X space to u-space 

    support: boundary in x space
    u: ndarray of shape(solver.ndim, n)

    """
    hs, tp = np.array(x, ndmin=2, copy=False)
    assert len(u_dist) == 2

    if support is None:
        x_cdf = x_dist.cdf(x)
        u     = np.array([iu_dist.ppf(ix_cdf) for iu_dist, ix_cdf in zip(u_dist, x_cdf)])
    else:
        hs_range, tp_range = support
        ## get the cdf values in non-truncate support
        x_cdf = x_dist.cdf(x)
        u_cdf = np.ones(x_cdf.shape)
        ## Hs not depend on any other x, return the Fa, Fb. the 1st row corresponds to hs
        if hs_range is None:
            Fa, Fb = 0,1
        else:
            Fa, Fb = x_dist.cdf(support)[0] 
        ## transform to cdf values in truncated support
        u_cdf[0] = (x_cdf[0] - Fa)/(Fb  - Fa)

        ## Tp is conditional on Hs, need to find the Fa, Fb for [Tp1, Tp2] condition on each Hs
        if tp_range is None:
            Fa, Fb = 0,1
        else:
            Fa = x_dist.cdf(np.array([hs, np.ones(hs.shape) * tp_range[0] ]))[1]
            Fb = x_dist.cdf(np.array([hs, np.ones(hs.shape) * tp_range[1] ]))[1]
        u_cdf[1] = (x_cdf[1] - Fa)/(Fb  - Fa)
        u = np.array([iu_dist.ppf(iu_cdf) for iu_dist, iu_cdf in zip(u_dist, u_cdf)])

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

def get_rounded_range(x):
    support = np.array([[np.floor(np.amin(ix)), np.ceil(np.amax(ix))] for ix in x])
    return support

def estimate_beta_params(x, support=None):
    support = get_rounded_range(x) if support is None else np.asarray(support)
    x_min   = support[:,0].reshape(-1,1)
    x_max   = support[:,1].reshape(-1,1)
    x_scaled  = (x- x_min)/(x_max- x_min) * 2 -1 ## scale to [-1,1]
    a, b = [], []
    for ix in x_scaled:
        a.append(stats.beta.fit(ix,floc=-1,fscale=2)[0])
        b.append(stats.beta.fit(ix,floc=-1,fscale=2)[1])
    return a, b
        
def main(theta):
    ## ------------------------ Displaying set up ------------------- ###
    print('\n'+'#' * 80)
    print(' >>>  Start simulation: {:s}, theta={:d}'.format(__file__, theta))
    print('#' * 80 + '\n')
    np.random.seed(100)
    np.set_printoptions(precision=4)
    np.set_printoptions(threshold=8)
    np.set_printoptions(suppress=True)
    Kvitebjorn = uqra.environment.Kvitebjorn()
    pf = 0.5/(50*365.25*24)

    ## ------------------------ Simulation Parameters ----------------- ###
    solver    = uqra.FPSO(random_state =theta)
    simparams = uqra.Parameters(solver, doe_method='MCS', optimality='D', fit_method='LASSOLARS')
    simparams.x_dist     = Kvitebjorn
    simparams.pce_degs   = np.array(range(2,4))
    simparams.n_cand     = int(1e5)
    simparams.n_test     = int(1e6)
    simparams.n_pred     = int(1e7)
    simparams.n_splits   = 50
    simparams.alphas     = 2
    simparams.rosenblatt = rosenblatt
    simparams.inverse_rosenblatt = inverse_rosenblatt
    n_initial = 20
    simparams.info()

    ## ----------- Define U distributions ----------- ###
    u_dist = [stats.norm(0,1),] * solver.ndim
    domain_samples= -stats.norm().ppf(1e-7)
    domain_space  = 'u'
    dist_support  = None
    fname_cand = 'FPSO_SURGE_DoE_UniformE5_Norm.npy'

    # u_dist = [stats.uniform(-1,2),]* solver.ndim
    # domain_samples = [[-1,1],] * solver.ndim 
    # domain_space   = 'u'
    # dist_support = np.array([[0, 18],[0,35]])
    # fname_cand = 'FPSO_SURGE_DoE_LhsE5_Uniform.npy'

    # a, b   = estimate_beta_params(x_pred, support=dist_support)
    # u_dist = [stats.beta(ia,ib,loc=-1,scale=2) for ia, ib in zip(a, b)]
    # support = np.array([[0, 18],[0,35]])
    simparams.set_udist(u_dist)

    ## ----------- Candidate data set for DoE ----------- ###
    print(' > Getting candidate data set...')
    u_cand, x_cand = simparams.get_candidate_data(fname_cand, domain=domain_samples, 
            domain_space=domain_space, support=dist_support)
    print('   - {:<25s} : {}'.format(' Dataset (U)', u_cand.shape))
    print('   - {:<25s} : {}, {}'.format(' U [min(U), max(U)]', np.amin(u_cand, axis=1), np.amax(u_cand, axis=1)))
    print('   - {:<25s} : {}, {}'.format(' X [min(X), max(X)]', np.amin(x_cand, axis=1), np.amax(x_cand, axis=1)))


    ## ----------- Predict data set ----------- ###
    print(' > Getting predict data set...')
    filename = 'CDF_McsE7R{:d}.npy'.format(theta)
    u_pred, x_pred, u_pred_outside, x_pred_outside = simparams.get_predict_data(filename, domain=domain_samples,
            domain_space=domain_space, support=dist_support)
    print('   - {:<25s} : {}'.format(' Sample domain', domain_samples))
    print('   - {:<25s} : {}'.format(' dist Support ', dist_support if dist_support is None else dist_support.reshape(1,-1)))
    print('   - {:<25s} '.format(' Samples inside domain...'))
    print('   - {:<25s} : {}, {}'.format(' U,X.shape', u_pred.shape, x_pred.shape))
    print('   - {:<25s} : {}, {}'.format(' U support', np.amin(u_pred, axis=1), np.amax(u_pred, axis=1)))
    print('   - {:<25s} : {}, {}'.format(' X support', np.amin(x_pred, axis=1), np.amax(x_pred, axis=1)))
    print('   - {:<25s} '.format(' Samples outside domain...'))
    y_pred_outside = np.array(solver.run(x_pred_outside), ndmin=1)
    print('   - {:<25s} : {}, {}'.format('(X,Y) ', x_pred_outside.shape, y_pred_outside.shape))
    print( u_pred_outside)
    print( x_pred_outside)
    print( y_pred_outside)

    ## ----------- Test data set ----------- ###
    print(' > Getting Test data set...')
    filename = '{:s}_DoE_McsE6R{:d}.npy'.format(solver.nickname,theta)
    u_test, x_test, y_test = simparams.get_test_data(filename, domain=domain_samples,
            domain_space=domain_space, support=dist_support)
    print('   - {:<25s} : {}, {}, {}'.format(' U,X,Y.shape', u_test.shape, x_test.shape, y_test.shape ))
    print('   - {:<25s} : {}, {}'.format(' U support', np.amin(u_test, axis=1), np.amax(u_test, axis=1)))
    print('   - {:<25s} : {}, {}'.format(' X support', np.amin(x_test, axis=1), np.amax(x_test, axis=1)))
    print('   - {:<25s} : [{}]'.format(' Y [min(Y), max(Y)]', np.array([np.amin(y_test),np.amax(y_test)])))


    metrics_each_deg   = []
    pred_ecdf_each_deg = []
    pred_topy_each_deg = []
    pce_model_each_deg = []

    print(' > Train data initialization ...')
    ## Initialize u_train with LHS 
    u_train = simparams.get_init_samples(n_initial, doe_method='lhs', random_state=100)
    ## mapping points to the square in X space
    x_train = inverse_rosenblatt(Kvitebjorn, u_train, simparams.u_dist, support=dist_support)
    ## mapping points to physical space
    y_train = solver.run(x_train)
    print('   - {:<25s} : {}, {}, {}'.format(' Dataset (U,X,Y)',u_train.shape, x_train.shape, y_train.shape))
    print('   - {:<25s} : {}, {}'.format(' U support', np.amin(u_train, axis=1), np.amax(u_train, axis=1)))
    print('   - {:<25s} : {}, {}'.format(' X support', np.amin(x_train, axis=1), np.amax(x_train, axis=1)))
    print('   - {:<25s} : [{}]'.format(' Y [min(Y), max(Y)]',np.array([np.amin(y_train),np.amax(y_train)])))
    # print(' > Train data at knot points...')
    # filename = 'FPSO_SURGE_Adap2Jac10_McsD_Alpha2_ST{:d}_Knot.npy'.format(theta)
    # knot_data = np.load(os.path.join(simparams.data_dir_result, 'TestData', filename))
    # u_knot = knot_data[:solver.ndim]
    # x_knot = knot_data[solver.ndim:2*solver.ndim]
    # y_knot = knot_data[-1]
    # print('   - {:<25s} : {}, {}, {}'.format(' Dataset (U,X,Y)',u_knot.shape, x_knot.shape, y_knot.shape))
    # print('   - {:<25s} : {}, {}'.format(' U support', np.amin(u_knot, axis=1), np.amax(u_knot, axis=1)))
    # print('   - {:<25s} : {}, {}'.format(' X support', np.amin(x_knot, axis=1), np.amax(x_knot, axis=1)))
    # print('   - {:<25s} : [{}]'.format(' Y [min(Y), max(Y)]',np.array([np.amin(y_knot),np.amax(y_knot)])))

    # u_train = np.concatenate((u_train, u_knot), axis=1)
    # x_train = np.concatenate((x_train, x_knot), axis=1)
    # y_train = np.concatenate((y_train, y_knot), axis=0)


    for deg in simparams.pce_degs:
        print('\n================================================================================')
        print('   - Sampling and Fitting:')
        print('     - {:<23s} : {}'.format('Sampling method'  , simparams.doe_method))
        print('     - {:<23s} : {}'.format('Optimality '      , simparams.optimality))
        print('     - {:<23s} : {}'.format('Fitting method'   , simparams.fit_method))

        print(' > Building surrogate model ...')
        ## ----------- Define PCE  ----------- ###
        basis     = simparams.get_basis(deg)
        # basis     = simparams.get_basis(deg, a=a, b=b)
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

        pce_model.fit('LASSOLARS', u_train, y_train.T, w=w_train,epsilon=1e-3)

        print('       Active Index: {}'.format(pce_model.active_index))
        print('     > 2. Getting new training data ...')
        pce_model_sparsity = pce_model.sparsity 
        n_train_new = int(simparams.alphas*pce_model_sparsity)
        tqdm.write('    > {}:{}; Basis: {}/{}; #samples = {:d}'.format(
            'Sampling', simparams.optimality, pce_model_sparsity, pce_model.num_basis, n_train_new ))

        u_train_new, _ = modeling.get_train_data(n_train_new, u_cand, u_train=u_train, 
                active_basis=pce_model.active_basis)
        # u_train_normal = u_train_new * u_square_vertice + u_square_center
        # x_train_new = Kvitebjorn.ppf(stats.norm.cdf(u_train_normal))
        x_train_new = inverse_rosenblatt(Kvitebjorn, u_train_new, simparams.u_dist, support=dist_support)
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

        pce_model.fit('OLS', u_train, y_train.T, w_train, active_basis=pce_model.active_basis)

        print(' > Train data ...')
        print('   - {:<25s} : {}, {}, {}'.format(' Dataset (U,X,Y)',u_train.shape, x_train.shape, y_train.shape))
        if w_train is None:
            print('   - {:<25s} : {}'.format(' weight ', 'None'))
        else:
            print('   - {:<25s} : {}'.format(' weight ', w_train.shape))
        print('   - {:<25s} : [{}]'.format(' max(U)[U1, U2]',np.amax(abs(u_train), axis=1)))
        print('   - {:<25s} : {}, {}'.format(' U support', np.amin(u_train, axis=1), np.amax(u_train, axis=1)))
        print('   - {:<25s} : {}, {}'.format(' X support', np.amin(x_train, axis=1), np.amax(x_train, axis=1)))
        print('   - {:<25s} : [{}]'.format(' Y [min(Y), max(Y)]',np.array([np.amin(y_train),np.amax(y_train)])))

        y_train_hat = pce_model.predict(u_train)
        y_test_hat  = pce_model.predict(u_test)
        train_error = uqra.metrics.mean_squared_error(y_train, y_train_hat, squared=False)
        test_error  = uqra.metrics.mean_squared_error(y_test , y_test_hat , squared=False)

        y_pred      = pce_model.predict(u_pred)
        if u_pred_outside is None or len(u_pred_outside) == 0:
            y_pred_     = y_pred
            x_pred_     = x_pred
            u_pred_     = u_pred
        else:
            y_pred_     = np.concatenate((y_pred_outside, y_pred)) 
            x_pred_     = np.concatenate((x_pred_outside, x_pred), axis=1) 
            u_pred_     = np.concatenate((u_pred_outside, u_pred), axis=1) ## could have nan in u

        y_pred_ecdf = uqra.utilities.helpers.ECDF(y_pred_, alpha=pf, compress=True)
        y50_pce_y   = uqra.metrics.mquantiles(y_pred_, 1-pf)
        y50_pce_idx = np.array(abs(y_pred_ - y50_pce_y)).argmin()
        y50_pce_uxy = np.concatenate((u_pred_[:,y50_pce_idx], x_pred_[:, y50_pce_idx], y50_pce_y)) 

        res = [deg, u_train.shape[1], train_error, pce_model.cv_error, test_error, kappa]
        for item in y50_pce_uxy:
            res.append(item)
        pred_ecdf_each_deg.append([deg, u_train.shape[1], u_pred.size, simparams.n_pred, y_pred_ecdf])
        metrics_each_deg.append(res)
        pce_model_each_deg.append(pce_model)

        ### ============ calculating & updating metrics ============
        tqdm.write(' > Summary')
        with np.printoptions(precision=4):
            tqdm.write('     - {:<15s} : {}'.format( 'y50 PCE Y:', y50_pce_y))
            tqdm.write('     - {:<15s} : {}'.format( 'Train MSE' , np.array(metrics_each_deg)[-1:, 2]))
            tqdm.write('     - {:<15s} : {}'.format( 'CV MSE'    , np.array(metrics_each_deg)[-1:, 3]))
            tqdm.write('     - {:<15s} : {}'.format( 'Test MSE ' , np.array(metrics_each_deg)[-1:, 4]))
            tqdm.write('     - {:<15s} : {:.2f}-> {:.2f}'.format( 'kappa ' , kappa0, kappa))
            tqdm.write('     ----------------------------------------')


    ### ============ Saving Parameters ============
    filename = '{:s}_Adap{:s}_{:s}_Alpha{}_ST{}_Parameters.pkl'.format(solver.nickname, pce_model.tag, 
            simparams.tag, str(simparams.alphas).replace('.', 'pt'), theta)
    with open(os.path.join(simparams.data_dir_result, filename), "wb") as output_file:
        pickle.dump(simparams, output_file)

    ### ============ Saving pce model============
    filename = '{:s}_Adap{:s}_{:s}_Alpha{}_ST{}_pce.pkl'.format(solver.nickname, pce_model.tag, 
            simparams.tag, str(simparams.alphas).replace('.', 'pt'), theta)
    with open(os.path.join(simparams.data_dir_result,filename), 'wb') as output:
        pickle.dump(pce_model_each_deg, output, pickle.HIGHEST_PROTOCOL)

    ### ============ Saving train ============
    data_train = np.concatenate((u_train, x_train, y_train.reshape(1,-1)), axis=0)
    filename = '{:s}_Adap{:s}_{:s}_Alpha{}_ST{}_Train'.format(solver.nickname, pce_model.tag, 
            simparams.tag, str(simparams.alphas).replace('.', 'pt'), theta)
    try:
        np.save(os.path.join(simparams.data_dir_result, filename), data_train)
    except:
        print(' Directory not found: {}, file save locally... '.format(simparams.data_dir_result))
        np.save(os.path.join(os.getcwd(), filename), data_train)

    ### ============ Saving test ============
    filename = '{:s}_Adap{:s}_{:s}_Alpha{}_ST{}_test'.format(solver.nickname, pce_model.tag, 
            simparams.tag, str(simparams.alphas).replace('.', 'pt'), theta)
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
            # ['deg', 'n_train', 'train error','cv_error', 'test error', 'kappa', 'y_pred_top mean', 'y_pred_top median']))

    filename = '{:s}_Adap{:s}_{:s}_Alpha{}_ST{}'.format(solver.nickname, pce_model.tag, 
            simparams.tag, str(simparams.alphas).replace('.', 'pt'), theta)
    try:
        np.save(os.path.join(simparams.data_dir_result, filename), metrics_each_deg)
    except:
        print(' Directory not found: {}, file save locally... '.format(simparams.data_dir_result))
        np.save(os.path.join(os.getcwd(), filename), metrics_each_deg)

    ## ============ Saving Predict ecdf data ============
    pred_ecdf_each_deg = np.array(pred_ecdf_each_deg, dtype=object)
    filename = '{:s}_Adap{:s}_{:s}_Alpha{}_ST{}_ecdf'.format(solver.nickname, pce_model.tag, 
            simparams.tag, str(simparams.alphas).replace('.', 'pt'), theta)
    try:
        np.save(os.path.join(simparams.data_dir_result, filename), pred_ecdf_each_deg)
    except:
        print(' Directory not found: {}, file save locally... '.format(simparams.data_dir_result))
        np.save(os.path.join(os.getcwd(), filename), pred_ecdf_each_deg)

if __name__ == '__main__':
    for s in range(10):
        main(s)
