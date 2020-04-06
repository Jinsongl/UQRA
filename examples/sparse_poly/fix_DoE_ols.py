#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""
This script run simulations to check the stabiltiy and accuracy of least square method.
A few cases could be tested:
    1. low dimension, high order (ndim, p_orders), p_order contains the maximal degree of polynomial
    2. high dimension, low order
    3. moderate dimension and order

Sampling scheme compared:
    1. MCS, repeated nrepeat times to have an average estimate
    2. D- Optimal design from the candidate data set of size n_cand
    3. S- Optimal design from the candidate data set of size n_cand
    4. Sampling based on Christoffel functions 
    For comparison, a reference solution could be returned with doe_method='all', which use n_cand samples from MCS
    
Testing:
    a new data set with size n_cand are used as testing set
Returns:
    rate_coef: Rate of successfully recovery of polynomial coefficients within given accuracy in l2 norm
    mse_global: Global mean square error of predict model for test data 
    mse_exceed: Mean square error exceeding threshold value of predict model for test data 
    cond_number: condition number of model matrix
    f_hat_coef: coefficients of predict model
"""
import numpy as np
from sklearn import metrics
import math
from tqdm import tqdm
import os, platform
import museuq
import scipy.stats as stats

def sparse_poly_coef_error(solver, model, normord=np.inf):
    beta    = np.array(solver.coef, copy=True)
    beta_hat= np.array(model.coef, copy=True)

    solver_basis_degree = solver.basis.basis_degree
    model_basis_degree  = model.basis.basis_degree

    if len(solver_basis_degree) > len(model_basis_degree):
        large_basis_degree = solver_basis_degree 
        small_basis_degree = model_basis_degree
        large_beta = beta
        small_beta = beta_hat
    else:
        small_basis_degree = solver_basis_degree 
        large_basis_degree = model_basis_degree
        large_beta = beta_hat
        small_beta = beta

    basis_common = np.where([ibasis_degree in small_basis_degree  for ibasis_degree in large_basis_degree ])[0]
    if normord == np.inf:
        error_common = np.linalg.norm(large_beta[basis_common]-small_beta, normord)
        large_beta[basis_common] = 0
        error_left   =  max(abs(large_beta))
        error = max( error_common, error_left )
    elif normord == 1 or normord == 2 :
        error = np.linalg.norm(large_beta[basis_common]-small_beta, normord)/ np.linalg.norm(beta, normord)
    return  error

def get_candidate_data(simparams, sampling_method, orth_poly, n_cand, n_test):
    """
    Return canndidate samples in u space
    """
    if sampling_method.lower().startswith('mcs'):
        filename= r'DoE_McsE6R0.npy'
        mcs_data_set  = np.load(os.path.join(simparams.data_dir_sample, 'MCS', orth_poly.dist_name, filename))
        u_cand = mcs_data_set[:orth_poly.ndim,:n_cand].reshape(orth_poly.ndim, -1)
        u_test = mcs_data_set[:orth_poly.ndim,:n_test] if n_test > 1 else mcs_data_set[:orth_poly.ndim,:]

    elif sampling_method.lower().startswith('cls') or sampling_method.lower() == 'reference':
        filename= r'DoE_McsE6d{:d}R0.npy'.format(orth_poly.ndim) if orth_poly.dist_name.lower() == 'normal' else r'DoE_McsE6R0.npy'
        cls_data_set  = np.load(os.path.join(simparams.data_dir_sample, 'Pluripotential', orth_poly.dist_name, filename))
        u_cand = cls_data_set[:orth_poly.ndim,:n_cand].reshape(orth_poly.ndim, -1)
        u_test = cls_data_set[:orth_poly.ndim,:n_test] if n_test > 1 else cls_data_set[:orth_poly.ndim,:]
    else:
        raise ValueError

    return u_cand, u_test

def get_train_data(simparams, sampling_method, optimality, pce_model, nsamples, u_cand_p, nrepeat):

    if optimality is None:
        idx     = np.random.randint(0, u_cand_p.shape[1], size=(nrepeat,nsamples))
        u_train = [u_cand_p[:, i] for i in idx]
    elif optimality:
        filename     = 'DoE_McsE5R0_d{:d}_p{:d}_{:s}.npy'.format(pce_model.ndim, pce_model.deg, optimality)
        oed_data_dir = 'MCS_OED' if sampling_method.lower().startswith('mcs') else 'CLS_OED'
        try:
            idx     = np.load(os.path.join(simparams.data_dir_sample, oed_data_dir, pce_model.basis.dist_name.capitalize(), filename))
            idx     = idx[:nsamples]
        except FileNotFoundError:
            print('Running museuq.experiment to get train data samples: {:s}-{:s} '.format(sampling_method, optimality))
            doe = museuq.OptimalDesign(optimality, curr_set=[])
            X   = pce_model.basis.vandermonde(u_cand_p)
            if sampling_method.lower().startswith('cls'):
                X  = pce_model.basis.num_basis**0.5*(X.T / np.linalg.norm(X, axis=1)).T
            idx = doe.samples(X, n_samples=nsamples, orth_basis=True)
        u_train     = [u_cand_p[:,idx],]

    return u_train

def get_test_data(simparams, u_test_p, pce_model, solver, sampling_method):

    if sampling_method.lower().startswith('mcs'):
        filename= r'DoE_McsE6R0.npy'
        try:
            data_set = np.load(os.path.join(simparams.data_dir_result, 'MCS', filename))
            y_test   = data_set[-1,:]
        except FileNotFoundError:
            print(' Running solver to get test data ')
            x_test = solver.map_domain(u_test_p, pce_model.basis.dist_u)
            y_test = solver.run(x_test)
            data   = np.vstack((u_test_p, x_test, y_test.reshape(1,-1)))
            np.save(os.path.join(simparams.data_dir_result, 'MCS', filename), data)

    elif sampling_method.lower().startswith('cls'):
        filename= r'DoE_McsE6d{:d}R0.npy'.format(solver.ndim) if pce_model.basis.dist_name.lower() == 'normal' else r'DoE_McsE6R0.npy'
        try:
            data_set = np.load(os.path.join(simparams.data_dir_result, 'Pluripotential', filename))
            y_test   = data_set[-1,:]
        except FileNotFoundError:
            print(' Running solver to get test data ')
            x_test = solver.map_domain(u_test_p, pce_model.basis.dist_u)
            y_test = solver.run(x_test)
            data   = np.vstack((u_test_p, x_test, y_test.reshape(1,-1)))
            np.save(os.path.join(simparams.data_dir_result, 'Pluripotential', filename), data)
    else:
        raise ValueError

    if np.isnan(y_test).any():
        raise ValueError('nan in y_test')
    return y_test

def main():

    ndim      = 2
    nrepeat   = 100
    n_splits  = 10
    p_orders  = np.array(np.arange(15,25,10))
    n_cand    = int(1e5)
    n_test    = -1 
    doe_method= 'CLS'
    optimality= None #'D', 'S', None
    fit_method= 'OLS'

    print(' Parameters:')
    print('------------------------------------------------------------')
    print(' - {:<25s} : {}'.format('Sampling method'  , doe_method))
    print(' - {:<25s} : {}'.format('Optimality '      , optimality))
    print(' - {:<25s} : {}'.format('Fitting method'   , fit_method))

    for p in p_orders:
        ## ------------------------  ----------------------- ###
        ### ============ Define solver ============
        # orth_poly   = museuq.Legendre(d=ndim, deg=p)
        orth_poly   = museuq.Hermite(d=ndim, deg=p, hem_type='physicists')
        # orth_poly   = museuq.Hermite(d=ndim, deg=p, hem_type='probabilists')
        solver      = museuq.sparse_poly(orth_poly, sparsity='full', seed=100)
        simparams = museuq.simParameters(solver.nickname)
        simparams.info()
        # print(solver.coef)

        ### ============ Candidate data set for DoE ============
        print(' > loading candidate data set...')
        u_cand, u_test = get_candidate_data(simparams, doe_method, orth_poly, n_cand, n_test)
        if doe_method.lower().startswith('cls') and orth_poly.dist_name.lower() == 'normal':
            u_cand_p = p**0.5 * u_cand
            # u_test_p = p**0.5 * u_test
        else:
            u_cand_p = u_cand
            # u_test_p = u_test

        
        ### ============ Define PCE ============
        # orth_poly   = museuq.Legendre(d=solver.ndim, deg=p)
        orth_poly   = museuq.Hermite(d=ndim, deg=p, hem_type='physicists')
        # orth_poly   = museuq.Hermite(d=ndim, deg=p, hem_type='probabilists')
        pce_model   = museuq.PCE(orth_poly)
        pce_model.info()
        


        ### ============ Number of samples based on oversampling ratio ============
        
        ### > 1. over_sampling_ratio nice to plot
        # Pmax= museuq.Hermite(d=ndim, deg=p_orders[-1]).num_basis 
        # P   = orth_poly.num_basis
        # nsamples1 = np.array([math.ceil(n) for n in np.linspace(1, 2, 11) *P]) 
        # nsamples2 = np.array([math.ceil(n) for n in np.linspace(2, 4, 11) *P])
        # nsamples3 = np.array([math.ceil(n) for n in np.linspace(4*P, 10*Pmax,7)])
        # nsamples_sets  = np.unique(np.hstack((nsamples1, nsamples2, nsamples3)))
        # over_sampling_ratio = nsamples_sets / P

        ### > 2. fixed over_sampling_ratio for each p
        over_sampling_ratio=[]
        over_sampling_ratio.append(np.linspace( 2, 2, 6)) 
        over_sampling_ratio.append(np.linspace( 2, 4,11))
        over_sampling_ratio.append(np.linspace( 4,20, 9))
        over_sampling_ratio = np.unique(np.hstack(over_sampling_ratio))
        # print(over_sampling_ratio)

        ### > 3. User defined over_sampling_ratio
        # over_sampling_ratio = np.array([1.2, 2.0])

        ### > 4. over_sampling_ratio for reference , 1.5 * P * log(P)  from christoffel  (CLS)
        # if doe_method.lower() == 'reference':
            # over_sampling_ratio = np.array([max(1, 1.5 * np.log(orth_poly.num_basis))])
        nsamples_sets = np.array([math.ceil(pce_model.num_basis * ialpha) for ialpha in over_sampling_ratio])
        # mse_global  = [] 
        # mse_exceed  = []
        cond_number = []
        rate_coef   = []
        print(' >>> ndim = {:d}, polynomial degree = {:d}, # simulation sets = {:d}, \n alpha = {}'.format(solver.ndim, p, len(nsamples_sets), over_sampling_ratio ))
        for i, nsamples in enumerate(nsamples_sets):

            print(' > Alpha = {:.2f}, {:d}/{:d}'.format(over_sampling_ratio[i], i+1, len(over_sampling_ratio)))
            ### 2. Get train data set

            print(' - Getting sample points ...', end='')
            u_train = get_train_data(simparams, doe_method, optimality, pce_model, nsamples, u_cand_p, nrepeat)
            print('    --> New samples: {:s} {}, #{:d}'.format(doe_method, optimality, nsamples))

            ### ============ Testing ============
            # y_test  = get_test_data(simparams, u_test_p, pce_model, solver, doe_method) 
            # y0      = np.sort(y_test)[-math.ceil(len(y_test)*0.01)] ## threshold value for y, top 1%

            # mse_exceed_   = []
            # mse_global_   = []
            cond_number_  = []
            coef_inf_norm_= []
            for iu_train in tqdm(u_train, ascii=True, desc='   repeat:'):
                ### 3. train model 
                y_train = solver.run(iu_train) 
                U_train = pce_model.basis.vandermonde(iu_train)
                # U_train = orth_poly.vandermonde(iu_train)
                if doe_method.lower().startswith('cls') or doe_method.lower() == 'reference':
                    # WU_train = pce_model.num_basis**0.5*(U_train.T / np.linalg.norm(U_train, axis=1)).T
                    ### reproducing kernel
                    Kp = np.sum(U_train * U_train, axis=1)
                    w =  np.sqrt(pce_model.num_basis / Kp)
                    WU_train = (U_train.T * w).T
                else:
                    WU_train = U_train
                    w = None

                ## condition number, kappa = max(svd)/min(svd)
                _, s, _ = np.linalg.svd(WU_train)
                kappa = max(abs(s)) / min(abs(s)) 

                if fit_method.lower() == 'ols':
                    pce_model.fit_ols(iu_train,y_train, w=w, n_splits=n_splits)
                elif fit_method.lower() == 'lassolars':
                    pce_model.fit_lassolars(iu_train, y_train, w=w)
                else:
                    raise ValueError

                ### 4. prediction 
                ## 4.1. calculate test data set 
                # y_pred  = pce_model.predict(u_test_p)
                coef_inf_norm_.append(sparse_poly_coef_error(solver, pce_model, normord=np.inf))
                # mse_global_.append(metrics.mean_squared_error(y_test, y_pred))
                ## exceeding mse
                # exceed_idx = np.argwhere(y_test >= y0)
                # mse_exceed_.append(metrics.mean_squared_error(y_test[exceed_idx], y_pred[exceed_idx]))
                cond_number_.append(kappa)

            # print(np.linalg.norm(pce_model.coef))
            rate_coef.append(np.mean(np.array(coef_inf_norm_)<1e-6))
            # mse_global.append(np.mean(np.array(mse_global_)))
            # mse_exceed.append(np.mean(np.array(mse_exceed_)))
            cond_number.append(np.mean(np.array(cond_number_)))
            print(' >>> Returns:')
            print('{:<15s} : {}'.format( 'Coef Recovery rate:', rate_coef))
            # print('{:<15s} : {}'.format( 'Global MSE', mse_global))
            # print('{:<15s} : {}'.format( 'Exceed MSE', mse_exceed))
            print('{:<15s} : {}'.format( 'Condition Num:', cond_number))

  

        data = np.vstack((nsamples_sets,  rate_coef, cond_number))
        if optimality:
            filename = 'Stability_{:s}_d{:d}_p{:d}_{:s}{:s}'.format(pce_model.basis.nickname, pce_model.ndim,pce_model.deg, doe_method.capitalize(), optimality)
        else:
            filename = 'Stability_{:s}_d{:d}_p{:d}_{:s}'.format(pce_model.basis.nickname, pce_model.ndim,pce_model.deg, doe_method.capitalize())
        print('>>> Saving result data to: \n{:s}'.format(simparams.data_dir_result))
        np.save(os.path.join(simparams.data_dir_result, filename), data)

if __name__ == '__main__':
    main()
