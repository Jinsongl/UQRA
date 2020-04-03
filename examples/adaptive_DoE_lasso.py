#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import museuq, warnings
import numpy as np, chaospy as cp, os, sys
from tqdm import tqdm
import scipy.stats as stats
import collections

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()
def sparse_poly_coef_error(solver, model):
    beta    = np.array(solver.coef, copy=True)
    beta_hat= np.array(model.coef, copy=True)

    solver_basis_degree = solver.poly.basis_degree
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
    error_common = np.linalg.norm(large_beta[basis_common]-small_beta, np.inf)
    large_beta[basis_common] = 0
    error_left   =  max(abs(large_beta))
    return  max( error_common, error_left )


def get_candidate_data(simparams, sampling_method, orth_poly, n_cand, n_test):
    """
    Return canndidate samples in u space
    """
    if sampling_method.lower().startswith('mcs'):
        filename= r'DoE_McsE6R0.npy'
        mcs_data_set  = np.load(os.path.join(simparams.data_dir_sample, 'MCS', orth_poly.dist_name, filename))
        u_cand = mcs_data_set[:orth_poly.ndim,:n_cand].reshape(orth_poly.ndim, -1)
        u_test = mcs_data_set[:orth_poly.ndim,:n_test] if n_test > 1 else mcs_data_set[:orth_poly.ndim,:]

    elif sampling_method.lower().startswith('cls'):
        filename= r'DoE_McsE6d{:d}R0.npy'.format(orth_poly.ndim) if orth_poly.dist_name.lower() == 'normal' else r'DoE_McsE6R0.npy'
        cls_data_set  = np.load(os.path.join(simparams.data_dir_sample, 'Pluripotential', orth_poly.dist_name, filename))
        u_cand = cls_data_set[:orth_poly.ndim,:n_cand].reshape(orth_poly.ndim, -1)
        u_test = cls_data_set[:orth_poly.ndim,:n_test] if n_test > 1 else cls_data_set[:orth_poly.ndim,:]
    else:
        raise ValueError

    return u_cand, u_test

def get_train_data(sampling_method, optimality, sample_selected, pce_model, active_basis, nsamples, u_cand_p):


        if sampling_method.lower() in ['mcs', 'cls'] or optimality is None:
            idx = list(set(np.random.randint(0, u_cand_p.shape[1], size=nsamples*10)).difference(set(sample_selected)))
            samples_new = idx[:nsamples]
            sample_selected += samples_new

        elif optimality:
            doe = museuq.OptimalDesign(optimality, curr_set=sample_selected)
            X   = pce_model.basis.vandermonde(u_cand_p)
            if sampling_method.lower().startswith('cls'):
                X  = pce_model.basis.num_basis**0.5*(X.T / np.linalg.norm(X, axis=1)).T

            X   = X[:, active_basis[-1]] if len(active_basis) else X
            samples_new = doe.samples(X, n_samples=nsamples, orth_basis=True)

        u_train_new = u_cand_p[:,samples_new]
        return u_train_new

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
    return y_test

def main():
    ## ------------------------ Parameters set-up ----------------------- ###
    pf          = 1e-4
    ## ------------------------ Define solver ----------------------- ###
    ndim        = 2
    orth_poly   = museuq.Legendre(d=ndim, deg=15)
    # orth_poly   = museuq.Hermite(d=ndim, deg=15, hem_type='physicists')
    # orth_poly   = museuq.Hermite(d=ndim, deg=15, hem_type='probabilists')
    solver      = museuq.sparse_poly(orth_poly, sparsity=5, seed=100)
    orth_poly   = museuq.Legendre(d=solver.ndim)
    # orth_poly   = museuq.Hermite(d=solver.ndim)

    # solver      = museuq.Ishigami()

    simparams   = museuq.simParameters(solver.nickname)

    ### ============ Adaptive parameters ============
    plim        = (2,100)
    n_budget    = 2000 
    n_cand      = int(1e5)
    n_test      = -1 
    sampling    = 'CLS'
    optimality  = None #'D'
    fit_method  = 'LASSOLARS'
    k_sparsity  = 25 # guess. K sparsity to meet RIP condition 
    simparams.set_adaptive_parameters(n_budget=n_budget, plim=plim, min_r2=0.9, qoi_val=0.0001)
    simparams.info()

    orth_poly   = museuq.Hermite(d=ndim, hem_type='physicists')
    orth_poly   = museuq.Hermite(d=ndim, hem_type='probabilists')
    orth_poly   = museuq.Legendre(d=solver.ndim)
    pce_model   = museuq.PCE(orth_poly)

    print(' Parameters:')
    print('------------------------------------------------------------')
    print(' - {:<25s} : {}'.format('Sampling method'  , sampling  ))
    print(' - {:<25s} : {}'.format('Optimality '      , optimality))
    print(' - {:<25s} : {}'.format('Fitting method'   , fit_method))
    print(' - {:<25s} : {}'.format('Simulation budget', n_budget  ))
    print(' - {:<25s} : {}'.format('Poly degree limit', plim      ))
    ### ============ Initial Values ============
    p_iter_0    = 10
    n_new       = 5
    n_eval_init = 20
    n_eval_init = max(n_eval_init, 2 * k_sparsity) ## for ols, oversampling rate at least 2
    i_iteration = -1
    sample_selected = []
    u_train = np.array([])
    x_train = np.array([])
    y_train = np.array([])

    ### ============ Stopping Criteria ============
    ## path values recording all the values calcualted in the adaptive process, including those removed bc overfitting

    n_eval_path     = []
    poly_order_path = []
    cv_error        = []
    cv_error_path   = []
    adj_r2        = []
    adj_r2_path   = []
    test_error      = []
    test_error_path = []
    QoI             = []
    QoI_path        = []
    active_basis    = collections.deque(maxlen=3)
    active_basis_path= [] 


    ### ============ Candidate data set for DoE ============
    print(' > loading candidate data set...')
    u_cand, u_test = get_candidate_data(simparams, sampling, orth_poly, n_cand, n_test)

    ### ============ Start adaptive iteration ============
    print(' > Starting iteration ...')
    while True:
        i_iteration += 1
        print(' >>> Iteration No: {:d}'.format(i_iteration))
        p = p_iter_0 if i_iteration == 0 else p
        ### ============ Redefine PCE model ============
        orth_poly.set_degree(p)
        pce_model = museuq.PCE(orth_poly)

        ### update candidate data set for this p degree, cls unbuounded
        if sampling.lower().startswith('cls') and orth_poly.dist_name.lower() == 'normal':
            u_cand_p = p**0.5 * u_cand
            u_test_p = p**0.5 * u_test
        else:
            u_cand_p = u_cand
            u_test_p = u_test

        ### ============ Get training points ============
        nsamples = n_new if i_iteration else n_eval_init
        print(' - Getting sample points ...', end='')
        u_train_new = get_train_data(sampling, optimality, sample_selected, pce_model, active_basis, nsamples, u_cand_p)
        print('    --> New samples: {:s} {}, #{:d}'.format(sampling, optimality, nsamples))
        ## need to check if sample_selected will be updated by reference
        if len(sample_selected) != len(np.unique(sample_selected)):
            print('sample_selected len: {}'.format(len(sample_selected)))

        x_train_new = solver.map_domain(u_train_new, pce_model.basis.dist_u)
        y_train_new = solver.run(x_train_new)

        u_train = np.hstack((u_train, u_train_new)) if u_train.any() else u_train_new
        x_train = np.hstack((x_train, x_train_new)) if x_train.any() else x_train_new
        y_train = np.hstack((y_train, y_train_new)) if y_train.any() else y_train_new

        if len(sample_selected) != len(np.unique(sample_selected)):
            print(len(sample_selected))
            print(len(np.unique(sample_selected)))
            raise ValueError('Duplciate samples')

        if i_iteration == 0:
            ### 0 iteration only initialize the samples, no fitting is need to be done 
            p = plim[0]
            continue
        ### ============ Build Surrogate Model ============

        U_train = pce_model.basis.vandermonde(u_train)
        if sampling.lower().startswith('cls'):
            ### reproducing kernel
            Kp = np.sum(U_train * U_train, axis=1)
            w =  np.sqrt(pce_model.num_basis / Kp)
        else:
            w = None

        pce_model.fit_lassolars(u_train, y_train, w=w)
        y_train_hat = pce_model.predict(u_train)


        ### ============ Testing ============
        y_test      = get_test_data(simparams, u_test_p, pce_model, solver, sampling) 
        y_test_hat  = pce_model.predict(u_test_p)

        ### ============ calculating & updating metrics ============
        qoi = sparse_poly_coef_error(solver, pce_model)
        n_eval_path.append(u_train.shape[1])
        poly_order_path.append(p)
        cv_error_path.append(pce_model.cv_error)
        active_basis_path.append(pce_model.active_)
        adj_r2_path.append(museuq.metrics.r2_score_adj(y_train, y_train_hat, pce_model.num_basis))
        QoI_path.append(qoi)
        test_error_path.append(museuq.metrics.mean_squared_error(y_test, y_test_hat))



        cv_error.append(pce_model.cv_error)
        active_basis.append(pce_model.active_)
        adj_r2.append(museuq.metrics.r2_score_adj(y_train, y_train_hat, pce_model.num_basis))
        # QoI.append(museuq.metrics.mquantiles(y_test_hat, 1-pf))
        QoI.append(qoi)
        test_error.append(museuq.metrics.mean_squared_error(y_test, y_test_hat))

        ### ============ Cheking Overfitting ============
        if simparams.check_overfitting(cv_error):
            print('     - Possible overfitting detected, setting p = p - 1')
            while len(active_basis) > 1:
                p -= 1
                active_basis.pop()
                cv_error.pop()
                adj_r2.pop()
                QoI.pop()
                test_error.pop()
            continue

        print(' - {:<25s} : {}'.format('Polynomial order (p)', p))
        print(' - {:<25s} : {}'.format('Evaluations done ', n_eval_path[-1]))
        if pce_model:
            print(' - {:<25s} : {} -> #{:d}'.format('Active basis', pce_model.active_, len(pce_model.active_)))
            beta = pce_model.coef
            beta = beta[abs(beta) > 1e-6]
            print(' - {:<25s} : #{:d}'.format('Active basis', len(beta)))
        print(' - {:<25s} : {}'.format('R2_adjusted ', np.around(adj_r2, 2)))
        print(' - {:<25s} : {}'.format('QoI', np.around(np.squeeze(np.array(QoI)), 2)))
        print(' - {:<25s} : {}'.format('test error', np.squeeze(np.array(test_error))))
        print('     ------------------------------------------------------------')

        ### ============ updating parameters ============
        p +=1
        if not simparams.is_adaptive_continue(n_eval_path[-1], p, adj_r2=adj_r2, qoi=QoI):
            break

    print('------------------------------------------------------------')
    print('>>>>>>>>>>>>>>> Adaptive simulation done <<<<<<<<<<<<<<<<<<<')
    print('------------------------------------------------------------')
    print(' - {:<25s} : {}'.format('Polynomial order (p)', p))
    print(' - {:<25s} : {} -> #{:d}'.format('Active basis', pce_model.active_, len(pce_model.active_)))
    print(' - {:<25s} : {}'.format('# Evaluations ', n_eval_path[-1]))
    print(' - {:<25s} : {}'.format('R2_adjusted ', np.around(adj_r2, 2)))
    print(' - {:<25s} : {}'.format('QoI', np.around(np.squeeze(np.array(QoI)), 2)))
    # print(np.linalg.norm(pce_model.coef - solver.coef, np.inf))
    print(pce_model.coef)
    print(solver.coef)

    # print(np.array(QoI).shape)
    # print(np.array(adj_r2).shape)
    # print(np.array(cv_error).shape)
    # print(np.array(n_eval_path).shape)

    if optimality:
        filename = 'Adaptive_{:s}_{:s}_{:s}_{:s}.npy'.format(solver.nickname.capitalize(), sampling.capitalize(), optimality, fit_method.capitalize())
    else:
        filename = 'Adaptive_{:s}_{:s}_{:s}.npy'.format(solver.nickname.capitalize(), sampling.capitalize(), fit_method.capitalize())
    data  = np.array([n_eval_path, poly_order_path, cv_error_path, active_basis_path, adj_r2_path, QoI_path, test_error_path]) 
    print(data.shape)
    np.save(filename, data)
    # np.save(os.path.join(simparams.data_dir_result, filename), np.array(QoI))



    # filename = 'mquantile_DoE_q15_OptD2040_PCE{:d}_{:s}_path.npy'.format(p, fit_method)
    # np.save(os.path.join(simparams.data_dir, filename), np.array(QoI))

    # filename = 'r2_DoE_q15_OptD2040_PCE{:d}_{:s}_path.npy'.format(p, fit_method)
    # np.save(os.path.join(simparams.data_dir, filename), np.array(adj_r2))

    # filename = 'cv_error_DoE_q15_OptD2040_PCE{:d}_{:s}_path.npy'.format(p, fit_method)
    # np.save(os.path.join(simparams.data_dir, filename), np.array(cv_error))

    # filename = 'n_eval_DoE_q15_OptD2040_PCE{:d}_{:s}_path.npy'.format(p, fit_method)
    # np.save(os.path.join(simparams.data_dir, filename), np.array(n_eval_path))

    # QoI = []
    # for r in tqdm(range(10), ascii=True, desc="   - " ):
        # filename = 'DoE_McsE6R{:d}.npy'.format(r)
        # data_set = np.load(os.path.join(simparams.data_dir, filename))
        # u_samples= data_set[0:ndim,:]
        # x_samples= data_set[ndim: 2*ndim,:]
        # y_samples= pce_model.predict(u_samples)
        # QoI.append(museuq.metrics.mquantiles(y_samples, [1-1e-4, 1-1e-5, 1-1e-6]))
        # filename = 'DoE_McsE6R{:d}_q15_OptD2040_PCE{:d}_{:s}.npy'.format(r, p, fit_method)
        # np.save(os.path.join(simparams.data_dir, filename), y_samples)

    # filename = 'mquantile_DoE_q15_OptD2040_PCE{:d}_{:s}.npy'.format(p, fit_method)
    # np.save(os.path.join(simparams.data_dir, filename), np.array(QoI))


if __name__ == '__main__':
    main()
