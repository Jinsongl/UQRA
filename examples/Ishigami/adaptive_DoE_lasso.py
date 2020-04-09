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
import numpy as np, os, sys
import collections
import scipy.stats as stats
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()

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

def get_candidate_data(simparams, doe_method, orth_poly, n_cand, n_test):
    """
    Return canndidate samples in u space
    """
    if doe_method.lower().startswith('mcs'):
        filename= r'DoE_McsE6R0.npy'
        mcs_data_set  = np.load(os.path.join(simparams.data_dir_sample, 'MCS', orth_poly.dist_name, filename))
        u_cand = mcs_data_set[:orth_poly.ndim,:n_cand].reshape(orth_poly.ndim, -1)
        u_test = mcs_data_set[:orth_poly.ndim,:n_test] if n_test > 1 else mcs_data_set[:orth_poly.ndim,:]

    elif doe_method.lower().startswith('cls') or doe_method.lower() == 'reference':
        filename= r'DoE_McsE6d{:d}R0.npy'.format(orth_poly.ndim) if orth_poly.dist_name.lower() == 'normal' else r'DoE_McsE6R0.npy'
        cls_data_set  = np.load(os.path.join(simparams.data_dir_sample, 'Pluripotential', orth_poly.dist_name, filename))
        u_cand = cls_data_set[:orth_poly.ndim,:n_cand].reshape(orth_poly.ndim, -1)
        u_test = cls_data_set[:orth_poly.ndim,:n_test] if n_test > 1 else cls_data_set[:orth_poly.ndim,:]
    else:
        raise ValueError

    return u_cand, u_test

def get_train_data(n, u_cand, doe_method, optimality=None, sample_selected=[], basis=None, active_basis=None):
    """
    Return train data from candidate data set. All sampels are in U-space

    Arguments:
        n           : int, number of samples 
        u_cand      : ndarray, candidate samples in U-space to be chosen from
        doe_method  : sampling strageties to get these sampels, e.g., MCS, CLS, LHS, etc..
        optimality  : Optimality criteria, 'S', 'D', None, default None
        sample_selected: samples already selected, need to be removed from candidate set to get n sampels
        basis       : When optimality is 'D' or 'S', one need the design matrix in the basis selected
        active_basis: activated basis used in optimality design

    """

    if optimality is None:
        idx = list(set(np.random.randint(0, u_cand.shape[1], size=n*10)).difference(set(sample_selected)))
        samples_new = idx[:n]
        sample_selected += samples_new

    elif optimality:
        doe = museuq.OptimalDesign(optimality, curr_set=sample_selected)
        if basis is None:
            raise ValueError('For optimality design, basis function are needed')
        else:
            X  = basis.vandermonde(u_cand)
        if doe_method.lower().startswith('cls'):
            X  = basis.num_basis**0.5*(X.T / np.linalg.norm(X, axis=1)).T
        if active_basis == 0 or active_basis is None:
            X = X
        else:
            X = X[:, active_basis]
        samples_new = doe.samples(X, n_samples=n, orth_basis=True)

    if len(sample_selected) != len(np.unique(sample_selected)):
        raise ValueError('get_train_data: duplciate samples ')
    if len(sample_selected) == 0:
        raise ValueError('get_train_data: No samples are selected')
    u_train_new = u_cand[:,samples_new]
    return u_train_new

def map_domain(u_data, solver, doe_method, dist_name):
    """
    Map variables from U-space to X-space

    """
    if doe_method.lower().startswith('mcs'):
        if dist_name.lower() == 'uniform':
            dist_u = [stats.uniform(-1,2),] * solver.ndim
        elif dist_name.lower() == 'normal':
            dist_u = [stats.norm(0,1), ] *solver.ndim
        else:
            raise ValueError('{:s} not defined'.format(dist_name))
        x_data = solver.map_domain(u_data, dist_u)
    elif doe_method.lower().startswith('cls'):
        if dist_name.lower() == 'uniform':
            x_data= solver.map_domain(u_data, np.arcsin(u_data)/np.pi + 0.5)
        elif dist_name.lower() == 'normal':
            raise NotImplementedError
        else:
            raise ValueError('{:s} not defined'.format(dist_name))
    return x_data

def get_test_data(simparams, u_test, solver, doe_method, dist_name):

    if doe_method.lower().startswith('mcs'):
        filename= r'DoE_McsE6R0.npy'
        try:
            data_set = np.load(os.path.join(simparams.data_dir_result, 'MCS', filename))
            y_test   = data_set[-1,:]
        except FileNotFoundError:
            print('   > Running solver to get test data ')
            if dist_name.lower() == 'uniform':
                dist_u = [stats.uniform(-1,2),] * solver.ndim
            elif dist_name.lower() == 'normal':
                dist_u = [stats.norm(0,1), ] *solver.ndim
            else:
                raise ValueError('{:s} not defined'.format(dist_name))
            x_test = solver.map_domain(u_test, dist_u)
            y_test = solver.run(x_test)
            data   = np.vstack((u_test, x_test, y_test.reshape(1,-1)))
            np.save(os.path.join(simparams.data_dir_result, 'MCS', filename), data)

    elif doe_method.lower().startswith('cls'):
        filename= r'DoE_McsE6R0.npy'
        try:
            data_set = np.load(os.path.join(simparams.data_dir_result, 'Pluripotential', filename))
            y_test   = data_set[-1,:]
        except FileNotFoundError:
            print('   > Running solver to get test data ')
            if dist_name.lower() == 'uniform':
                x_test = solver.map_domain(u_test, np.arcsin(u_test)/np.pi + 0.5)
            elif dist_name.lower() == 'normal':
                raise NotImplementedError
            else:
                raise ValueError('{:s} not defined'.format(dist_name))
            y_test = solver.run(x_test)
            data   = np.vstack((u_test, x_test, y_test.reshape(1,-1)))
            np.save(os.path.join(simparams.data_dir_result, 'Pluripotential', filename), data)
    else:
        raise ValueError
    return y_test

def cal_weight(doe_method, u_data, pce_model):
    X = pce_model.basis.vandermonde(u_data)
    if doe_method.lower().startswith('cls'):
        ### reproducing kernel
        Kp = np.sum(X* X, axis=1)
        w =  np.sqrt(pce_model.num_basis / Kp)
    else:
        w = None
    return w

def main():

    ## ------------------------ Displaying set up ------------------- ###
    np.set_printoptions(precision=4)
    np.set_printoptions(threshold=8)
    np.set_printoptions(suppress=True)
    n_new       = 5
    n_eval_init = 15
    iter_max    = 100

    ## ------------------------ Define solver ----------------------- ###
    pf          = 1e-4
    solver      = museuq.Ishigami()
    simparams   = museuq.simParameters(solver.nickname)
    # print(simparams.data_dir_result)
    ## ------------------------ Adaptive parameters ----------------- ###
    plim        = (2,100)
    n_budget    = 200
    n_cand      = int(1e5)
    n_test      = -1 
    doe_method  = 'MCS'
    optimality  = None #'D', 'S', None
    fit_method  = 'LASSOLARS'
    simparams.set_adaptive_parameters(n_budget=n_budget, plim=plim, abs_qoi=0.01)
    simparams.info()
    print('   * Sampling and Fitting:')
    print('     - {:<23s} : {}'.format('Sampling method'  , doe_method  ))
    print('     - {:<23s} : {}'.format('Optimality '      , optimality))
    print('     - {:<23s} : {}'.format('Fitting method'   , fit_method))

    ## ------------------------ Define PCE model --------------------- ###
    orth_poly   = museuq.Legendre(d=solver.ndim)
    pce_model   = museuq.PCE(orth_poly)
    pce_model.info()

    ## ----------- Candidate and testing data set for DoE ----------- ###
    print(' > Getting candidate data set...')
    u_cand, u_test  = get_candidate_data(simparams, doe_method, orth_poly, n_cand, n_test)
    y_test          = get_test_data(simparams, u_test, solver, doe_method, orth_poly.dist_name) 
    qoi             = museuq.metrics.mquantiles(y_test, 1-pf)
    print('   * {:<25s} : {}'.format('Candidate', u_cand.shape))
    print('   * {:<25s} : {}'.format('Test'     , u_test.shape))
    print('   * {:<25s} : {}'.format('Target QoI',qoi))

    print('utest: \n{}'.format(u_test[:,:3]))
    print('max y_test :{}'.format(max(y_test)))

    ## Here selecte the initial samples
    print(' > Getting initial sample set...')
    init_basis_deg  = 10
    sample_selected = []
    init_doe_method = doe_method
    init_optimality = None
    orth_poly.set_degree(init_basis_deg)

    if doe_method.lower().startswith('cls') and orth_poly.dist_name.lower() == 'normal':
        u_cand_p = init_basis_deg **0.5 * u_cand
    else:
        u_cand_p = u_cand

    u_train = get_train_data(n_eval_init, u_cand_p, init_doe_method, optimality=init_optimality, sample_selected=sample_selected, basis=orth_poly)

    x_train = map_domain(u_train, solver, doe_method, orth_poly.dist_name)
    y_train = solver.run(x_train)
    print('   * {:<25s} : {}'.format(' doe_method ', init_doe_method))
    print('   * {:<25s} : {}'.format(' Optimality ', init_optimality))
    print('   * {:<25s} : {}'.format(' Basis deg', init_basis_deg))
    print('   * {:<25s} : {}'.format(' # samples ', u_train.shape[1]))
    print('   * {:<25s} : [{:.2f},{:.2f}]'.format(' u Domain ', np.amin(u_train), np.amax(u_train)))
    print('   * {:<25s} : [{:.2f},{:.2f}]'.format(' x Domain ', np.amin(x_train), np.amax(x_train)))
    # print('   * {:<25s} : {}'.format('Target QoI',qoi))

    ### ============ Stopping Criteria ============
    ## path values recording all the values calcualted in the adaptive process, including those removed bc overfitting

    n_eval_path     = []
    poly_order_path = []
    cv_error        = [0,] * (plim[1]+1)
    cv_error_path   = []
    adj_r2          = [0,] * (plim[1]+1)
    adj_r2_path     = []
    test_error      = [0,] * (plim[1]+1)
    test_error_path = []
    QoI             = [0,] * (plim[1]+1)
    QoI_path        = []
    active_basis    = [0,] * (plim[1]+1)
    active_basis_path= [] 



    ### update candidate data set for this p degree, cls unbuounded
    ### ============ Initial Values ============
    i_iteration = 0
    p           = plim[0] 

    ### ============ Start adaptive iteration ============
    print(' > Starting iteration ...')
    while i_iteration < iter_max:
        print(' >>> Iteration No: {:d}'.format(i_iteration))

        ### ============ Update PCE model ============
        orth_poly.set_degree(p)
        pce_model = museuq.PCE(orth_poly)

        if doe_method.lower().startswith('cls') and orth_poly.dist_name.lower() == 'normal':
            u_cand_p = p**0.5 * u_cand
        else:
            u_cand_p = u_cand

        ### ============ Get training points ============
        print(' - Getting new samples ({:s} {}) '.format(doe_method, optimality))
        u_train_new = get_train_data(n_new,  u_cand_p,doe_method, optimality, sample_selected, pce_model.basis, active_basis[p])
        x_train_new = map_domain(u_train_new, solver, doe_method, orth_poly.dist_name)
        y_train_new = solver.run(x_train_new)
        u_train = np.hstack((u_train, u_train_new)) 
        x_train = np.hstack((x_train, x_train_new)) 
        y_train = np.hstack((y_train, y_train_new)) 
        w       = cal_weight(doe_method, u_train, pce_model)
        print('   New samples shape: {}, total iteration samples: {:d}'.format(u_train_new.shape, len(sample_selected)))

        ### ============ Build Surrogate Model ============
        # print('u train min: {}'.format(np.min(u_train, axis=1)))
        # print('u train max: {}'.format(np.max(u_train, axis=1)))
        # print('x train min: {}'.format(np.min(x_train, axis=1)))
        # print('x train max: {}'.format(np.max(x_train, axis=1)))
        # print('y train: {}'.format(y_train))
        # print('w train: {}'.format(w))

        pce_model.fit_lassolars(u_train, y_train, w=w)
        y_train_hat = pce_model.predict(u_train, w=w)

        w = cal_weight(doe_method, u_test, pce_model)
        y_test_hat  = pce_model.predict(u_test, w=w)
        qoi = museuq.metrics.mquantiles(y_test_hat, 1-pf)
        print('utest: {}'.format(u_test[:,:3]))
        print('u test min: {}'.format(np.min(u_test, axis=1)))
        print('u test max: {}'.format(np.max(u_test, axis=1)))
        print('y test max: {}'.format(np.max(y_test)))
        print('pf, y_test_hat: {}'.format(qoi))
        print('y_test_hat max: {}'.format(max(y_test_hat)))

        ### ============ calculating & updating metrics ============
       
        n_eval_path.append(u_train.shape[1])
        poly_order_path.append(p)
        
        cv_error[p] = pce_model.cv_error        
        cv_error_path.append(pce_model.cv_error)
        active_basis[p] = pce_model.active_        
        active_basis_path.append(pce_model.active_)
        adj_r2[p] = museuq.metrics.r2_score_adj(y_train, y_train_hat, pce_model.num_basis)        
        adj_r2_path.append(museuq.metrics.r2_score_adj(y_train, y_train_hat, pce_model.num_basis))
        qoi = museuq.metrics.mquantiles(y_test_hat, 1-pf)
        QoI[p] = qoi
        QoI_path.append(qoi)

        test_error[p] = museuq.metrics.mean_squared_error(y_test, y_test_hat)
        test_error_path.append(museuq.metrics.mean_squared_error(y_test, y_test_hat))

        ### ============ Cheking Overfitting ============
        if simparams.check_overfitting(cv_error[plim[0]:p+1]):
            print('     >>> Possible overfitting detected')
            print('         cv error: {}'.format(cv_error[plim[0]:p+1]))
            p = plim[0] + np.argmin(cv_error[plim[0]:p+1])
            # p = max(plim[0], p -3)
            print('         Reseting results for PCE order higher than p = {:d} '.format(p))
            for i in range(p+1, len(active_basis)):
                QoI[i]          = 0
                adj_r2[i]       = 0
                cv_error[i]     = 0
                test_error[i]   = 0
                active_basis[i] = 0
            continue

        print(' - {:<25s} : {}'.format('Polynomial order (p)', p))
        print(' - {:<25s} : {}'.format('# samples ', n_eval_path[-1]))
        if pce_model:
            # print(' - {:<25s} : {} -> #{:d}'.format('Active basis', pce_model.active_, len(pce_model.active_)))
            beta = pce_model.coef
            beta = beta[abs(beta) > 1e-6]
            print(' - {:<25s} : #{:d}'.format('Active basis', len(beta)))
        # print(' - {:<25s} : {}'.format('R2_adjusted ', np.around(adj_r2, 2)))
        print(' - {:<25s} : {}'.format('cv error ', np.squeeze(np.array(cv_error[plim[0]:p+1]))))
        print(' - {:<25s} : {}'.format('QoI', np.around(np.squeeze(np.array(QoI[plim[0]:p+1])), 2)))
        print(' - {:<25s} : {}'.format('test error', np.squeeze(np.array(test_error[plim[0]:p+1]))))
        print('     ------------------------------------------------------------')

        ### ============ updating parameters ============
        p +=1
        i_iteration += 1
        if not simparams.is_adaptive_continue(n_eval_path[-1], p, qoi=QoI[plim[0]:p]):
            break


    print('------------------------------------------------------------')
    print('>>>>>>>>>>>>>>> Adaptive simulation done <<<<<<<<<<<<<<<<<<<')
    print('------------------------------------------------------------')
    print(' - {:<25s} : {}'.format('Polynomial order (p)', p))
    print(' - {:<25s} : {} -> #{:d}'.format('Active basis', pce_model.active_, len(pce_model.active_)))
    print(' - {:<25s} : {}'.format('# samples', n_eval_path[-1]))
    # print(' - {:<25s} : {}'.format('R2_adjusted ', np.around(adj_r2, 2)))
    print(' - {:<25s} : {}'.format('QoI', np.around(np.squeeze(np.array(QoI[plim[0]:p+1], dtype=np.float)), 2)))
    # print(np.linalg.norm(pce_model.coef - solver.coef, np.inf))
    print(pce_model.coef[pce_model.coef!=0])
    # print(solver.coef[solver.coef!=0])

    if optimality:
        filename = 'Adaptive_{:s}_{:s}{:s}_{:s}.npy'.format(solver.nickname.capitalize(), doe_method.capitalize(), optimality, fit_method.capitalize())
    else:
        filename = 'Adaptive_{:s}_{:s}_{:s}.npy'.format(solver.nickname.capitalize(), doe_method.capitalize(), fit_method.capitalize())
    data  = np.array([n_eval_path, poly_order_path, cv_error_path, active_basis_path, adj_r2_path, QoI_path, test_error_path]) 
    np.save(os.path.join(simparams.data_dir_result, filename), data)

if __name__ == '__main__':
    main()
