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
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()

def get_candidate_data(simparams, doe_method, orth_poly, n_cand, n_test):
    """
    Return canndidate samples in u space
    """
    if orth_poly.dist_name == 'norm':
        dist_name = 'Normal' 
    elif orth_poly.dist_name == 'uniform':
        dist_name = 'Uniform' 
    else:
        raise ValueError
    if doe_method.lower().startswith('mcs'):
        filename= r'DoE_McsE6R0.npy'
        mcs_data_set  = np.load(os.path.join(simparams.data_dir_sample, 'MCS', dist_name, filename))
        u_cand = mcs_data_set[:orth_poly.ndim,:n_cand].reshape(orth_poly.ndim, -1)

    elif doe_method.lower().startswith('cls') or doe_method.lower() == 'reference':
        filename= r'DoE_McsE6d{:d}R0.npy'.format(orth_poly.ndim) if dist_name == 'Normal' else r'DoE_McsE6R0.npy'
        cls_data_set  = np.load(os.path.join(simparams.data_dir_sample, 'Pluripotential', dist_name, filename))
        u_cand = cls_data_set[:orth_poly.ndim,:n_cand].reshape(orth_poly.ndim, -1)
    else:
        raise ValueError

    return u_cand

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
        idx = []
        while len(idx) < n:
            idx += list(set(np.random.randint(0, u_cand.shape[1], size=n*10)).difference(set(sample_selected)).difference(set(idx)))
        samples_new = idx[:n]
        sample_selected += samples_new

    elif optimality:
        doe = museuq.OptimalDesign(optimality, curr_set=sample_selected)
        if basis is None:
            raise ValueError('For optimality design, basis function are needed')
        else:
            X  = basis.vandermonde(u_cand)
        if active_basis == 0 or active_basis is None:
            print('     - {:<23s} : {}/{}'.format('Optimal design based on ', basis.num_basis, basis.num_basis))
            X = X
        else:
            active_index = np.array([i for i in range(basis.num_basis) if basis.basis_degree[i] in active_basis])
            print('     - {:<23s} : {}/{}'.format('Optimal design based on ', len(active_index), basis.num_basis))
            X = X[:, active_index]
        if doe_method.lower().startswith('cls'):
            X  = basis.num_basis**0.5*(X.T / np.linalg.norm(X, axis=1)).T
        samples_new = doe.samples(X, n_samples=n, orth_basis=True)

    if len(sample_selected) != len(np.unique(sample_selected)):
        raise ValueError('get_train_data: duplciate samples ')
    if len(sample_selected) == 0:
        raise ValueError('get_train_data: No samples are selected')
    u_train_new = u_cand[:,samples_new]

    return u_train_new

def get_test_data(simparams, solver, pce_model, n_test, filename = r'DoE_McsE6R0.npy'):
    """
    Return test data. 

    Test data should always be in X-space. The correct sequence is X->y->u

    To be able to generate MCS samples for X, we use MCS samples in Samples/MCS, noted as z here

    If already exist in simparams.data_dir_result, then load and return
    else, run solver

    """
    
    if solver.nickname.lower().startswith('poly') or solver.nickname.lower().startswith('sparsepoly'):
        filename_result = filename[:-4]+'_{:s}{:d}_p{:d}.npy'.format(solver.basis.nickname, solver.ndim,solver.basis.deg)
    else:
        filename_result = filename

    data_dir_result = os.path.join(simparams.data_dir_result, 'MCS')
    try: 
        os.makedirs(data_dir_result)
    except OSError as e:
        pass

    if pce_model.basis.hem_type == 'physicists':
        filename_result = filename_result.replace('Mcs', 'Cls')

    try:
        data_set = np.load(os.path.join(data_dir_result, filename_result))
        print('   > Retrieving test data from {}'.format(os.path.join(data_dir_result, filename_result)))
        assert data_set.shape[0] == 2*solver.ndim+1
        u_test = data_set[:solver.ndim,-n_test:] if n_test > 0 else data_set[:solver.ndim,:]
        x_test = data_set[solver.ndim:2*solver.ndim,-n_test:] if n_test > 0 else data_set[solver.ndim:2*solver.ndim,:]
        y_test = data_set[-1,-n_test:] if n_test > 0 else data_set[-1,:]

    except FileNotFoundError:
        ### 1. Get MCS samples for X
        if solver.dist_name.lower() == 'uniform':
            print('   > Solving test data from {} '.format(os.path.join(simparams.data_dir_sample, 'MCS','Uniform', filename)))
            data_set = np.load(os.path.join(simparams.data_dir_sample, 'MCS','Uniform', filename))
            z_test = data_set[:solver.ndim,:] 
            x_test = solver.map_domain(z_test, [stats.uniform(-1,2),] * solver.ndim)
        elif solver.dist_name.lower().startswith('norm'):
            print('   > Solving test data from {} '.format(os.path.join(simparams.data_dir_sample, 'MCS','Normal', filename)))
            data_set = np.load(os.path.join(simparams.data_dir_sample, 'MCS','Normal', filename))
            z_test = data_set[:solver.ndim,:] 
            x_test = solver.map_domain(z_test, [stats.norm(0,1),] * solver.ndim)
        else:
            raise ValueError
        y_test = solver.run(x_test)

        ### 2. Mapping MCS samples from X to u
        ###     dist_u is defined by pce_model
        ### Bounded domain maps to [-1,1] for both mcs and cls methods. so u = z
        ### Unbounded domain, mcs maps to N(0,1), cls maps to N(0,sqrt(0.5))
        if pce_model.basis.hem_type == 'physicists':
            u_test = 0.0 + z_test * np.sqrt(0.5) ## mu + sigma * x
        else:
            u_test = z_test
        data   = np.vstack((u_test, x_test, y_test.reshape(1,-1)))
        np.save(os.path.join(data_dir_result, filename_result), data)

        u_test = u_test[:,-n_test:] if n_test > 0 else u_test
        x_test = x_test[:,-n_test:] if n_test > 0 else x_test
        y_test = y_test[:,-n_test:] if n_test > 0 else y_test

    return u_test, x_test, y_test

def get_init_samples(n, solver, pce_model, doe_method='lhs', random_state=None, **kwargs):

    if doe_method.lower() == 'lhs':
        doe = museuq.LHS(pce_model.basis.dist_u)
        z, u= doe.samples(n, random_state=random_state)
    else:
        raise NotImplementedError
    x = solver.map_domain(u, z) ## z_train is the cdf of u_train
    y = solver.run(x)
    return u, x, y

def cal_weight(doe_method, u_data, pce_model):
    """
    Calculate weight for CLS based on Christoffel function evaluated in U-space
    """
    X = pce_model.basis.vandermonde(u_data)
    if doe_method.lower().startswith('cls'):
        ### reproducing kernel
        Kp = np.sum(X* X, axis=1)
        w  = pce_model.num_basis / Kp
    else:
        w = None 
    return w

def main():

    ## ------------------------ Displaying set up ------------------- ###
    np.set_printoptions(precision=4)
    np.set_printoptions(threshold=1000)
    np.set_printoptions(suppress=True)
    n_new       = 5
    iter_max    = 100

    ## ------------------------ Define solver ----------------------- ###
    pf          = 1e-4
    solver      = museuq.four_branch_system()
    simparams   = museuq.simParameters(solver.nickname)
    # print(simparams.data_dir_result)
    ## ------------------------ Adaptive parameters ----------------- ###
    plim        = (2,100)
    n_budget    = 2000
    n_cand      = int(1e5)
    n_test      = -1 
    doe_method  = 'CLS'
    optimality  = 'S'#'D', 'S', None
    fit_method  = 'LASSOLARS'
    simparams.set_adaptive_parameters(n_budget=n_budget, plim=plim, abs_qoi=0.02, min_r2=0.95)
    simparams.info()
    print('   * Sampling and Fitting:')
    print('     - {:<23s} : {}'.format('Sampling method'  , doe_method  ))
    print('     - {:<23s} : {}'.format('Optimality '      , optimality))
    print('     - {:<23s} : {}'.format('Fitting method'   , fit_method))

    ## ------------------------ Define PCE model --------------------- ###
    if doe_method.lower() == 'mcs':
        orth_poly = museuq.Hermite(d=solver.ndim, hem_type='probabilists')
    elif doe_method.lower() == 'cls':
        orth_poly = museuq.Hermite(d=solver.ndim, hem_type='physicists')
    pce_model = museuq.PCE(orth_poly)
    pce_model.info()

    ## ----------- Candidate and testing data set for DoE ----------- ###
    print(' > Getting candidate data set...')
    u_cand  = get_candidate_data(simparams, doe_method, orth_poly, n_cand, n_test)
    u_test, x_test, y_test = get_test_data(simparams, solver, pce_model, n_test) 
    qoi_test= museuq.metrics.mquantiles(y_test, 1-pf)[0]
    print('   * {:<25s} : {}'.format('Candidate', u_cand.shape))
    print('   * {:<25s} : {}'.format('Test'     , u_test.shape))
    print('   * {:<25s} : {}'.format('Target QoI',qoi_test))

    ## ----------- Initial DoE ----------- ###
    # init_optimality = optimality 
    # init_basis_deg  = 10
    # orth_poly.set_degree(init_basis_deg)
    print(' > Getting initial sample set...')
    sample_selected = []
    init_doe_method = 'lhs' 
    init_n_eval     = 16
    u_train, x_train, y_train = get_init_samples(init_n_eval, solver, pce_model, doe_method=init_doe_method, random_state=100)
    print('   * {:<25s} : {}'.format(' doe_method ', init_doe_method))
    print('   * {:<25s} : {}'.format(' u train shape ', u_train.shape))
    print('   * {:<25s} : [{:.2f},{:.2f}]'.format(' u Domain ', np.amin(u_train), np.amax(u_train)))
    print('   * {:<25s} : [{:.2f},{:.2f}]'.format(' x Domain ', np.amin(x_train), np.amax(x_train)))

    ### ============ Initial Values ============
    p               = plim[0] 
    i_iteration     = 0
    n_eval_path     = []
    sparsity        = [0,] * (plim[1]+1)
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
    new_samples_pct = [1,]* (plim[1]+1)
    score = [0,] * (plim[1]+1)

    ### ============ Start adaptive iteration ============
    print(' > Starting iteration ...')
    while i_iteration < iter_max:
        print(' >>> Iteration No. {:d}: '.format(i_iteration))
        ### ============ Update PCE model ============
        orth_poly.set_degree(p)
        pce_model = museuq.PCE(orth_poly)

        ### ============ Build Surrogate Model ============
        w_train = cal_weight(doe_method, u_train, pce_model)
        pce_model.fit_lassolars(u_train, y_train, w_train)
        y_train_hat = pce_model.predict(u_train)
        y_test_hat  = pce_model.predict(u_test)
        # print(np.min(w_train), np.max(w_train))
        # print('u train min: {}'.format(np.min(u_train, axis=1)))
        # print('u train max: {}'.format(np.max(u_train, axis=1)))
        # print('x train min: {}'.format(np.min(x_train, axis=1)))
        # print('x train max: {}'.format(np.max(x_train, axis=1)))
        res = museuq.metrics.mean_squared_error(y_train, y_train_hat)
        tot = np.var(y_train)
        # print('MSE y train (res): {}'.format(res))
        # print('var y train (tot): {}'.format(tot))
        if res > tot:
            idx1 = np.argwhere(y_train_hat > np.max(y_train))
            idx2 = np.argwhere(y_train_hat < np.min(y_train))
            bad_points = np.hstack((u_train[:,idx1], u_train[:, idx2]))
            np.save('u_train.npy', u_train)
            np.save('u_bad_points', bad_points)
        # print('u train: {}'.format(u_train))
        # print('x train: {}'.format(x_train))
        # print('y train: {}'.format(y_train))
        # print('u_test: {}'.format(u_test[:,:3]))
        # print('u test min: {}'.format(np.min(u_test, axis=1)))
        # print('u test max: {}'.format(np.max(u_test, axis=1)))
        # print('u test mean: {}'.format(np.mean(u_test, axis=1)))
        # print('u test std: {}'.format(np.std(u_test, axis=1)))
        # print('y test max: {}'.format(np.max(y_test)))
        # qoi = museuq.metrics.mquantiles(y_test_hat, 1-pf)
        # print('pf, y_test_hat: {}'.format(qoi))
        # print('y_test_hat max: {}'.format(max(y_test_hat)))

        ### ============ calculating & updating metrics ============
       
        n_eval_path.append(u_train.shape[1])
        poly_order_path.append(p)
        
        cv_error[p] = pce_model.cv_error        
        cv_error_path.append(pce_model.cv_error)

        cum_var         = -np.cumsum(np.sort(-pce_model.coef[1:] **2))
        y_hat_var_pct   = cum_var / cum_var[-1] 
        sparsity_p      = np.argwhere(y_hat_var_pct > 0.95)[0][-1] + 1 ## return index
        sparsity[p]     = sparsity_p + 1 ## phi_0
        acitve_index    = [0,] + list(np.argsort(-pce_model.coef[1:])[:sparsity_p]+1)
        active_basis[p] = [pce_model.basis.basis_degree[i] for i in acitve_index ]
        active_basis_path.append(active_basis[p])

        adj_r2[p] = museuq.metrics.r2_score(y_train, y_train_hat)        
        adj_r2_path.append(museuq.metrics.r2_score(y_train, y_train_hat))
        score[p] = pce_model.score
        qoi = museuq.metrics.mquantiles(y_test_hat, 1-pf)
        QoI[p] = qoi
        QoI_path.append(qoi)

        test_error[p] = museuq.metrics.mean_squared_error(y_test, y_test_hat)
        test_error_path.append(museuq.metrics.mean_squared_error(y_test, y_test_hat))

        ### ============ Cheking Overfitting ============
        if simparams.check_overfitting(cv_error[plim[0]:p+1]):
            print('     >>> Possible overfitting detected')
            print('         - cv error: {}'.format(np.around(cv_error[max(p-4,plim[0]):p+1], 4)))
            new_samples_pct[p]  = new_samples_pct[p] *0.5
            p = max(plim[0], p-2)
            new_samples_pct[p]= new_samples_pct[p] *0.1
            # plim[0] + np.argmin(cv_error[plim[0]:p+1])
            ### update candidate data set for this p degree, cls unbuounded
            print('         - Reseting results for PCE order higher than p = {:d} '.format(p))
            for i in range(p+1, len(active_basis)):
                QoI[i]          = 0
                adj_r2[i]       = 0
                cv_error[i]     = 0
                sparsity[i]     = 0
                test_error[i]   = 0
                active_basis[i] = 0
            continue

        ### ============ Get new samples if not overfitting ============
        ### update candidate data set for this p degree, cls unbuounded
        if doe_method.lower().startswith('cls') and pce_model.basis.dist_name=='norm':
            u_cand_p = p**0.5 * u_cand
        else:
            u_cand_p = u_cand
        orth_poly.set_degree(p)
        pce_model = museuq.PCE(orth_poly)
        ### ============ Get training points ============
        n = math.ceil(sparsity[p] * new_samples_pct[p])
        print('   -- New samples ({:s} {}): p={:d}, {:d}/{:d}, pct:{:.2f} '.format(doe_method, optimality, p,n,sparsity[p], new_samples_pct[p]))
        u_train_new = get_train_data(n, u_cand_p,doe_method, optimality, sample_selected, pce_model.basis, active_basis[p])
        x_train_new = solver.map_domain(u_train_new, pce_model.basis.dist_u)
        y_train_new = solver.run(x_train_new)
        u_train = np.hstack((u_train, u_train_new)) 
        x_train = np.hstack((x_train, x_train_new)) 
        y_train = np.hstack((y_train, y_train_new)) 
        print('   -> New samples shape: {}, total iteration samples: {:d}'.format(u_train_new.shape, len(sample_selected)))

        print('         --------- Iteration No. {:d} Summary ---------- '.format(i_iteration))
        print(' - {:<25s} : {}'.format('Polynomial order (p)', p))
        print(' - {:<25s} : {}'.format('# samples ', n_eval_path[-1]))
        try:
            beta = pce_model.coef
            beta = beta[abs(beta) > 1e-6]
            print(' - {:<25s} : #{:d}'.format('Active basis', len(beta)))
        except:
            pass
        print(' - {:<25s} : {}'.format('R2_adjusted ', np.around(adj_r2[plim[0]:p+1], 2)))
        print(' - {:<25s} : {}'.format('Score  ', np.around(score[plim[0]:p+1], 2)))
        print(' - {:<25s} : {}'.format('cv error ', np.squeeze(np.array(cv_error[plim[0]:p+1]))))
        print(' - {:<25s} : {}'.format('QoI [{:.2f}]'.format(qoi_test), np.around(np.squeeze(np.array(QoI[plim[0]:p+1])), 2)))
        print(' - {:<25s} : {}'.format('test error', np.squeeze(np.array(test_error[plim[0]:p+1]))))
        print('     ------------------------------------------------------------')

        ### ============ updating parameters ============
        p           += 1
        i_iteration += 1
        if not simparams.is_adaptive_continue(n_eval_path[-1], p, qoi=QoI[plim[0]:p]):
            break


    print('------------------------------------------------------------')
    print('>>>>>>>>>>>>>>> Adaptive simulation done <<<<<<<<<<<<<<<<<<<')
    print('------------------------------------------------------------')
    print(' - {:<25s} : {}'.format('Polynomial order (p)', p))
    # print(' - {:<25s} : {} -> #{:d}'.format(' # Active basis', pce_model.active_basis, len(pce_model.active_index)))
    print(' - {:<25s} : {}'.format('# samples', n_eval_path[-1]))
    print(' - {:<25s} : {}'.format('R2_adjusted ', np.around(np.squeeze(np.array(adj_r2[plim[0]:p], dtype=np.float)), 2)))
    print(' - {:<25s} : {}'.format('score ', np.around(np.squeeze(np.array(score[plim[0]:p], dtype=np.float)), 2)))
    print(' - {:<25s} : {}'.format('QoI [{:.2f}]'.format(qoi_test), np.around(np.squeeze(np.array(QoI[plim[0]:p], dtype=np.float)), 2)))
    # print(np.linalg.norm(pce_model.coef - solver.coef, np.inf))
    # print(pce_model.coef[pce_model.coef!=0])
    # print(solver.coef[solver.coef!=0])

    if optimality:
        filename = 'Adaptive_{:s}_{:s}{:s}_{:s}'.format(solver.nickname.capitalize(), doe_method.capitalize(), optimality, fit_method.capitalize())
    else:
        filename = 'Adaptive_{:s}_{:s}_{:s}'.format(solver.nickname.capitalize(), doe_method.capitalize(), fit_method.capitalize())
    path_data  = np.array([n_eval_path, poly_order_path, cv_error_path, active_basis_path, adj_r2_path, QoI_path, test_error_path]) 
    np.save(os.path.join(simparams.data_dir_result, filename+'_path'), path_data)
    data  = np.array([n_eval_path, poly_order_path, cv_error, active_basis, adj_r2, QoI, test_error]) 
    np.save(os.path.join(simparams.data_dir_result, filename), data)

if __name__ == '__main__':
    main()
