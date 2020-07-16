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
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()

def main():

    ## ------------------------ Displaying set up ------------------- ###
    np.set_printoptions(precision=2)
    np.set_printoptions(threshold=8)
    np.set_printoptions(suppress=True)
    n_new       = 5
    iter_max    = 100

    ## ------------------------ Define solver ----------------------- ###
    pf          = 1e-4
    solver      = uqra.CornerPeak(stats.uniform(-1,2),d=2)
    simparams   = uqra.Parameters(solver.nickname)
    # print(simparams.data_dir_result)
    ## ------------------------ Modeling parameters ----------------- ###
    modeling = uqra.Modeling()
    modeling.n_cand     = int(1e5)
    modeling.n_test     = -1
    modeling.doe_method = 'CLS'
    modeling.optimality = 'S'#'D', 'S', None
    modeling.fit_method = 'LASSOLARS'
    modeling.n_splits   = 5

    ## ------------------------ Adaptive parameters ----------------- ###
    n_budget    = 2000
    plim        = (2,100)
    simparams.set_adaptive_parameters(n_budget=n_budget, plim=plim, abs_qoi=0.02, min_r2=0.95)
    simparams.info()
    print('   * Sampling and Fitting:')
    print('     - {:<23s} : {}'.format('Sampling method'  , modeling.doe_method  ))
    print('     - {:<23s} : {}'.format('Optimality '      , modeling.optimality))
    print('     - {:<23s} : {}'.format('Fitting method'   , modeling.fit_method))

    ## ------------------------ Define PCE model --------------------- ###
    orth_poly = uqra.Legendre(d=solver.ndim)
    # orth_poly = uqra.Hermite(d=solver.ndim, hem_type='probabilists')
    # orth_poly = uqra.Hermite(d=solver.ndim, hem_type='physicists')
    pce_model = uqra.PCE(orth_poly)
    pce_model.info()
    modeling.set_dist_names(solver, pce_model)

    ## ----------- Candidate and testing data set for DoE ----------- ###
    print(' > Getting candidate data set...')
    u_cand = modeling.get_candidate_data(simparams, orth_poly)
    u_test, x_test, y_test = modeling.get_test_data(simparams, solver, pce_model) 
    qoi_test= uqra.metrics.mquantiles(y_test, 1-pf)[0]
    print('   * {:<25s} : {}'.format('Candidate', u_cand.shape))
    print('   * {:<25s} : {}'.format('Test'     , u_test.shape))
    print('   * {:<25s} : {}'.format('Target QoI [pf={:.2e}]'.format(pf),qoi_test))

    ## ----------- Initial DoE ----------- ###
    print(' > Getting initial sample set...')
    modeling.sample_selected = []

    # init_doe_method = 'cls' 
    # init_optimality = None 
    # init_basis_deg  = 10
    # pce_model.set_degree(init_basis_deg) 
    # init_n_eval     = math.ceil(1.5*pce_model.num_basis*np.log(pce_model.num_basis))
    # u_train, x_train, y_train = modeling.get_init_samples(init_n_eval, solver, pce_model,doe_method=init_doe_method, random_state=100)
    init_n_eval     = 32
    init_doe_method = 'lhs' 
    u_train, x_train, y_train = modeling.get_init_samples(init_n_eval, solver, pce_model, doe_method=init_doe_method, random_state=100)
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
    score          = [0,] * (plim[1]+1)
    score_path     = []
    test_error      = [0,] * (plim[1]+1)
    test_error_path = []
    QoI             = [0,] * (plim[1]+1)
    QoI_path        = []
    active_basis    = [0,] * (plim[1]+1)
    active_basis_path= [] 
    new_samples_pct = [1,]* (plim[1]+1)

    ### ============ Start adaptive iteration ============
    print(' > Starting iteration ...')
    while i_iteration < iter_max:
        print(' >>> Iteration No. {:d}: '.format(i_iteration))
        ### ============ Update PCE model ============
        orth_poly.set_degree(p)
        pce_model = uqra.PCE(orth_poly)

        ### ============ Build Surrogate Model ============
        if modeling.doe_method.lower().startswith('cls'):
            w_train = modeling.cal_cls_weight(u_train, pce_model.basis)
        else:
            w_train = None
        pce_model.fit_lassolars(u_train, y_train, w_train)
        y_train_hat = pce_model.predict(u_train)
        y_test_hat  = pce_model.predict(u_test)
        # print(np.min(w_train), np.max(w_train))
        # print('u train min: {}'.format(np.min(u_train, axis=1)))
        # print('u train max: {}'.format(np.max(u_train, axis=1)))
        # print('x train min: {}'.format(np.min(x_train, axis=1)))
        # print('x train max: {}'.format(np.max(x_train, axis=1)))
        res = uqra.metrics.mean_squared_error(y_train, y_train_hat)
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
        # qoi = uqra.metrics.mquantiles(y_test_hat, 1-pf)
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

        score[p] = pce_model.score
        score_path.append(pce_model.score)
        qoi = uqra.metrics.mquantiles(y_test_hat, 1-pf)
        QoI[p] = qoi
        QoI_path.append(qoi)

        test_error[p] = uqra.metrics.mean_squared_error(y_test, y_test_hat)
        test_error_path.append(uqra.metrics.mean_squared_error(y_test, y_test_hat))

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
                score[i]        = 0
                cv_error[i]     = 0
                sparsity[i]     = 0
                test_error[i]   = 0
                active_basis[i] = 0
            continue

        ### ============ Get new samples if not overfitting ============
        ### update candidate data set for this p degree, cls unbuounded
        u_cand_p = p ** 0.5 * u_cand if modeling.is_cls_unbounded() else u_cand
        orth_poly.set_degree(p)
        pce_model = uqra.PCE(orth_poly)
        ### ============ Get training points ============
        n = math.ceil(sparsity[p] * new_samples_pct[p])
        if u_train.shape[1] + n < 2*sparsity[p]:
            n = 2 * sparsity[p] - u_train.shape[1]
        print('   -- New samples ({:s} {}): p={:d}, {:d}/{:d}, pct:{:.2f} '.format(modeling.doe_method, modeling.optimality, p,n,sparsity[p], new_samples_pct[p]))
        u_train_new, _ = modeling.get_train_data(n, u_cand_p, basis=pce_model.basis, active_basis=active_basis[p])
        x_train_new = solver.map_domain(u_train_new, pce_model.basis.dist_u)
        y_train_new = solver.run(x_train_new)
        u_train = np.hstack((u_train, u_train_new)) 
        x_train = np.hstack((x_train, x_train_new)) 
        y_train = np.hstack((y_train, y_train_new)) 
        print('   -> New samples shape: {}, total iteration samples: {:d}'.format(u_train_new.shape, len(modeling.sample_selected)))

        np.set_printoptions(precision=2)
        print('         --------- Iteration No. {:d} Summary ---------- '.format(i_iteration))
        print(' - {:<25s} : {}'.format('Polynomial order (p)', p))
        print(' - {:<25s} : {}'.format('# samples ', n_eval_path[-1]))
        try:
            beta = pce_model.coef
            beta = beta[abs(beta) > 1e-6]
            print(' - {:<25s} : #{:d}'.format('Active basis', len(beta)))
        except:
            pass
        print(' - {:<25s} : {}'.format('Score  ', np.around(score[plim[0]:p+1], 2)))
        print(' - {:<25s} : {}'.format('cv error ', np.squeeze(np.array(cv_error[plim[0]:p+1]))))
        print(' - {:<25s} : {}'.format('QoI [{:.2f}]'.format(qoi_test), np.around(np.squeeze(np.array(QoI[plim[0]:p+1])), 2)))
        print(' - {:<25s} : {}'.format('test error', np.squeeze(np.array(test_error[plim[0]:p+1]))))
        print('     ------------------------------------------------------------')

        ### ============ updating parameters ============
        p           += 1
        i_iteration += 1
        if not simparams.is_adaptive_continue(n_eval_path[-1], p, qoi=QoI[plim[0]:p], r2=score[plim[0]:p]):
            break


    print('------------------------------------------------------------')
    print('>>>>>>>>>>>>>>> Adaptive simulation done <<<<<<<<<<<<<<<<<<<')
    print('------------------------------------------------------------')
    print(' - {:<25s} : {}'.format('Polynomial order (p)', p))
    # print(' - {:<25s} : {} -> #{:d}'.format(' # Active basis', pce_model.active_basis, len(pce_model.active_index)))
    print(' - {:<25s} : {}'.format('# samples', n_eval_path[-1]))
    print(' - {:<25s} : {}'.format('R2_adjusted ', np.around(np.squeeze(np.array(score[plim[0]:p], dtype=np.float)), 2)))
    print(' - {:<25s} : {}'.format('QoI [{:.2f}]'.format(qoi_test), np.around(np.squeeze(np.array(QoI[plim[0]:p], dtype=np.float)), 2)))
    # print(np.linalg.norm(pce_model.coef - solver.coef, np.inf))
    # print(pce_model.coef[pce_model.coef!=0])
    # print(solver.coef[solver.coef!=0])

    filename = 'Adaptive_{:s}_{:s}_{:s}'.format(solver.nickname, pce_model.tag[:-1], modeling.get_tag())
    path_data  = np.array([n_eval_path, poly_order_path, cv_error_path, active_basis_path, score_path, QoI_path, test_error_path]) 
    np.save(os.path.join(simparams.data_dir_result, filename+'_path'), path_data)
    data  = np.array([n_eval_path, poly_order_path, cv_error, active_basis, score, QoI, test_error]) 
    np.save(os.path.join(simparams.data_dir_result, filename), data)

if __name__ == '__main__':
    main()
