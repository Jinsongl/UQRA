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
from tqdm import tqdm
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()

def main():

    ## ------------------------ Displaying set up ------------------- ###
    np.set_printoptions(precision=4)
    np.set_printoptions(threshold=8)
    np.set_printoptions(suppress=True)
    iter_max    = 200
    pf          = 1e-4
    random_seed = 100
    n_short_term= 1

    ## ------------------------ Define solver ----------------------- ###
    # out_responses= [2,] 
    # solver      = uqra.linear_oscillator(out_responses=out_responses, out_stats='absmax',n_short_term=n_short_term) 

    out_responses= [2,] 
    solver      = uqra.duffing_oscillator(out_responses=out_responses, out_stats='absmax',
            n_short_term=n_short_term, spec_name='JONSWAP') 
    ## ------------------------ Simulation Parameters ----------------- ###
    simparams = uqra.Parameters()
    simparams.solver     = solver
    # simparams.pce_degs   = np.array(range(2,16))
    simparams.n_cand     = int(1e5)
    simparams.n_test     = -1
    simparams.doe_method = 'MCS' ### 'mcs', 'D', 'S', 'reference'
    simparams.optimality = 'S'#'D', 'S', None
    # simparams.hem_type   = 'physicists'
    simparams.hem_type   = 'probabilists'
    simparams.fit_method = 'LASSOLARS'
    simparams.n_splits   = 50
    # simparams.update_dir(data_dir_result='/Users/jinsongliu/BoxSync/PhD_UT/Reproduce_Papers/OptimalityS_JSC2016/Data')
    simparams.update()
    ## ------------------------ Adaptive parameters ----------------- ###
    n_budget = 300
    plim     = (2,100)
    simparams.set_adaptive_parameters(n_budget=n_budget, plim=plim, rel_qoi=0.0001, min_r2=0.95)
    simparams.info()

    ## ------------------------ Define Initial PCE model --------------------- ###
    # orth_poly = uqra.Legendre(d=solver.ndim, deg=plim[0])
    orth_poly = uqra.Hermite(d=solver.ndim, deg=plim[0], hem_type=simparams.hem_type)
    pce_model = uqra.PCE(orth_poly)
    pce_model.info()

    modeling = uqra.Modeling(solver, pce_model, simparams)
    ## ----------- Candidate and testing data set for DoE ----------- ###
    print(' > Getting candidate data set...')
    u_cand = modeling.get_candidate_data()
    u_test, x_test, y_test = modeling.get_test_data(solver, pce_model, n=10000, 
            qoi=out_responses,n_short_term=n_short_term, random_seed=random_seed)
    y_test = np.mean(y_test, axis=0)
    exceed_val_y_test = uqra.metrics.mquantiles(y_test, 1-pf)[0]
    with np.printoptions(precision=2):
        u_cand_mean_std = np.array((np.mean(u_cand[0]), np.std(u_cand[0])))
        u_test_mean_std = np.array((np.mean(u_test[0]), np.std(u_test[0])))
        x_test_mean_std = np.array((np.mean(x_test[0]), np.std(x_test[0])))
        u_cand_ref = np.array(modeling.candidate_data_reference())
        u_test_ref = np.array(modeling.test_data_reference())

        print('    - {:<25s} : {}'.format('Candidate Data ', u_cand.shape))
        print('    - {:<25s} : {}'.format('Test Data ', u_test.shape))
        print('    - {:<25s} : {}'.format('Target QoI [pf={:.0e}]'.format(pf), exceed_val_y_test))
        print('    > {:<25s}'.format('Validate data set '))
        print('    - {:<25s} : {} {} '.format('u cand (mean, std)', u_cand_mean_std, u_cand_ref))
        print('    - {:<25s} : {} {} '.format('u test (mean, std)', u_test_mean_std, u_test_ref))
        try:
            x_test_ref = np.array((solver.distributions[0].mean(), solver.distributions[0].std()))
            print('    - {:<25s} : {} {} '.format('x test (mean, std)', x_test_mean_std, x_test_ref))
        except:
            pass

    ## ----------- Initial DoE ----------- ###
    print(' > Getting initial sample set...')
    init_n_eval     = 32
    init_doe_method = 'lhs' 
    u_train, x_train, y_train = modeling.get_init_samples(init_n_eval, doe_method=init_doe_method, random_state=random_seed)
    y_train = np.mean(y_train, axis=0)
    u_sampling_pdf  = np.prod(pce_model.basis.dist_u[0].pdf(u_train), axis=0)
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
    new_samples_pct = [1, ]* (plim[1]+1)
    bias_weight     = [1,]* (plim[1]+1)

    ### ============ Start adaptive iteration ============
    print(' > Starting iteration ...')
    while i_iteration < iter_max:
        print('\n===================  >>> Iteration No. {:d}:   <<< =================='.format(i_iteration))
        print('   - Sampling and Fitting:')
        print('     - {:<23s} : {}'.format('Sampling method'  , simparams.doe_method))
        print('     - {:<23s} : {}'.format('Optimality '      , simparams.optimality))
        print('     - {:<23s} : {}'.format('Fitting method'   , simparams.fit_method))
        ### ============ Update PCE model ============
        orth_poly.set_degree(p)
        pce_model = uqra.PCE(orth_poly)
        modeling  = uqra.Modeling(solver, pce_model, simparams)

        ### ============ Estimate sparsity ============
        tqdm.write(' > {:<20s}: alpha = {:.2f}, # samples = {:d}'.format(
                    'Sparsity estimating',u_train.shape[1]/pce_model.num_basis, u_train.shape[1]))
        bias_weight = modeling.cal_adaptive_bias_weight(u_train, p, u_sampling_pdf)
        if simparams.doe_method.lower().startswith('cls'):
            w_train = modeling.cal_cls_weight(u_train, pce_model.basis)
        else:
            w_train = 1
        w_train = w_train * bias_weight
        pce_model.fit(simparams.fit_method, u_train, y_train, w_train, n_splits=simparams.n_splits, epsilon=1e-4)
        pce_model.var(0.95)

        ### ============ Get new samples ============
        ### update candidate data set for this p degree, cls unbuounded
        u_cand_p = p ** 0.5 * u_cand if modeling.is_cls_unbounded() else u_cand
        n = max(3,math.ceil(len(pce_model.var_pct_basis)* new_samples_pct[p]))
        # n = math.ceil(len(pce_model.active_basis)* new_samples_pct[p])
        if u_train.shape[1] + n < math.ceil(pce_model.least_ns_ratio * pce_model.sparsity):
            n = math.ceil(pce_model.least_ns_ratio * pce_model.sparsity)  - u_train.shape[1]
        if u_train.shape[1] + n > n_budget:
            n = n_budget - u_train.shape[1]

        if n == 0:
            break

        tqdm.write(' > {:<20s}: Optimality-> {}; Basis-> {}/{}; # new samples = {:d}; pct={:.2f} '.format(
            'New samples', simparams.optimality, len(pce_model.active_basis), pce_model.num_basis, n, new_samples_pct[p]))
        u_train_new, _ = modeling.get_train_data(n, u_cand_p, u_train, basis=pce_model.basis, active_basis=pce_model.active_basis)
        x_train_new = solver.map_domain(u_train_new, pce_model.basis.dist_u)
        print('u_train: {}'.format(u_train_new.shape))
        print('x_train: {}'.format(x_train_new.shape))
        y = solver.run(x_train_new, random_seed=random_seed)
        y_train_new = np.mean(y,axis=0)
        u_sampling_pdf = np.concatenate((u_sampling_pdf, modeling.sampling_density(u_train_new, p))) 
        u_train = np.hstack((u_train, u_train_new)) 
        x_train = np.hstack((x_train, x_train_new)) 
        y_train = np.hstack((y_train, y_train_new)) 

        ### ============ Build 2nd Surrogate Model ============
        bias_weight = modeling.cal_adaptive_bias_weight(u_train, p, u_sampling_pdf)
        # print(bias_weight)
        if simparams.doe_method.lower().startswith('cls'):
            w_train = modeling.cal_cls_weight(u_train, pce_model.basis, pce_model.active_index)
            w_train = w_train * bias_weight[p]
        else:
            w_train = 1
        w_train = w_train * bias_weight

        tqdm.write(' > {:<20s}: alpha = {:.2f}, # samples = {:d}'.format(
            'Fitting sparse model', u_train.shape[1]/pce_model.num_basis, u_train.shape[1]))
        pce_model.fit('ols', u_train, y_train, w_train, n_splits=simparams.n_splits, active_basis=pce_model.active_basis)
        y_train_hat = pce_model.predict(u_train)
        y_test_hat  = pce_model.predict(u_test)
        # print(np.min(w_train), np.max(w_train))
        # print('u train min: {}'.format(np.min(u_train, axis=1)))
        # print('u train max: {}'.format(np.max(u_train, axis=1)))
        # print('x train min: {}'.format(np.min(x_train, axis=1)))
        # print('x train max: {}'.format(np.max(x_train, axis=1)))
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

        sparsity[p]     = pce_model.sparsity 
        active_basis[p] = pce_model.active_basis 
        active_basis_path.append(pce_model.active_basis)

        score[p] = pce_model.score
        score_path.append(pce_model.score)
        qoi = uqra.metrics.mquantiles(y_test_hat, 1-pf)
        QoI[p] = qoi
        QoI_path.append(qoi)

        test_error[p] = uqra.metrics.mean_squared_error(y_test, y_test_hat)
        test_error_path.append(uqra.metrics.mean_squared_error(y_test, y_test_hat))

        print(' > Summary:')
        print('   - {:<25s} : {}'.format('Polynomial order (p)', p))
        print('   - {:<25s} : {}'.format('# samples ', n_eval_path[-1]))
        try:
            beta = pce_model.coef
            beta = beta[abs(beta) > 1e-6]
            print('   - {:<25s} : #{:d}'.format('Active basis', len(beta)))
        except:
            pass
        print('   - {:<25s} : {}'.format('Score  ', np.around(score[plim[0]:p+1], 2)))
        print('   - {:<25s} : {}'.format('cv error ', np.squeeze(np.array(cv_error[plim[0]:p+1]))))
        print('   - {:<25s} : {}'.format('QoI [{:.2f}]'.format(exceed_val_y_test), np.around(np.squeeze(np.array(QoI[plim[0]:p+1])), 2)))
        print('   - {:<25s} : {}'.format('test error', np.squeeze(np.array(test_error[plim[0]:p+1]))))

        ### ============ Cheking Overfitting ============
        if simparams.check_overfitting(cv_error[plim[0]:p+1]):
            print(' > Overfitting Correction <<<')
            print('   - cv error: {}'.format(np.around(cv_error[max(p-4,plim[0]):p+1], 4)))
            # new_samples_pct[p]  = new_samples_pct[p] *0.5
            p = max(plim[0], p-2)
            new_samples_pct[p]= new_samples_pct[p] *0.5
            # plim[0] + np.argmin(cv_error[plim[0]:p+1])
            ### update candidate data set for this p degree, cls unbuounded
            print('   - Reseting results for PCE order higher than p = {:d} '.format(p))
            for i in range(p+1, len(active_basis)):
                QoI[i]          = 0
                score[i]        = 0
                cv_error[i]     = 0
                sparsity[i]     = 0
                test_error[i]   = 0
                active_basis[i] = 0
            continue

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
    print(' - {:<25s} : {}'.format('QoI [{:.2f}]'.format(exceed_val_y_test), np.around(np.squeeze(np.array(QoI[plim[0]:p], dtype=np.float)), 2)))
    # print(np.linalg.norm(pce_model.coef - solver.coef, np.inf))
    # print(pce_model.coef[pce_model.coef!=0])
    # print(solver.coef[solver.coef!=0])
    for iqoi in out_responses:
        filename = 'Adaptive_{:s}_{:s}_{:s}_y{:d}'.format(solver.nickname, pce_model.tag[:4], simparams.tag, iqoi)
        path_data  = np.array([n_eval_path, poly_order_path, cv_error_path, active_basis_path, score_path, QoI_path, test_error_path]) 
        np.save(os.path.join(simparams.data_dir_result, filename+'_path'), path_data)
        data  = np.array([n_eval_path, poly_order_path, cv_error, active_basis, score, QoI, test_error]) 
        np.save(os.path.join(simparams.data_dir_result, filename), data)
        data_train = np.vstack((u_train, x_train, y_train.reshape(1, -1)))
        np.save(os.path.join(simparams.data_dir_result, filename+'_train'), data_train)


if __name__ == '__main__':
    main()
