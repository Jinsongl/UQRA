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
        except:
            x_test = solver.map_domain(u_test_p, pce_model.basis.dist_u)
            y_test = solver.run(x_test)
            data   = np.vstack((u_test_p, x_test, y_test.reshape(1,-1)))
            np.save(os.path.join(simparams.data_dir_result, 'MCS', filename), data)

    elif sampling_method.lower().startswith('cls'):
        filename= r'DoE_McsE6d{:d}R0.npy'.format(solver.ndim) if pce_model.basis.dist_name.lower() == 'normal' else r'DoE_McsE6R0.npy'
        try:
            data_set = np.load(os.path.join(simparams.data_dir_result, 'Pluripotential', filename))
            y_test   = data_set[-1,:]
        except:
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
    # orth_poly   = museuq.Legendre(d=ndim, deg=15)
    # orth_poly   = museuq.Hermite(d=ndim, deg=15)
    # solver      = museuq.sparse_poly(orth_poly, sparsity=5, seed=100)
    # orth_poly   = museuq.Legendre(d=solver.ndim)
    # orth_poly   = museuq.Hermite(d=solver.ndim)

    solver      = museuq.Ishigami()
    orth_poly   = museuq.Legendre(d=solver.ndim)
    pce_model   = museuq.PCE(orth_poly)

    simparams   = museuq.simParameters(solver.nickname)

    ### ============ Adaptive parameters ============
    plim        = (2,100)
    n_budget    = 2000 
    n_cand      = int(1e5)
    n_test      = -1 
    sampling    = 'CLS_OED'
    optimality  = 'S'
    fit_method  = 'LASSOLARS'
    k_sparsity  = 25 # guess. K sparsity to meet RIP condition 
    simparams.set_adaptive_parameters(n_budget=n_budget, plim=plim, min_r2=0.9, abs_qoi=0.01)
    simparams.info()

    # elif orth_poly.dist_name.lower() == 'normal':
        # if sampling.lower().startswith('cls'):
            # pce_model = museuq.PCE(museuq.Hermite(d=ndim, deg=p, hem_type='physicists'))
        # else:
            # pce_model = museuq.PCE(museuq.Hermite(d=ndim, deg=p, hem_type='probabilists'))
    # else:
        # raise NotImplementedError

    print(' Parameters:')
    print('------------------------------------------------------------')
    print(' - {:<25s} : {}'.format('Sampling method'  , sampling  ))
    print(' - {:<25s} : {}'.format('Optimality '      , optimality))
    print(' - {:<25s} : {}'.format('Fitting method'   , fit_method))
    print(' - {:<25s} : {}'.format('Simulation budget', n_budget  ))
    print(' - {:<25s} : {}'.format('Poly degree limit', plim      ))
    ### ============ Initial Values ============
    p_init      = 2
    n_new       = 5
    n_eval_done = 0
    n_eval_init = 20
    n_eval_init = max(n_eval_init, 2 * k_sparsity) ## for ols, oversampling rate at least 2
    n_eval_path = [n_eval_done,]
    i_iteration = -1
    sample_selected = []
    active_basis= collections.deque(maxlen=3)
    u_train = np.array([])
    x_train = np.array([])
    y_train = np.array([])

    ### ============ Stopping Criteria ============
    cv_error        = []
    mquantiles      = []
    r2_score_adj    = []
    test_error      = []
    f_hat           = None

    ### ============ Candidate data set for DoE ============
    print(' > loading candidate data set...')
    u_cand, u_test = get_candidate_data(simparams, sampling, orth_poly, n_cand, n_test)

    ### ============ Start adaptive iteration ============
    print(' > Starting iteration ...')
    while True:
        i_iteration += 1
        print(' >>> Iteration No: {:d}'.format(i_iteration))
        p = p_init if i_iteration == 0 else p
        ### update candidate data set for this p degree, cls unbuounded
        if sampling.lower().startswith('cls') and orth_poly.dist_name.lower() == 'normal':
            u_cand_p = p**0.5 * u_cand
            u_test_p = p**0.5 * u_test
        else:
            u_cand_p = u_cand
            u_test_p = u_test
        ### ============ Redefine PCE model ============
        orth_poly.set_degree(p)
        pce_model = museuq.PCE(orth_poly)

        ### ============ Get training points ============
        nsamples = n_new if i_iteration else n_eval_init
        print(' > Getting sample points ...')
        u_train_new = get_train_data(sampling, optimality, sample_selected, pce_model, active_basis, nsamples, u_cand_p)
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

        n_eval_done += nsamples

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

        ### ============ Cheking Overfitting ============
        cv_error.append(pce_model.cv_error)
        if simparams.check_overfitting(cv_error):
            print('     - Overfitting detected, setting p = p - 1')
            while len(active_basis) > 1:
                p -= 1
                active_basis.pop()
            continue
        ### ============ calculating & updating metrics ============
        r2_score_adj.append(museuq.metrics.r2_score_adj(y_train, y_train_hat, len(pce_model.active_)))
        mquantiles.append(museuq.metrics.mquantiles(y_test_hat, 1-pf))
        test_error.append(museuq.metrics.mean_squared_error(y_test, y_test_hat))
        n_eval_path.append(n_eval_done)

        print(' - {:<25s} : {}'.format('Polynomial order (p)', p))
        print(' - {:<25s} : {}'.format('Evaluations done ', n_eval_done))
        if f_hat:
            print(' - {:<25s} : {} -> #{:d}'.format('Active basis', f_hat.active_, len(f_hat.active_)))
            beta = f_hat.coef
            beta = beta[abs(beta) > 1e-9]
            print(' - {:<25s} : #{:d}'.format('Active basis', len(beta)))
        print(' - {:<25s} : {}'.format('R2_adjusted ', np.around(r2_score_adj, 2)))
        print(' - {:<25s} : {}'.format('mquantiles', np.around(np.squeeze(np.array(mquantiles)), 2)))
        print(' - {:<25s} : {}'.format('test error', np.squeeze(np.array(test_error))))
        print('     ------------------------------------------------------------')

        ### ============ updating parameters ============
        p           +=1
        f_hat        = pce_model
        active_basis.append(f_hat.active_)
        if not simparams.is_adaptive_continue(n_eval_done, p, adj_r2=r2_score_adj, qoi=mquantiles):
            break

    print('------------------------------------------------------------')
    print('>>>>>>>>>>>>>>> Adaptive simulation done <<<<<<<<<<<<<<<<<<<')
    print('------------------------------------------------------------')
    print(' - {:<25s} : {}'.format('Polynomial order (p)', p))
    print(' - {:<25s} : {} -> #{:d}'.format('Active basis', f_hat.active_, len(f_hat.active_)))
    print(f_hat.coef)
    print(' - {:<25s} : {}'.format('# Evaluations ', n_eval_done))
    print(' - {:<25s} : {}'.format('R2_adjusted ', np.around(r2_score_adj, 2)))
    print(' - {:<25s} : {}'.format('mquantiles', np.around(np.squeeze(np.array(mquantiles)), 2)))

    # filename = 'mquantile_DoE_q15_OptD2040_PCE{:d}_{:s}_path.npy'.format(p, fit_method)
    # np.save(os.path.join(simparams.data_dir, filename), np.array(mquantiles))

    # filename = 'r2_DoE_q15_OptD2040_PCE{:d}_{:s}_path.npy'.format(p, fit_method)
    # np.save(os.path.join(simparams.data_dir, filename), np.array(r2_score_adj))

    # filename = 'cv_error_DoE_q15_OptD2040_PCE{:d}_{:s}_path.npy'.format(p, fit_method)
    # np.save(os.path.join(simparams.data_dir, filename), np.array(cv_error))

    # filename = 'n_eval_DoE_q15_OptD2040_PCE{:d}_{:s}_path.npy'.format(p, fit_method)
    # np.save(os.path.join(simparams.data_dir, filename), np.array(n_eval_path))

    # mquantiles = []
    # for r in tqdm(range(10), ascii=True, desc="   - " ):
        # filename = 'DoE_McsE6R{:d}.npy'.format(r)
        # data_set = np.load(os.path.join(simparams.data_dir, filename))
        # u_samples= data_set[0:ndim,:]
        # x_samples= data_set[ndim: 2*ndim,:]
        # y_samples= f_hat.predict(u_samples)
        # mquantiles.append(museuq.metrics.mquantiles(y_samples, [1-1e-4, 1-1e-5, 1-1e-6]))
        # filename = 'DoE_McsE6R{:d}_q15_OptD2040_PCE{:d}_{:s}.npy'.format(r, p, fit_method)
        # np.save(os.path.join(simparams.data_dir, filename), y_samples)

    # filename = 'mquantile_DoE_q15_OptD2040_PCE{:d}_{:s}.npy'.format(p, fit_method)
    # np.save(os.path.join(simparams.data_dir, filename), np.array(mquantiles))


if __name__ == '__main__':
    main()
