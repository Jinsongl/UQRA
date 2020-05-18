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
from scipy import sparse
from tqdm import tqdm
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

def main():
    ## ------------------------ Displaying set up ------------------- ###
    np.set_printoptions(precision=4)
    np.set_printoptions(threshold=8)
    np.set_printoptions(suppress=True)
    np.random.seed(100)
    ndim = 2
    ## ------------------------ Simulation Parameters ----------------- ###
    simparams = museuq.Parameters()
    simparams.pce_degs   = np.array([20])
    simparams.n_cand     = int(1e5)
    simparams.doe_method = 'MCS' ### 'mcs', 'D', 'S', 'reference'
    simparams.optimality = 'S'#'D', 'S', None
    # simparams.hem_type   = 'physicists'
    simparams.hem_type   = 'probabilists'
    simparams.fit_method = 'LASSOLARS'
    simparams.n_splits   = 50
    repeats              = 1 if simparams.optimality == 'D'  else 50
    ratio_sn             = np.linspace(0,1,21)[1:]
    ratio_nP             = np.linspace(0,1,21)[1:]
    # psn_done             = np.load('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/SparsePoly/SparsePoly_Phase_2Hem20_Cls_Lassolars_psn.npy')
    # alphas             = np.linspace(0,1,51)
    # alphas = np.append(alphas,np.linspace(2,4,11))
    # alphas = np.append(alphas,np.linspace(4,10,13))
    # alphas               = [1.2, 2.0]
    # simparams.num_samples=np.arange(21+1, 130, 5)
    ### ============ Initial Values ============
    print(' > Starting simulation...')
    data_p = []
    for p in simparams.pce_degs:
        ## ------------------------ Define solver ----------------------- ###
        # orth_poly   = museuq.Legendre(d=ndim, deg=p)
        orth_poly   = museuq.Hermite(d=ndim, deg=p, hem_type=simparams.hem_type)
        ## ----------- Oversampling ratio ----------- ###
        # simparams.update_num_samples(orth_poly.num_basis, alphas=alphas)
        nsamples = np.unique(np.rint(ratio_nP * orth_poly.num_basis).astype(np.int32))
        for j, n in enumerate(nsamples):
            sparsity = np.unique(np.rint(ratio_sn * n).astype(np.int32))
            sparsity = sparsity[sparsity != 0]
            sparsity = sparsity[sparsity != 1]
            data_s = []
            for i, s in enumerate(sparsity):
                print(' > Case: n={:d} [{:d}/{:d}], s={:d}[{:d}/{:d}]'.format(n,j,len(nsamples),s,i,len(sparsity)))
                # check_psn = psn_done == [s,n]
                # if np.logical_and(check_psn[:,0], check_psn[:,1]).any():
                    # print('     pass')
                    # continue
                if s > n:
                    print('     pass')
                    continue
                # nsamples = np.arange(6,81) 
                solver = museuq.SparsePoly(orth_poly, sparsity=s, seed=100)
                simparams.solver = solver
                simparams.update()
                simparams.info()
                ## ----------- Define PCE  ----------- ###
                pce_model= museuq.PCE(orth_poly)
                pce_model.info()

                modeling = museuq.Modeling(solver, pce_model, simparams)
                # modeling.sample_selected=[]

                print('\n================================================================================')
                print('   - Sampling and Fitting:')
                print('     - {:<23s} : {}'.format('Sampling method'  , simparams.doe_method))
                print('     - {:<23s} : {}'.format('Optimality '      , simparams.optimality))
                print('     - {:<23s} : {}'.format('Fitting method'   , simparams.fit_method))

                ## ----------- Candidate and testing data set for DoE ----------- ###
                print(' > Getting candidate data set...')
                u_cand = modeling.get_candidate_data()
                with np.printoptions(precision=2):
                    u_cand_mean_std = np.array((np.mean(u_cand[0]), np.std(u_cand[0])))
                    u_cand_ref = np.array(modeling.candidate_data_reference())
                    print('    - {:<25s} : {}'.format('Candidate Data ', u_cand.shape))
                    print('    > {:<25s}'.format('Validate data set '))
                    print('    - {:<25s} : {} {} '.format('u cand (mean, std)', u_cand_mean_std, u_cand_ref))


                ### ============ Initialize pce_model for each  ============
                pce_model= museuq.PCE(orth_poly)
                ### ============ Get training points ============
                u_cand_p = p ** 0.5 * u_cand if modeling.is_cls_unbounded() else u_cand
                _, u_train = modeling.get_train_data((repeats,n), u_cand_p, u_train=None, basis=pce_model.basis)
                for iu_train in tqdm(u_train, ascii=True, ncols=80,
                        desc='   [s={:d}, ={:d}, P={:d}]'.format(s, n, pce_model.num_basis)):

                    ix_train = solver.map_domain(iu_train, pce_model.basis.dist_u)
                    iy_train = solver.run(ix_train)
                    ### ============ Build Surrogate Model ============
                    U_train = pce_model.basis.vandermonde(iu_train)
                    if simparams.doe_method.lower().startswith('cls'):
                        w_train = modeling.cal_cls_weight(iu_train, pce_model.basis)
                        U_train = modeling.rescale_data(U_train, w_train) 
                    else:
                        w_train = None
                        U_train = U_train

                    # tqdm.write("iu_train.shape {}".format(iu_train.shape))
                    # tqdm.write("iy_train.shape {}".format(iy_train.shape))
                    pce_model.fit(simparams.fit_method, iu_train, iy_train, w_train, n_splits=simparams.n_splits)

                    ## condition number, kappa = max(svd)/min(svd)
                    _, sig_value, _ = np.linalg.svd(U_train)
                    kappa = max(abs(sig_value)) / min(abs(sig_value)) 

                    # QoI_.append(np.linalg.norm(solver.coef- pce_model.coef, np.inf) < 1e-2)
                    coef_abserr_inf = sparse_poly_coef_error(solver, pce_model, np.inf)
                    coef_relerr_l2 = (sparse_poly_coef_error(solver, pce_model, 2)/np.linalg.norm(solver.coef,2))

                    data_p.append([p, s, n, coef_abserr_inf, kappa, pce_model.score, pce_model.cv_error, coef_relerr_l2])
                ### ============ calculating & updating metrics ============
                # with np.printoptions(precision=4):
                    # print('     - {:<15s} : {:.4f}'.format( '|coef|'    , np.mean(coef_err_repeat)))
                    # print('     - {:<15s} : {:.4f}'.format( 'CV MSE'    , np.mean(cv_err_repeat)))
                    # print('     - {:<15s} : {:.4f}'.format( 'Score '    , np.mean(score_repeat)))
                    # print('     - {:<15s} : {:.4e}'.format( 'kappa '    , np.mean(cond_num_repeat)))
                    # print('     ----------------------------------------')

    filename = '{:s}_Phase_{:s}_{:s}'.format(solver.nickname, pce_model.tag, simparams.tag)
    try:
        np.save(os.path.join(simparams.data_dir_result, filename), np.array(data_p))
    except:
        np.save(os.path.join(os.getcwd(), filename), np.array(data_p))


if __name__ == '__main__':
    main()
