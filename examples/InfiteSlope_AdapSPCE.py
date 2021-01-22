#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import uqra
import numpy as np, os, sys
import scipy.stats as stats
from tqdm import tqdm
import itertools, copy, math, collections
import multiprocessing as mp
# warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()
class Data():
    pass
def observation_error(y, mu=0, cov=0.03, random_state=100):
    e = stats.norm(0, cov * abs(y)).rvs(size=len(y), random_state=random_state)
    return e

def run_UQRA_OptimalDesign(x, poly, doe_sampling, optimality, n_samples, optimal_samples=[], active_index=None):
    optimal_samples = 'RRQR' if len(optimal_samples) == 0 else copy.deepcopy(optimal_samples)
    x = poly.deg**0.5 * x if doe_sampling.lower() in ['cls4', 'cls5'] else x
    if doe_sampling.lower().startswith('cls'):
        X   = poly.vandermonde(x)
        X   = poly.num_basis**0.5*(X.T / np.linalg.norm(X, axis=1)).T
    else:
        X = poly.vandermonde(x)
    if active_index is None:
        X = X
    else:
        X = X[:, active_index]
    uqra.blockPrint()
    doe = uqra.OptimalDesign(X)
    idx = doe.samples(optimality, n_samples, initialization=optimal_samples) ## additional n_samples new samples
    uqra.enablePrint()
    if isinstance(optimal_samples, (list, tuple)):
        idx = [i for i in idx if i not in optimal_samples]
    assert len(idx) == n_samples
    return idx


def isOverfitting(cv_err):
    if len(cv_err) < 3 :
        return False
    if cv_err[-1] > cv_err[-2] and cv_err[-2] > cv_err[0]:
        print('WARNING: Overfitting')
        return False

def isConverge(pf, abs_tol=1e-4):
    if len(pf) < 2:
        return False
    if abs(pf[-2]-pf[-1]) > abs_tol:
        return False
    else:
        return True

def check_converge(pf, abs_tol=1e-4):
    """
    Return Y/N, converged pf/ raw pf, relative changes 
    """
    is_converge = isConverge(pf, abs_tol)
    if len(pf) >= 2:
        abs_err = abs(pf[-2]-pf[-1])
    else:
        abs_err = None

    if is_converge:
        res = (True,  pf[-1], abs_err)
    else:
        res = (False, pf    , abs_err)

    return res
# def isConverge(y0, pf, e=0.025, abs_tol=1e-4):
    # if len(y0) < 2 or len(pf) < 2:
        # return False
    # if abs(y0[-2]-y0[-1])/abs(y0[-2]) > e:
        # return False
    # elif abs(pf[-2]-pf[-1]) > abs_tol:
        # return False
    # else:
        # return True

# def check_converge(y0, e=0.025):
    # """
    # Return Y/N, converged y0/ raw y0, relative changes 
    # """
    # is_converge = isConverge(y0, e)
    # if len(y0) >= 2:
        # rel_change = abs(y0[-2]-y0[-1])/abs(y0[-2])
    # else:
        # rel_change = None
    # if is_converge:
        # res = (True, y0[-1], rel_change)
    # else:
        # res = (False,y0    , rel_change)
    # return res
def main(r=0):
    ## ------------------------ Displaying set up ------------------- ###
    print('\n#################################################################################')
    print(' >>>  Start UQRA : {:d}'.format(r), __file__)
    print('#################################################################################\n')
    np.random.seed(100)
    np.set_printoptions(precision=4)
    np.set_printoptions(threshold=1000)
    np.set_printoptions(suppress=True)
    pf = np.array([5.78e-2])
    n_jobs = mp.cpu_count()
    ## ------------------------ Define solver ----------------------- ###
    # solver      = uqra.ExpAbsSum(stats.uniform(-1,2),d=2,c=[-2,1],w=[0.25,-0.75])
    # solver      = uqra.ExpSquareSum(stats.uniform(-1,2),d=2,c=[1,1],w=[1,0.5])
    # solver      = uqra.CornerPeak(stats.uniform(-1,2), d=2)
    # solver      = uqra.ProductPeak(stats.uniform(-1,2), d=2,c=[-3,2],w=[0.5,0.5])
    # solver      = uqra.Franke()
    # solver      = uqra.Ishigami()

    # solver      = uqra.ExpAbsSum(stats.norm(0,1),d=2,c=[-2,1],w=[0.25,-0.75])
    # solver      = uqra.ExpSquareSum(stats.norm(0,1),d=2,c=[1,1],w=[1,0.5])
    # solver      = uqra.CornerPeak(stats.norm(0,1), d=3, c=np.array([1,2,3]), w=[0.5,]*3)
    # solver      = uqra.ProductPeak(stats.norm(0,1), d=2, c=[-3,2], w=[0.5,]*2)
    # solver      = uqra.ExpSum(stats.norm(0,1), d=3)
    # solver      = uqra.FourBranchSystem()
    solver      = uqra.InfiniteSlope()

    ## ------------------------ UQRA Modeling Parameters ----------------- ###
    model_params = uqra.Modeling()
    model_params.name    = 'PCE'
    model_params.degs    = np.arange(2,15) #[2,6,10]#
    model_params.ndim    = solver.ndim
    model_params.basis   = 'Leg'
    model_params.fitting = 'OLSLAR' 
    model_params.n_splits= 50
    model_params.alpha   = 2
    model_params.num_test= int(1e6)
    model_params.num_pred= int(1e6)
    model_params.info()
    ## ------------------------ UQRA DOE Parameters ----------------- ###
    doe_params = uqra.ExperimentParameters()
    doe_params.doe_sampling = 'CLS1' 
    doe_params.optimality   = ['S']
    doe_params.poly_name    = model_params.basis 
    doe_params.num_cand     = int(1e5)
    # data_dir_cand   = '/Users/jinsongliu/BoxSync/Research/Working_Papers/OE2020_LongTermExtreme/Data/FPSO_SURGE/UniformBall'
    if doe_params.doe_sampling.lower() == 'lhs':
        data_dir_optimal = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/ExperimentalDesign/LHS'
        doe_params.update_output_dir(data_dir_optimal=data_dir_optimal)

    optimal_samples = []
    ndim_deg_cases  = np.array(list(itertools.product([model_params.ndim,], model_params.degs)))
    output_ndim_deg = []
    deg_stop_cv_err = [] 
    # deg_stop_y0_hat = [] 
    deg_stop_pf_hat = [] 
    for ndim, deg in ndim_deg_cases:
        print('    ==================================================================================')
        print('    ==================================================================================')
        ## ------------------------ UQRA Surrogate model----------------- ###
        orth_poly = uqra.poly.orthogonal(ndim, deg, model_params.basis)
        pce_model = uqra.PCE(orth_poly)
        if orth_poly.dist_name.lower() == 'norm':
            dist_u = stats.norm() 
        elif orth_poly.dist_name.lower() == 'uniform':
            dist_u = stats.uniform(-1,2)
        else:
            raise ValueError(' {} not defined'.format(orth_poly.dist_name)) 
        dist_xi   = orth_poly.dist_u
        dist_x    = solver.distributions
        pce_model.info()

        ## ------------------------ Updating DoE parameters ----------------- ###
        idoe_params = copy.deepcopy(doe_params)
        idoe_params.ndim = ndim
        idoe_params.deg  = int(deg)
        ## Specify filename template function
        # idoe_params.update_filenames(r, filename_template)
        ## If not specified, default values will be used
        idoe_params.update_filenames(r)
        ### return data dirctories and filenames
        filename_cand   = idoe_params.fname_cand
        data_dir_cand   = idoe_params.data_dir_cand
        data_dir_optimal= idoe_params.data_dir_optimal

        ## ------------------------ UQRA Simulation Parameters ----------------- ###
        sim_params = uqra.Simulation(solver, pce_model, idoe_params)
        filename_test   = lambda r: r'Mcs5E7R{:d}'.format(r)
        sim_params.update_filenames(r, filename_test)
        filename_testin = sim_params.fname_testin
        filename_test   = sim_params.fname_test
        data_dir_result = sim_params.data_dir_result
        figure_dir      = sim_params.figure_dir
        data_dir_test   = sim_params.data_dir_test
        data_dir_testin = sim_params.data_dir_testin

        print('   > {:<25s}'.format('Input/Output Directories:'))
        print('     - {:<23s} : {:s}'.format(' Candiate samples'  , data_dir_cand))
        print('     - {:<23s} : {:s}'.format(' UQRA DoE data '    , data_dir_optimal))
        print('     - {:<23s} : {:s}'.format(' Test input '       , data_dir_testin))
        print('     - {:<23s} : {:s}'.format(' Test output'       , data_dir_test))
        print('     - {:<23s} : {:s}'.format(' UQRA output data ' , data_dir_result))
        print('     - {:<23s} : {:s}'.format(' UQRA output figure', figure_dir))
        print('   > {:<25s}'.format('Input/Output files'))
        print('     - {:<23s} : {}'.format(' Cadidate samples'  , filename_cand  ))
        print('     - {:<23s} : {}'.format(' Test input data'   , filename_testin))
        print('     - {:<23s} : {}'.format(' Test output data'  , filename_test  ))
        if filename_cand:
            data_cand = np.load(os.path.join(data_dir_cand, filename_cand))[:ndim, :idoe_params.num_cand]
            print('     ..{:<23s} : {}'.format(' Candidate samples', data_cand.shape))

        ### 2. Get test data set
        data_test = np.load(os.path.join(data_dir_test, filename_test), allow_pickle=True).tolist()
        print(data_test.__dict__.keys())
        data_test.x = solver.map_domain(data_test.u, dist_u) 
        data_test.xi= data_test.u
        data_test.y = solver.run(data_test.x)
        u_test      = data_test.xi[:, :model_params.num_test] 
        y_test      = data_test.y

        output_indim_ideg = uqra.Data()
        if idoe_params.doe_sampling.lower() == 'lhs':
            all_doe_cases = [(idoe_params.doe_sampling, None)]
        else:
            all_doe_cases = [(idoe_params.doe_sampling, ioptimality) for ioptimality in idoe_params.optimality] 

        for idoe_sampling, ioptimality in all_doe_cases:
            idoe_sampling = idoe_sampling.lower()
            idoe_nickname = idoe_params.doe_nickname(idoe_sampling, ioptimality)
            print('   --------------------------------------------------------------------------------')
            print('   >> UQRA Training with Experimental Design {} '.format(idoe_nickname))
            ### temp data object containing results from intermedia steps
            data_temp= uqra.Data()
            # data_temp.y0_hat   = []
            data_temp.pf_hat   = []
            data_temp.cv_err   = []
            data_temp.kappa    = []
            data_temp.rmse_y   = []
            data_temp.model    = []
            data_temp.score    = []
            data_temp.ypred_ecdf=[]
            optimal_samples_ideg=[]
            print('   1. Optimal samples based on FULL basis')
            n_samples = 5*(deg == model_params.degs[0])+orth_poly.num_basis
            print('     - {:s}: adding {:d} optimal samples'.format(idoe_nickname, n_samples))
            idx = run_UQRA_OptimalDesign(data_cand, orth_poly, idoe_sampling, ioptimality, n_samples)
            optimal_samples_ideg = optimal_samples_ideg + idx
            optimal_samples      = optimal_samples      + idx 
            assert n_samples == len(idx)
            print('     - # optimal samples [p={:d}]: {:d}'.format(deg, len(optimal_samples_ideg)))
            print('     - Total number of optimal samples: {:d}'.format(len(optimal_samples)))
            print('     - {:s} with (n={:d}, alpha={:.2f}) samples'.format(model_params.fitting.upper(),
                len(optimal_samples), len(optimal_samples)/orth_poly.num_basis))
            u_train = data_cand[:, optimal_samples] 
            if idoe_sampling.lower()=='cls4':
                u_train = u_train * deg **0.5
            x_train = solver.map_domain(u_train, dist_xi)
            y_train = solver.run(x_train)
            # y_train = y_train + observation_error(y_train)
            ## condition number, kappa = max(svd)/min(svd)
        
            pce_model.fit(model_params.fitting, u_train, y_train, w=idoe_sampling, n_jobs=n_jobs)
            y_test_hat = pce_model.predict(u_test, n_jobs=n_jobs)
            data_temp.rmse_y.append(uqra.metrics.mean_squared_error(y_test, y_test_hat, squared=False))
            data_temp.model.append(pce_model)
            data_temp.pf_hat.append(np.sum(y_test_hat<0)/len(y_test_hat))
            print('      pf test [PCE]  : {:.2e}/{:d}'.format(np.sum(y_test_hat<0)/len(y_test_hat), len(y_test_hat)))
            # print('      y0 pred [TRUE] : {:.4f}'.format(uqra.metrics.mquantiles(solver.run(data_pred.x), prob=1-pf)))
            # data_temp.ypred_ecdf.append(uqra.ECDF(y_test_hat, alpha=pf, compress=True))
            # data.y0_ecdf=y0_ecdf
            data_temp.score.append(pce_model.score)
            data_temp.cv_err.append(pce_model.cv_error)
            is_converge, pf_hat, pf_hat_err = check_converge(data_temp.pf_hat, abs_tol=1e-2)
            active_basis = pce_model.active_basis 
            active_index = pce_model.active_index
            print('     - # Active basis: {:d}'.format(len(active_index)))
            print('     > pf: {}, pf abs_err: {}'.format(np.array(data_temp.pf_hat), pf_hat_err))


            print('   2. Optimal samples based on SIGNIFICANT basis')
            while True:
                ### increase number of samples by n_new
                n_samples = len(active_index)
                print('     - {:s}: Sparsity: {:d}, adding {:d} optimal samples'.format(
                    idoe_nickname,len(active_index),n_samples))
                idx = run_UQRA_OptimalDesign(data_cand, orth_poly, idoe_sampling, ioptimality, n_samples, 
                        optimal_samples=optimal_samples_ideg, active_index=active_index)
                optimal_samples      = optimal_samples      + idx
                optimal_samples_ideg = optimal_samples_ideg + idx
                assert n_samples == len(idx)
                print('     - # optimal samples [p={:d}]: {:d}'.format(deg, len(optimal_samples_ideg)))
                print('     - Total number of optimal samples: {:d}'.format(len(optimal_samples)))
                print('     - {:s} with (n={:d}, alpha={:.2f}) samples'.format(model_params.fitting.upper(),
                    len(optimal_samples), len(optimal_samples)/orth_poly.num_basis))
                u_train = data_cand[:, optimal_samples] 
                if idoe_sampling.lower()=='cls4':
                    u_train = u_train * deg **0.5
                x_train = solver.map_domain(u_train, dist_xi)
                y_train = solver.run(x_train)
                # y_train = y_train + observation_error(y_train)
                # w = pce_model.christoffel_weight(u_train, active=active_index) if idoe_sampling.lower().startswith('cls') else None
                pce_model.fit(model_params.fitting, u_train, y_train, w=idoe_sampling, n_jobs=n_jobs)
                y_test_hat = pce_model.predict(u_test, n_jobs=n_jobs)
                data_temp.rmse_y.append(uqra.metrics.mean_squared_error(y_test, y_test_hat, squared=False))
                data_temp.model.append(pce_model)
                data_temp.pf_hat.append(np.sum(y_test_hat<0)/len(y_test_hat))
                print('      pf test [PCE]  : {:.2e}/{:d}'.format(np.sum(y_test_hat<0)/len(y_test_hat), len(y_test_hat)))
                # print('      y0 pred [TRUE] : {:.4f}'.format(uqra.metrics.mquantiles(solver.run(data_pred.x), prob=1-pf)))
                # data_temp.ypred_ecdf.append(uqra.ECDF(y_test_hat, alpha=pf, compress=True))
                # data.y0_ecdf=y0_ecdf
                data_temp.score.append(pce_model.score)
                data_temp.cv_err.append(pce_model.cv_error)
                active_index = pce_model.active_index
                active_basis = pce_model.active_basis 
                isOverfitting(data_temp.cv_err) ## check Overfitting
                is_converge, pf_hat, pf_hat_err = check_converge(data_temp.pf_hat, abs_tol=1e-2)
                print('     - # Active basis: {:d}'.format(len(active_index)))
                print('     > pf: {}, pf abs_err: {}'.format(np.array(data_temp.pf_hat), pf_hat_err))
                # print('    ==================================================' )
                if is_converge:
                    print('     !<>! Model converge for order {:d}'.format(deg))
                    # print('    ==================================================' )
                    break
                if len(optimal_samples_ideg)>=2*orth_poly.num_basis:
                    print('     !<>! Number of samples exceeding 2P')
                    # print('    ==================================================' )
                    break

            deg_stop_cv_err.append(data_temp.cv_err[-1])
            deg_stop_pf_hat.append(data_temp.pf_hat[-1])

            data = uqra.Data()
            data.ndim   = ndim
            data.deg    = deg
            data.u_train= u_train
            data.x_train= x_train
            data.y_train= y_train
            data.rmse_y = data_temp.rmse_y[-1]
            data.pf_hat = data_temp.pf_hat[-1]
            # data.y0_hat_x = uqra_env.ppf(pce_model.orth_poly.dist_u.cdf(data.y0_hat_u.reshape(solver.ndim, -1)))
            data.cv_err = data_temp.cv_err[-1]
            data.model  = data_temp.model[-1]
            data.score  = data_temp.score[-1]
            # data.ypred_ecdf = data_temp.ypred_ecdf[-1]

            print(' ------------------------------')
            tqdm.write(' > Summary PCE: ndim={:d}, p={:d}'.format(ndim, deg))
            # tqdm.write('     - {:<15s} : {}'.format( 'QoI'       , QoI))
            tqdm.write('     - {:<15s} : {}'.format( 'RMSE y ' , np.array(data.rmse_y)))
            tqdm.write('     - {:<15s} : {}'.format( 'CV MSE'  , np.array(data.cv_err)))
            tqdm.write('     - {:<15s} : {}'.format( 'Score '  , np.array(data.score)))
            # tqdm.write('     - {:<15s} : {}'.format( 'kappa '   , data.kappa))
            tqdm.write('     - {:<15s} : {} [{}]'.format( 'pf ' , np.array(data.pf_hat), pf_pred))

            print(uqra.metrics.mquantiles(y_test_hat, 1-pf))
            setattr(output_indim_ideg, idoe_nickname, data)
        output_ndim_deg.append(output_indim_ideg)
        isOverfitting(deg_stop_cv_err) ## check Overfitting
        is_converge, pf_hat, pf_hat_err = check_converge(deg_stop_pf_hat, abs_tol=1e-2)
        print('     > pf: {}, pf abs_err: {}'.format(np.array(deg_stop_pf_hat), pf_hat_err))
        if is_converge:
            print('Simulation Done')
            print(' ------------------------------')
            break

    ## ============ Saving QoIs ============
    filename = '{:s}_Adap{:s}_{:s}E5R{:d}'.format(solver.nickname, pce_model.tag, doe_params.doe_sampling.capitalize(), r)
    try:
        np.save(os.path.join(data_dir_result, filename), output_ndim_deg, allow_pickle=True)
        print(' >> Simulation Done! Data saved to {:s}'.format(os.path.join(data_dir_result, filename)))
    except:
        np.save(os.path.join(os.getcwd(), filename), output_ndim_deg, allow_pickle=True)
        print(' >> Simulation Done! Data saved to {:s}'.format(os.path.join(os.getcwd(), filename)))

if __name__ == '__main__':
    main(0)
