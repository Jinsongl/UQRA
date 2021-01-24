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
import random
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

# def isConverge(pf, abs_tol=1e-4):
    # if len(pf) < 2:
        # return False
    # if abs(pf[-2]-pf[-1]) > abs_tol:
        # return False
    # else:
        # return True

# def check_converge(pf, abs_tol=1e-4):
    # """
    # Return Y/N, converged pf/ raw pf, relative changes 
    # """
    # is_converge = isConverge(pf, abs_tol)
    # if len(pf) >= 2:
        # abs_err = abs(pf[-2]-pf[-1])
    # else:
        # abs_err = None

    # if is_converge:
        # res = (True,  pf[-1], abs_err)
    # else:
        # res = (False, pf    , abs_err)

    # return res

def isConverge(y0, e=0.025):
    if len(y0) < 2:
        return False
    if abs(y0[-2]-y0[-1])/abs(y0[-2]) > e:
        return False
    else:
        return True

def check_converge(data, rel_err= 0.025, abs_err=1e-4, threshold=0.95):
    """
    self-defined coverge function
    """
    
    status  = []
    res     = {}
    ### check exceedance value
    is_converge = False
    y0 = np.array(data.y0_hat)
    if len(y0) < 2:
        is_converge = False
        rel_error  = np.nan
    else:
        rel_error = abs(y0[-2]-y0[-1])/abs(y0[-2])
        if rel_error < rel_err:
            is_converge=True
        else:
            is_converge=False

    status.append(is_converge)
    res['y0'] = (list(y0), rel_error)

    ### check fitting score 
    score = data.score[-1]
    is_converge= True if score > threshold else False
    status.append(is_converge)
    res['score'] = (score, np.nan)
    is_converge = np.array(status).all()
    return is_converge, res

def main(model_params, doe_params, solver, r=0, random_state=None):
    data = uqra.Data()
    data.ndim       = []
    data.deg        = []
    data.xi_train   = []
    data.x_train    = []
    data.y_train    = [] 
    data.rmse_y     = []
    data.y0_hat     = [] 
    data.cv_err     = [] 
    data.model      = []
    data.score      = []
    data.yhat_ecdf  = [] 

    optimal_samples = []
    ndim_deg_cases  = np.array(list(itertools.product([model_params.ndim,], model_params.degs)))

    for i, (ndim, deg) in enumerate(ndim_deg_cases):
        print('==================================================================================')
        print('     Iteration No. {:d} '.format(i))
        print('==================================================================================')
        ## ------------------------ UQRA Surrogate model----------------- ###
        orth_poly = uqra.poly.orthogonal(ndim, deg, model_params.basis)
        pce_model = uqra.PCE(orth_poly)
        dist_u    = model_params.dist_u 
        dist_xi   = orth_poly.weight
        dist_x    = solver.distributions
        pce_model.info()

        ## ------------------------ Updating DoE parameters ----------------- ###
        idoe_params = copy.deepcopy(doe_params)
        idoe_params.ndim = ndim
        idoe_params.deg  = int(deg)
        ### Specify candidate data filename template function
        ### e.g.  filename_template= lambda r: r'DoE_Ball5pt6E5R{:d}.npy'.format(r)
        ### if not specified, default values will be used
        idoe_params.update_filenames(filename_template=None)
        filename_cand   = idoe_params.fname_cand(r)
        # filename_design = idoe_params.fname_design(r)
        print('     - {:<23s} : {}'.format(' Cadidate samples'  , filename_cand  ))

        if filename_cand:
            data_cand = np.load(os.path.join(data_dir_cand, filename_cand))
            idx_cand  = np.arange(data_cand.shape[1])
            random.seed(random_state)
            random.shuffle(idx_cand)
            idx_cand  = idx_cand[:idoe_params.num_cand]
            data_cand = data_cand[:ndim,idx_cand]
            print('     ..{:<23s} : {}'.format(' Candidate samples', data_cand.shape))

        idoe_sampling = idoe_params.doe_sampling.lower()
        idoe_nickname = idoe_params.doe_nickname()
        ioptimality   = idoe_params.optimality
        print('   --------------------------------------------------------------------------------')
        print('   >> UQRA Training with Experimental Design {} '.format(idoe_nickname))
        ### temp data object containing results from intermedia steps
        data_temp= uqra.Data()
        data_temp.y0_hat   = []
        data_temp.cv_err   = []
        data_temp.kappa    = []
        data_temp.rmse_y   = []
        data_temp.model    = []
        data_temp.score    = []
        data_temp.yhat_ecdf= []
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
        xi_train = data_cand[:, optimal_samples] 
        if idoe_sampling.lower()=='cls4':
            xi_train = xi_train * deg **0.5
        x_train = solver.map_domain(xi_train, dist_xi)
        y_train = solver.run(x_train)
    
        pce_model.fit(model_params.fitting, xi_train, y_train, w=idoe_sampling, n_jobs=model_params.n_jobs, n_splits=model_params.n_splits)
        y_test_hat = pce_model.predict(xi_test, n_jobs=model_params.n_jobs)
        data_temp.model.append(pce_model)
        data_temp.rmse_y.append(uqra.metrics.mean_squared_error(y_test, y_test_hat, squared=False))
        data_temp.y0_hat.append(uqra.metrics.mquantiles(y_test_hat, 1-model_params.pf))
        data_temp.score.append(pce_model.score)
        data_temp.cv_err.append(pce_model.cv_error)
        data_temp.yhat_ecdf.append(uqra.ECDF(y_test_hat, model_params.pf, compress=True))
        active_basis = pce_model.active_basis 
        active_index = pce_model.active_index
        print('       #|Active basis|: {:d}'.format(len(active_index)))
        print('       y0 test [PCE ] : {:.4e}'.format(data_temp.y0_hat[-1]))
        print('       y0 test [TRUE] : {:.4e}'.format(y0_test))
        is_converge, check_res = check_converge(data_temp)
        for ikey, ivalue in check_res.items():
            print('     > {}: {}, {:.2e}'.format(ikey, np.array(ivalue[0]), ivalue[1]))
        print('     ------------------------------------------------------------')

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
            xi_train = data_cand[:, optimal_samples] 
            if idoe_sampling.lower()=='cls4':
                xi_train = xi_train * deg **0.5
            x_train = solver.map_domain(xi_train, dist_xi)
            y_train = solver.run(x_train)
            # y_train = y_train + observation_error(y_train)
            pce_model.fit(model_params.fitting, xi_train, y_train, w=idoe_sampling,n_jobs=model_params.n_jobs, n_splits=model_params.n_splits)
            y_test_hat = pce_model.predict(xi_test, n_jobs=model_params.n_jobs)
            data_temp.model.append(pce_model)
            data_temp.rmse_y.append(uqra.metrics.mean_squared_error(y_test, y_test_hat, squared=False))
            data_temp.y0_hat.append(uqra.metrics.mquantiles(y_test_hat, 1-model_params.pf))
            data_temp.score.append(pce_model.score)
            data_temp.cv_err.append(pce_model.cv_error)
            data_temp.yhat_ecdf.append(uqra.ECDF(y_test_hat, model_params.pf, compress=True))
            # isOverfitting(data_temp.cv_err) ## check Overfitting
            active_index = pce_model.active_index
            active_basis = pce_model.active_basis 
            print('       #|Active basis|: {:d}'.format(len(active_index)))
            print('       y0 test [PCE ] : {:.4e}'.format(data_temp.y0_hat[-1]))
            print('       y0 test [TRUE] : {:.4e}'.format(y0_test))
            is_converge, check_res = check_converge(data_temp)
            for ikey, ivalue in check_res.items():
                print('     > {}: {}, {:.2e}'.format(ikey, np.array(ivalue[0]), ivalue[1]))
            print('     ------------------------------------------------------------')
            if is_converge:
                print('  !<>! Model converge for order {:d}'.format(deg))
                break
            if len(optimal_samples_ideg)>=2*orth_poly.num_basis:
                print('  !<>! Number of samples exceeding 2P')
                break
        print('     ------------------------------------------------------------')
        tqdm.write('    > Summary PCE: ndim={:d}, p={:d}'.format(ndim, deg))
        tqdm.write('     - {:<15s} : {:.4e}'.format( 'RMSE y ' , data_temp.rmse_y[-1]))
        tqdm.write('     - {:<15s} : {:.4e}'.format( 'CV MSE'  , data_temp.cv_err[-1]))
        tqdm.write('     - {:<15s} : {:.4f}'.format( 'Score '  , data_temp.score[-1] ))
        tqdm.write('     - {:<15s} : {:.4e} [{:.4e}]'.format( 'y0 ' , data_temp.y0_hat[-1], y0_test))

        data.ndim.append(ndim)
        data.deg.append(deg)
        data.xi_train.append( xi_train)
        data.x_train.append( x_train)
        data.y_train.append( y_train)
        data.rmse_y.append ( data_temp.rmse_y[-1])
        data.y0_hat.append ( data_temp.y0_hat[-1])
        data.cv_err.append ( data_temp.cv_err[-1])
        data.model.append  ( data_temp.model[-1])
        data.score.append  ( data_temp.score[-1])
        data.yhat_ecdf.append(data_temp.yhat_ecdf[-1])

        isOverfitting(data.cv_err) ## check Overfitting
        is_converge, check_res = check_converge(data)
        for ikey, ivalue in check_res.items():
            print('     > {}: {}, {:.2e}'.format(ikey, np.array(ivalue[0]), ivalue[1]))
        if is_converge:
            tqdm.write('###############################################################################')
            tqdm.write('         Model Converge ')
            tqdm.write('    > Summary PCE: ndim={:d}, p={:d}'.format(ndim, deg))
            tqdm.write('     - {:<15s} : {}'.format( 'RMSE y ' , np.array(data.rmse_y)))
            tqdm.write('     - {:<15s} : {}'.format( 'CV MSE'  , np.array(data.cv_err)))
            tqdm.write('     - {:<15s} : {}'.format( 'Score '  , np.array(data.score)))
            tqdm.write('     - {:<15s} : {} [{:.2e}]'.format( 'y0 ' , np.array(data.y0_hat), y0_test))
            for ikey, ivalue in check_res.items():
                tqdm.write('     > {}: {}, {:.2e}'.format(ikey, np.array(ivalue[0]), ivalue[1]))
            tqdm.write('###############################################################################')
            break
    return data

if __name__ == '__main__':
    ## ------------------------ Displaying set up ------------------- ###
    r = 0
    np.random.seed(100)
    np.set_printoptions(precision=4)
    np.set_printoptions(threshold=1000)
    np.set_printoptions(suppress=True)
    ## ------------------------ Define solver ----------------------- ###
    # solver      = uqra.ExpAbsSum(stats.uniform(-1,2),d=2,c=[-2,1],w=[0.25,-0.75])
    # solver      = uqra.ExpSquareSum(stats.uniform(-1,2),d=2,c=[1,1],w=[1,0.5])
    # solver      = uqra.CornerPeak(stats.uniform(-1,2), d=2)
    # solver      = uqra.ProductPeak(stats.uniform(-1,2), d=2,c=[-3,2],w=[0.5,0.5])
    # solver      = uqra.Franke()
    solver      = uqra.Ishigami()

    # solver      = uqra.ExpAbsSum(stats.norm(0,1),d=2,c=[-2,1],w=[0.25,-0.75])
    # solver      = uqra.ExpSquareSum(stats.norm(0,1),d=2,c=[1,1],w=[1,0.5])
    # solver      = uqra.CornerPeak(stats.norm(0,1), d=3, c=np.array([1,2,3]), w=[0.5,]*3)
    # solver      = uqra.ProductPeak(stats.norm(0,1), d=2, c=[-3,2], w=[0.5,]*2)
    # solver      = uqra.ExpSum(stats.norm(0,1), d=3)
    # solver      = uqra.FourBranchSystem()
    # solver      = uqra.LiqudHydrogenTank()

    ## ------------------------ UQRA Modeling Parameters ----------------- ###
    model_params = uqra.Modeling('PCE')
    model_params.degs    = np.arange(2,15) #[2,6,10]#
    model_params.ndim    = solver.ndim
    model_params.basis   = 'Leg'
    model_params.dist_u  = stats.uniform(0,1)  #### random CDF values for samples
    model_params.fitting = 'OLSLAR' 
    model_params.n_splits= 50
    model_params.alpha   = 2
    model_params.num_test= int(1e6)
    model_params.num_pred= int(1e6)
    model_params.pf      = np.array([1e-4])
    model_params.n_jobs  = mp.cpu_count()
    model_params.update_basis()
    model_params.info()
    ## ------------------------ UQRA DOE Parameters ----------------- ###
    doe_params = uqra.ExperimentParameters('CLS1', 'S')
    doe_params.poly_name    = model_params.basis 
    doe_params.num_cand     = int(1e5)

    ## ------------------------ UQRA Simulation Parameters ----------------- ###
    sim_params = uqra.Simulation(solver, model_params, doe_params)
    filename_test   = lambda r: r'McsE6R{:d}'.format(r)
    sim_params.update_filenames(filename_test)

    data_dir_cand   = doe_params.data_dir_cand
    data_dir_optimal= doe_params.data_dir_optimal
    filename_testin = sim_params.fname_testin(r)
    filename_test   = sim_params.fname_test(r)
    data_dir_result = sim_params.data_dir_result
    figure_dir      = sim_params.figure_dir
    data_dir_test   = sim_params.data_dir_test
    data_dir_testin = sim_params.data_dir_testin


    ### 1. Get test data set
    data_test   = np.load(os.path.join(data_dir_test, filename_test), allow_pickle=True).tolist()
    data_test.x = solver.map_domain(data_test.u, model_params.dist_u)
    data_test.xi= model_params.map_domain(data_test.u, model_params.dist_u)
    data_test.y = solver.run(data_test.x) if not hasattr(data_test, 'y') else data_test.y
    xi_test     = data_test.xi[:, :model_params.num_test] 
    y_test      = data_test.y[:model_params.num_test] 
    y0_test     = uqra.metrics.mquantiles(y_test, 1-model_params.pf)

    res = []
    nrepeat = 25
    for irepeat in range(nrepeat):
        print('\n#################################################################################')
        print(' >>>  File: ', __file__)
        print(' >>>  Start UQRA : {:d}/{:d}'.format(irepeat, nrepeat))
        print(' >>>  Test data R={:d}'.format(r))
        print('#################################################################################\n')
        print('   > {:<25s}'.format('Input/Output Directories:'))
        print('     - {:<23s} : {:s}'.format(' Candiate samples'  , data_dir_cand))
        print('     - {:<23s} : {:s}'.format(' UQRA DoE data '    , data_dir_optimal))
        print('     - {:<23s} : {:s}'.format(' Test input '       , data_dir_testin))
        print('     - {:<23s} : {:s}'.format(' Test output'       , data_dir_test))
        print('     - {:<23s} : {:s}'.format(' UQRA output data ' , data_dir_result))
        print('     - {:<23s} : {:s}'.format(' UQRA output figure', figure_dir))
        print('   > {:<25s}'.format('Input/Output files'))
        print('     - {:<23s} : {}'.format(' Test input data'   , filename_testin))
        print('     - {:<23s} : {}'.format(' Test output data'  , filename_test  ))
        res.append(main(model_params, doe_params, solver, r=r, random_state=irepeat))
    filename = '{:s}_Adap{:d}{:s}_{:s}E5R{:d}'.format(solver.nickname, solver.ndim, model_params.basis,doe_params.doe_sampling.capitalize(), r)
    # ## ============ Saving QoIs ============
    try:
        np.save(os.path.join(data_dir_result, filename), res, allow_pickle=True)
        print(' >> Simulation Done! Data saved to {:s}'.format(os.path.join(data_dir_result, filename)))
    except:
        np.save(filename, res, allow_pickle=True)
        print(' >> Simulation Done! Data saved to {:s}'.format(os.path.join(os.getcwd(), filename)))
