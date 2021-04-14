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

def observation_error(y, mu=0, cov=0.03, random_state=100):
    e = stats.norm(0, cov * abs(y)).rvs(size=len(y), random_state=random_state)
    return e

def list_union(ls1, ls2):
    """
    append ls2 to ls1 and check if there exist duplicates
    return the union of two lists and remove duplicates
    """
    if ls1 is None:
        ls1 = []
    if ls2 is None:
        ls2 = []
    ls = list(copy.deepcopy(ls1)) + list(copy.deepcopy(ls2))
    if len(ls) != len(set(ls1).union(set(ls2))):
        print('[WARNING]: list_union: duplicate elements found in list when append to each other')
    ls = list(set(ls))
    return ls

def isOverfitting(cv_err):
    if len(cv_err) < 3 :
        return False
    if cv_err[-1] > cv_err[-2] and cv_err[-2] > cv_err[0]:
        print('WARNING: Overfitting')
        return False

def threshold_converge(y, threshold=0.95):
    y = np.array(y)
    status = True if y[-1]> threshold else False
    return status, threshold

def relative_converge(y, err=0.05):
    """ 
    check if y is converge in relative error
    return: (status, error)
        status: Boolean for convergeing or not
        error: absolute error

    """
    y = np.array(y)
    if len(y) < 2:
        res = (False, np.nan)
    else:
        error = abs((y[-2]-y[-1])/ y[-1])
        res = (error < err, error)
    return res 

def absolute_converge(y, err=1e-4):
    """ 
    check if y is converge in absolute error
    return: (status, error)
        status: Boolean for convergeing or not
        error: absolute error

    """
    y = np.array(y)
    if len(y) < 2:
        res = (False, np.nan)
    else:
        error = abs(y[-2]-y[-1])
        res = (error < err, error)
    return res 

def main(model_params, doe_params, solver, r=0, random_state=None):
    random.seed(random_state)
    ndim_deg_cases = np.array(list(itertools.product([model_params.ndim,], model_params.degs)))

    main_res = []
    # data_train = uqra.Data()
    # data_train.xi = np.empty((model_params.ndim, 0))
    # data_train.x  = np.empty((model_params.ndim, 0))
    # data_train.y  = np.empty((0,))

    xi_train= np.empty((solver.ndim, 0))
    x_train = np.empty((solver.ndim, 0))
    y_train = []
    idx_selected= []
    for i, (ndim, deg) in enumerate(ndim_deg_cases):
        print('\n==================================================================================')
        print('         <<<< Global iteration No. {:d}: ndim={:d}, p={:d} >>>>'.format(i+1, ndim, deg))
        print('==================================================================================\n')
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
        filename_cand = idoe_params.fname_cand(r)
        # filename_design = idoe_params.fname_design(r)
        print('     - {:<23s} : {}'.format(' Candidate filename'  , filename_cand  ))

        if filename_cand:
            data_cand = np.load(os.path.join(data_dir_cand, filename_cand))
            data_cand = data_cand[:ndim,random.sample(range(data_cand.shape[1]), k=idoe_params.num_cand)]
            print('       {:<23s} : {}'.format(' shape', data_cand.shape))
        else:
            data_cand = None
            print('       {:<23s} : {}'.format(' shape', data_cand))

        idoe_sampling = idoe_params.doe_sampling.lower()
        idoe_nickname = idoe_params.doe_nickname()
        ioptimality   = idoe_params.optimality
        print('     - {:<23s} : {}'.format(' UQRA DoE '  , idoe_nickname))
        ### temp data object containing results from intermedia steps
        data_ideg = uqra.Data()
        data_ideg.ndim      = ndim
        data_ideg.deg       = deg 

        ## ------------------------ #1: Obtain global optimal samples ----------------- ###
        print(' ------------------------------------------------------------')
        print(' > Adding optimal samples in global domain... ')
        print('   1. optimal samples based on FULL basis')
        active_index = pce_model.active_index
        active_basis = pce_model.active_basis
        # if deg == model_params.degs[0]:
            # n_samples = math.ceil(len(active_index) * model_params.alpha)
        # else:
            # n_samples = len(active_index)
        n_samples = math.ceil(len(active_index) * model_params.alpha) - len(y_train)
        print('     - Optimal design:{:s}, Adding {:d} optimal samples'.format(idoe_nickname, n_samples))

        ## obtain global optimal samples
        xi_train_, idx_optimal = idoe_params.get_samples(data_cand, orth_poly, n_samples, x0=idx_selected, 
                active_index=None, initialization='RRQR', return_index=True) 
        idx_selected = uqra.list_union(idx_optimal, idx_selected)

        xi_train = np.concatenate((xi_train, xi_train_), axis=1)
        x_train = solver.map_domain(xi_train, dist_xi)
        y_train = solver.run(x_train)
        print('     - {:<32s} : {:d}'.format('No. optimal samples [p='+str(deg)+']', xi_train_.shape[1]))
        print('     - {:<32s} : {:d}'.format('Total number of samples', len(y_train)))


        print('   2. Training with {} '.format(model_params.fitting))
        weight  = doe_params.sampling_weight()   ## weight function
        pce_model.fit(model_params.fitting, xi_train, y_train, w=weight,
                n_jobs=model_params.n_jobs) #, n_splits=model_params.n_splits
        print('     - {:<32s} : ({},{}),    Alpha: {:.2f}'.format('X train', x_train.shape[1], pce_model.num_basis, 
                        x_train.shape[1]/pce_model.num_basis))
        print('     - {:<32s} : {}'.format('Y train'    , y_train.shape))
        print('     - {:<32s} : {}'.format('Sparsity'   , len(pce_model.active_index)))

        print('   3. Prediction with {} samples '.format(xi_test.shape))
        y_test_hat = pce_model.predict(xi_test, n_jobs=model_params.n_jobs)
        data_ideg.model     = pce_model
        data_ideg.rmse_y    = uqra.metrics.mean_squared_error(y_test, y_test_hat, squared=False)
        data_ideg.y0_hat    = uqra.metrics.mquantiles(y_test_hat, 1-model_params.pf)
        data_ideg.score     = pce_model.score
        data_ideg.cv_err    = pce_model.cv_error
        data_ideg.yhat_ecdf = uqra.ECDF(y_test_hat, model_params.pf, compress=True)
        data_ideg.xi_train  = xi_train
        data_ideg.x_train   = x_train
        data_ideg.y_train   = y_train
        print('     - {:<32s} : {:.4e}'.format('y0 test [ PCE ]', data_ideg.y0_hat))
        print('     - {:<32s} : {:.4e}'.format('y0 test [TRUE ]', y0_test))

        print(' ------------------------------------------------------------')
        tqdm.write(' > Summary PCE: ndim={:d}, p={:d}'.format(ndim, deg))
        tqdm.write('  - {:<15s} : {:.4e}'.format( 'RMSE y ' , data_ideg.rmse_y))
        tqdm.write('  - {:<15s} : {:.4e}'.format( 'CV MSE'  , data_ideg.cv_err))
        tqdm.write('  - {:<15s} : {:.4f}'.format( 'Score '  , data_ideg.score ))
        tqdm.write('  - {:<15s} : {:.4e} [{:.4e}]'.format( 'y0 ' , data_ideg.y0_hat, y0_test))
        print(' ------------------------------------------------------------')

        main_res.append(data_ideg)

        cv_err_global = np.array([idata.cv_err for idata in main_res]).T
        y0_hat_global = np.array([idata.y0_hat for idata in main_res]).T
        score_global  = np.array([idata.score  for idata in main_res]).T
        rmsey_global  = np.array([idata.rmse_y for idata in main_res]).T
        y0_hat_global = np.array([idata.y0_hat for idata in main_res]).T

        isOverfitting(cv_err_global) ## check Overfitting
        isConverge0, error_converge0 = relative_converge(y0_hat_global, err=2*model_params.rel_err)
        isConverge1, error_converge1 = threshold_converge(score_global)
        error_converge = [error_converge0, error_converge1]
        isConverge = [isConverge0, isConverge1]
        for i, (ikey, ivalue) in enumerate(zip([isConverge0, isConverge1], [error_converge0, error_converge1])):
            print('  >  Checking #{:d} : {}, {:.4f}'.format(i, ikey, ivalue))
        if np.array(isConverge).all():
            tqdm.write('###############################################################################')
            tqdm.write('         Model Converge ')
            tqdm.write('    > Summary PCE: ndim={:d}, p={:d}'.format(ndim, deg))
            tqdm.write('     - {:<15s} : {}'.format( 'RMSE y ' , np.array(rmsey_global)))
            tqdm.write('     - {:<15s} : {}'.format( 'CV MSE'  , np.array(cv_err_global)))
            tqdm.write('     - {:<15s} : {}'.format( 'Score '  , np.array(score_global)))
            tqdm.write('     - {:<15s} : {} [{:.2e}]'.format( 'y0 ' , np.array(y0_hat_global), y0_test))
            for i, (ikey, ivalue) in enumerate(zip(isConverge, error_converge)):
                print('     >  Checking #{:d} : {}, {:.2e}'.format(i, ikey, ivalue))
            tqdm.write('###############################################################################')
            # break
    return main_res

if __name__ == '__main__':
    ## ------------------------ Displaying set up ------------------- ###
    r, theta= 0, 0
    ith_batch  = 0
    batch_size = 1
    np.random.seed(100)
    random.seed(100)
    np.set_printoptions(precision=4)
    np.set_printoptions(threshold=1000)
    np.set_printoptions(suppress=True)
    uqra_env = uqra.environment.Norway5(ndim=2)

    ## ------------------------ Define solver ----------------------- ###
    solver = uqra.FPSO(random_state=theta, distributions=uqra_env)
    ## ------------------------ UQRA Modeling Parameters ----------------- ###
    model_params = uqra.Modeling('PCE')
    model_params.degs    = np.arange(2,11) #[2,6,10]#
    model_params.ndim    = solver.ndim
    model_params.basis   = 'Heme'
    model_params.dist_u  = stats.uniform(0,1)  #### random CDF values for samples
    model_params.fitting = 'OLSLAR' 
    model_params.n_splits= 50
    model_params.alpha   = 2
    model_params.num_test= int(1e7)
    model_params.num_pred= int(1e7)
    model_params.pf      = np.array([0.5/(365.25*24*50)])
    model_params.abs_err = 1e-4
    model_params.rel_err = 2.5e-2
    model_params.n_jobs  = mp.cpu_count()
    model_params.update_basis()
    model_params.info()
    ## ------------------------ UQRA DOE Parameters ----------------- ###
    doe_params = uqra.ExperimentParameters('MCS', 'D')
    # doe_params = uqra.ExperimentParameters('CLS4', 'S')
    doe_params.poly_name = model_params.basis 
    doe_params.num_cand  = int(1e5)

    ## ------------------------ UQRA Simulation Parameters ----------------- ###
    sim_params = uqra.Simulation(solver, model_params, doe_params)
    filename_test   = lambda r: r'McsE7R{:d}'.format(r)
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
    try:
        data_test.y = data_test.y[theta]
    except:
        pass
    xi_test = data_test.xi[:, :model_params.num_test] 
    y_test  = data_test.y [   :model_params.num_test] 
    y0_test = uqra.metrics.mquantiles(y_test, 1-model_params.pf)

    res = []
    for i, irepeat in enumerate(range(batch_size*ith_batch, batch_size*(ith_batch+1))):
        print('\n#################################################################################')
        print(' >>>  File: ', __file__)
        print(' >>>  Start UQRA : {:d}[{:d}]/{:d} x {:d}'.format(i, irepeat, batch_size, ith_batch))
        print(' >>>  Test data R={:d}'.format(r))
        print('#################################################################################\n')
        print('   > {:<25s}'.format('Input/Output Directories:'))
        print('     - {:<23s} : {}'.format  (' Candiate samples'  , data_dir_cand))
        print('     - {:<23s} : {:s}'.format(' UQRA DoE data '    , data_dir_optimal))
        print('     - {:<23s} : {:s}'.format(' Test input '       , data_dir_testin))
        print('     - {:<23s} : {:s}'.format(' Test output'       , data_dir_test))
        print('     - {:<23s} : {:s}'.format(' UQRA output data ' , data_dir_result))
        print('     - {:<23s} : {:s}'.format(' UQRA output figure', figure_dir))
        print('   > {:<25s}'.format('Input/Output files'))
        print('     - {:<23s} : {}'.format(' Test input data'   , filename_testin))
        print('     - {:<23s} : {}'.format(' Test output data'  , filename_test  ))
        res.append(main(model_params, doe_params, solver, r=r, random_state=irepeat))
    if len(res) == 1:
        res = res[0]
    filename = '{:s}_{:d}{:s}_{:s}E5R{:d}_S{:d}_Alpha{:s}_{:d}{:d}'.format(solver.nickname, 
            solver.ndim, model_params.basis, doe_params.doe_nickname(), r, theta, str(model_params.alpha),
            batch_size, ith_batch)
    # ## ============ Saving QoIs ============
    try:
        np.save(os.path.join(data_dir_result, filename), res, allow_pickle=True)
        print(' >> Simulation Done! Data saved to {:s}'.format(os.path.join(data_dir_result, filename)))
    except:
        np.save(filename, res, allow_pickle=True)
        print(' >> Simulation Done! Data saved to {:s}'.format(os.path.join(os.getcwd(), filename)))
