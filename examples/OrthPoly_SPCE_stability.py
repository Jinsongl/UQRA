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
import warnings
# warnings.filterwarnings(action="ignore", module="sklearn")
warnings.filterwarnings(action="ignore")
sys.stdout  = uqra.utilities.classes.Logger()

class Data():
    pass

def observation_error(y, mu=0, cov=0.01, random_state=100):
    e = stats.norm(0, cov * abs(y)).rvs(size=len(y), random_state=random_state)
    return e

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
    ndim, deg = model_params.ndim, model_params.degs

    print('\n==================================================================================')
    print('         <<<< Initial Exploration: ndim={:d}, p={:d} >>>>'.format(ndim, deg))
    print('==================================================================================\n')

    ## ------------------------ Updating DoE parameters ----------------- ###
    doe_params.ndim = ndim
    doe_params.deg  = int(deg)
    ### Specify candidate data filename template function
    ### e.g.  filename_template= lambda r: r'DoE_Ball5pt6E5R{:d}.npy'.format(r)
    ### if not specified, default values will be used
    doe_params.update_filenames(filename_template=None)
    filename_cand = doe_params.fname_cand(r)
    # filename_design = doe_params.fname_design(r)
    if filename_cand: 
        data_cand = np.load(os.path.join(data_dir_cand, filename_cand))
        print('     - {:<23s} : {}'.format(' Candidate filename'  , filename_cand  ))
        print('       {:<23s} : {}'.format(' shape', data_cand.shape))
        if doe_params.doe_sampling.lower() in ['cls4', 'cls5']:
            data_cand = data_cand * model_params.degs**0.5
    else:
        data_cand = None
        print('       {:<23s} : {}'.format(' shape', data_cand))
    print('     - {:<23s} : {}'.format(' UQRA DoE '  , doe_params.doe_nickname()))

    if filename_optimal:
        data_optimal  = np.load(os.path.join(data_dir_optimal, filename_optimal), allow_pickle=True)
    ### data object containing results from intermedia steps
    main_res = []
    orth_poly = uqra.poly.orthogonal(ndim, deg, model_params.basis)
    # while True:
    alpha1 = np.linspace(1,2,6)
    alpha2 = np.linspace(2,5,7)
    alpha3 = np.array([5,10,20,40,80,100])
    alphas = np.unique(np.concatenate([alpha1, alpha2,alpha3], axis=0))
    # for n_samples in np.arange(10,150,5):
    for n_samples in [math.ceil(ialpha * orth_poly.num_basis) for ialpha in alphas]:
        data_nsample = uqra.Data()
        data_nsample.ndim     = ndim
        data_nsample.deg      = deg 
        data_nsample.y0_hat   = []
        data_nsample.cv_err   = []
        data_nsample.rmse_y   = []
        data_nsample.score    = []
        data_nsample.model    = []
        data_nsample.xi_train = []
        data_nsample.x_train  = []
        data_nsample.y_train  = []
        data_nsample.beta_hat = []
        data_nsample.yhat_ecdf= []
        error = []
        print(' ------------------------------------------------------------')
        for i in tqdm(range(50), ascii=True, ncols=80):
            ## ------------------------ UQRA Surrogate model----------------- ###
            orth_poly = uqra.poly.orthogonal(ndim, deg, model_params.basis)
            pce_model = uqra.PCE(orth_poly)
            dist_u    = model_params.dist_u 
            dist_xi   = orth_poly.weight
            dist_x    = solver.distributions

            if doe_params.doe_sampling.lower() in ['lhs']:
                xi_train = uqra.LHS([dist_xi, ] *ndim).samples(size=n_samples)
            else:
                optimal_idx = getattr(data_optimal[i], doe_params.doe_nickname())
                xi_train= data_cand[:solver.ndim, optimal_idx[:n_samples]]
            x_train = solver.map_domain(xi_train, dist_xi)
            y_train = solver.run(x_train)
            y_train = y_train + observation_error(y_train)

            weight  = doe_params.sampling_weight()   ## weight function
            pce_model.fit(model_params.fitting, xi_train, y_train, w=weight,
                    n_jobs=model_params.n_jobs, n_splits=model_params.n_splits) #


            y_test_hat = pce_model.predict(xi_test, n_jobs=model_params.n_jobs)

            beta_hat = np.zeros(pce_model.num_basis)
            for i, beta_i in zip(pce_model.active_index, pce_model.coef):
                beta_hat[i] = beta_i

            data_nsample.rmse_y.append(uqra.metrics.mean_squared_error(y_test, y_test_hat, squared=False))
            data_nsample.y0_hat.append(uqra.metrics.mquantiles(y_test_hat, 1-model_params.pf))
            data_nsample.score.append(pce_model.score)
            data_nsample.cv_err.append(pce_model.cv_error)
            data_nsample.yhat_ecdf.append(uqra.ECDF(y_test_hat, model_params.pf, compress=True))
            data_nsample.model.append(pce_model)
            data_nsample.xi_train.append(xi_train)
            data_nsample.x_train.append(x_train)
            data_nsample.y_train.append(y_train)
            data_nsample.beta_hat.append(beta_hat)
            error.append(np.linalg.norm(beta-beta_hat)/np.linalg.norm(beta))
        error = np.array(error)
        pce_model.info()
        print('   Training with {} '.format(model_params.fitting))
        print('     - {:<32s} : ({},{}),    Alpha: {:.2f}'.format('X train', x_train.shape[1], pce_model.num_basis, 
                        x_train.shape[1]/pce_model.num_basis))
        print('     - {:<32s} : {}'.format('Y train'    , y_train.shape))
        print('     - {:<32s} : {:.4e} {:.4e}'.format('y0 test [ PCE ] (mean, std)', np.mean(data_nsample.y0_hat), np.std(data_nsample.y0_hat)))
        print('     - {:<32s} : {:.4e}'.format('y0 test [TRUE ]', y0_test))
        print('     - {:<32s} : {}'.format('Beta Error<0.01', np.sum(error<0.01)/len(error)))

        print(' ------------------------------------------------------------')
        main_res.append(data_nsample)

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
    ## ------------------------ Define solver ----------------------- ###
    # solver      = uqra.ExpAbsSum(stats.uniform(-1,2),d=2,c=[-2,1],w=[0.25,-0.75])
    # solver      = uqra.ExpSquareSum(stats.uniform(-1,2),d=2,c=[1,1],w=[1,0.5])
    # solver      = uqra.CornerPeak(stats.uniform(-1,2), d=2)
    # solver      = uqra.ProductPeak(stats.uniform(-1,2), d=2,c=[-3,2],w=[0.5,0.5])
    # solver      = uqra.Franke()
    # solver      = uqra.ExpTanh()
    # solver      = uqra.Ishigami()
    # solver      = uqra.InfiniteSlope()

    # solver      = uqra.ExpAbsSum(stats.norm(0,1),d=2,c=[-2,1],w=[0.25,-0.75])
    # solver      = uqra.ExpSquareSum(stats.norm(0,1),d=2,c=[1,1],w=[1,0.5])
    # solver      = uqra.CornerPeak(stats.norm(0,1), d=3, c=np.array([1,2,3]), w=[0.5,]*3)
    # solver      = uqra.ProductPeak(stats.norm(0,1), d=2, c=[-3,2], w=[0.5,]*2)
    # solver      = uqra.ExpSum(stats.norm(0,1), d=3)
    # solver      = uqra.FourBranchSystem()
    # solver      = uqra.LiqudHydrogenTank()
    np.random.seed(10)
    orth_poly = uqra.Hermite(d=2,deg=6, hem_type='probabilists')
    coef = np.zeros((orth_poly.num_basis))
    for i in [0, 1,4,7,12,18,25]:
        coef[i] = stats.norm.rvs(0,1)
    solver = uqra.OrthPoly(orth_poly, coef=coef)
    print(solver.coef)

    ## ------------------------ UQRA Modeling Parameters ----------------- ###
    model_params = uqra.Modeling('PCE')
    model_params.degs    = 6 #np.arange(10,11) #[2,6,10]#
    model_params.ndim    = solver.ndim
    model_params.basis   = 'Heme'
    model_params.dist_u  = stats.uniform(0,1)  #### random CDF values for samples
    model_params.fitting = 'OLSLAR' 
    model_params.n_splits= 10
    model_params.alpha   = 3
    model_params.num_test= int(1e6)
    model_params.num_pred= int(1e6)
    model_params.pf      = np.array([1e-4])
    model_params.abs_err = 1e-4
    model_params.rel_err = 2.5e-2
    model_params.n_jobs  = mp.cpu_count()
    model_params.update_basis()
    model_params.info()
    ## ------------------------ UQRA DOE Parameters ----------------- ###
    doe_params = uqra.ExperimentParameters('MCS', 'D')
    # doe_params = uqra.ExperimentParameters('LHS', None)
    doe_params.update_poly_name(model_params.basis)
    doe_params.num_cand  = int(1e5)

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
    if doe_params.doe_sampling.lower() == 'lhs':
        filename_optimal = None
    else:
        filename_optimal= 'DoE_{:s}E5R0_{:d}{:s}{:d}.npy'.format(doe_params.doe_sampling.capitalize(), model_params.ndim, 
            model_params.basis[:3], model_params.degs)
    ### 1. Get test data set
    data_test   = np.load(os.path.join(data_dir_test, filename_test), allow_pickle=True).tolist()
    data_test.x = solver.map_domain(data_test.u, model_params.dist_u)
    data_test.xi= model_params.map_domain(data_test.u, model_params.dist_u)
    if not hasattr(data_test, 'y'):
        data_test.y = solver.run(data_test.x) 
        data_test.y = data_test.y + observation_error(data_test.y)
    xi_test     = data_test.xi[:, :model_params.num_test] 
    y_test      = data_test.y [   :model_params.num_test] 
    y0_test     = uqra.metrics.mquantiles(y_test, 1-model_params.pf)
    print(' Test Data set:')
    print('     X: shape={}, mean={}, std={}, [min, max]=({}, {})'.format(
                data_test.x.shape, np.mean(data_test.x, axis=1), np.std(data_test.x, axis=1),
                np.amin(data_test.x, axis=1),np.amax(data_test.x, axis=1)))

    print('     Xi: shape={}, mean={}, std={}, [min, max]=({}, {})'.format(
                xi_test.shape, np.mean(xi_test, axis=1), np.std(xi_test, axis=1),
                np.amin(xi_test, axis=1),np.amax(xi_test, axis=1)))

    print('     Y: shape={}, mean={}, std={}, [min, max]=({}, {})'.format(
                y_test.shape, np.mean(y_test), np.std(y_test),
                np.amin(y_test),np.amax(y_test)))

    np.random.seed(10)
    orth_poly = uqra.Hermite(d=2,deg=model_params.degs, hem_type='probabilists')
    beta = np.zeros((orth_poly.num_basis))
    for i in [0, 1,4,7,12,18,25]:
        beta[i] = stats.norm.rvs(0,1)

    np.random.seed(100)
    random.seed(100)
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
    filename = '{:s}_Adap{:d}{:s}{:d}_{:s}E5R{:d}_stability'.format(solver.nickname, 
            solver.ndim, model_params.basis, model_params.degs, doe_params.doe_nickname(), r)
    # ## ============ Saving QoIs ============
    try:
        np.save(os.path.join(data_dir_result, filename), res, allow_pickle=True)
        print(' >> Simulation Done! Data saved to {:s}'.format(os.path.join(data_dir_result, filename)))
    except:
        np.save(filename, res, allow_pickle=True)
        print(' >> Simulation Done! Data saved to {:s}'.format(os.path.join(os.getcwd(), filename)))
