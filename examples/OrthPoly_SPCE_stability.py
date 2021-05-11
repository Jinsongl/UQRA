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
import multiprocessing as mp
import random
# warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()
class Data():
    pass
def observation_error(y, mu=0, cov=0.03, random_state=100):
    e = stats.norm(0, cov * abs(y)).rvs(size=len(y), random_state=random_state)
    return e
def main():

    ## ------------------------ Displaying set up ------------------- ###
    r, theta= 0, 0
    ith_batch  = 0
    batch_size = 1
    np.set_printoptions(precision=4)
    np.set_printoptions(threshold=1000)
    np.set_printoptions(suppress=True)
    ## ------------------------ Define solver ----------------------- ###

    np.random.seed(10)
    orth_poly = uqra.Hermite(d=2,deg=6, hem_type='probabilists')
    coef = np.zeros((orth_poly.num_basis))
    for i in [0,1,4,7,12,18,25]:
        coef[i] = stats.norm.rvs(0,1)
    solver = uqra.OrthPoly(orth_poly, coef=coef)
    print(solver.coef, len(solver.coef))

    beta = coef
    np.random.seed(10)
    orth_poly = uqra.Hermite(d=2,deg=10, hem_type='probabilists')
    beta = np.zeros((orth_poly.num_basis))
    for i in [0,1,4,7,12,18,25]:
        beta[i] = stats.norm.rvs(0,1)
    solver1 = uqra.OrthPoly(orth_poly, coef=beta)
    print(solver1.coef, len(solver1.coef))
    # x = stats.norm(0,1).rvs(size=(2,1000))
    # y0 = solver0.run(x)
    # y1 = solver1.run(x)
    # print(np.amin(y0-y1))
    # print(np.array_equal(y0,y1))

    np.random.seed(100)
    random.seed(100)
    ## ------------------------ UQRA Modeling Parameters ----------------- ###
    model_params = uqra.Modeling('PCE')
    model_params.degs    = 10 #np.arange(10,11) #[2,6,10]#
    model_params.ndim    = solver.ndim
    model_params.basis   = 'Heme'
    model_params.dist_u  = stats.uniform(0,1)  #### random CDF values for samples
    model_params.fitting = 'OLSLAR' 
    model_params.alpha   = 3
    model_params.num_test= int(1e6)
    model_params.num_pred= int(1e6)
    model_params.n_jobs  = mp.cpu_count()
    model_params.update_basis()
    model_params.info()

    ## ------------------------ UQRA DOE Parameters ----------------- ###
    doe_params = uqra.ExperimentParameters('MCS', 'D')
    doe_params.update_poly_name(model_params.basis)
    doe_params.num_cand = int(1e5)
    doe_params.ndim     = int(model_params.ndim)
    doe_params.deg      = int(model_params.degs)
    ### Specify candidate data filename template function
    ### e.g.  filename_template= lambda r: r'DoE_Ball5pt6E5R{:d}.npy'.format(r)
    ### if not specified, default values will be used
    doe_params.update_filenames(filename_template=None)
    filename_cand = doe_params.fname_cand(r)
    # filename_design = doe_params.fname_design(r)

    ### data object containing results from intermedia steps

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
    filename_optimal = 'DoE_{:s}E5R0_{:d}{:s}{:d}.npy'.format(doe_params.doe_sampling.capitalize(), model_params.ndim, 
            model_params.basis[:3], model_params.degs)

    data_test   = np.load(os.path.join(data_dir_test, filename_test), allow_pickle=True).tolist()
    data_test.x = solver.map_domain(data_test.u, model_params.dist_u)
    data_test.xi= model_params.map_domain(data_test.u, model_params.dist_u)
    data_test.y = solver.run(data_test.x) if not hasattr(data_test, 'y') else data_test.y
    xi_test     = data_test.xi[:, :model_params.num_test] 
    y_test      = data_test.y [   :model_params.num_test] 
    # y0_test     = uqra.metrics.mquantiles(y_test, 1-model_params.pf)

    print('\n#################################################################################')
    print(' >>>  File: ', __file__)
    # print(' >>>  Start UQRA : {:d}[{:d}]/{:d} x {:d}'.format(i, irepeat, batch_size, ith_batch))
    print(' >>>  Test data R={:d}'.format(r))
    print('#################################################################################\n')
    print('   > {:<25s}'.format('Input/Output Directories:'                     ))
    print('     - {:<23s} : {}'.format  (' Candiate samples'  , data_dir_cand   ))
    print('     - {:<23s} : {:s}'.format(' UQRA DoE data '    , data_dir_optimal))
    print('     - {:<23s} : {:s}'.format(' Test input '       , data_dir_testin ))
    print('     - {:<23s} : {:s}'.format(' Test output'       , data_dir_test   ))
    print('     - {:<23s} : {:s}'.format(' UQRA output data ' , data_dir_result ))
    print('     - {:<23s} : {:s}'.format(' UQRA output figure', figure_dir      ))
    print('   > {:<25s}'.format('Input/Output files'                            ))
    print('     - {:<23s} : {}'.format(' Test input data'     , filename_testin ))
    print('     - {:<23s} : {}'.format(' Test output data'    , filename_test   ))
    print('     - {:<23s} : {}'.format(' Candidate filename'  , filename_cand   ))
    print('     - {:<23s} : {}'.format(' Optimal   filename'  , filename_optimal))
    print('     - {:<23s} : {}'.format(' UQRA DoE '  , doe_params.doe_nickname()))
    print('-'*80)




    data_cand = np.load(os.path.join(data_dir_cand, filename_cand))
    if doe_params.doe_sampling.lower().startswith('cls'):
        data_cand = data_cand * model_params.degs**0.5
    data_optimal = np.load(os.path.join(data_dir_optimal, filename_optimal), allow_pickle=True)
    print(data_cand.shape)
    print(np.mean(data_cand, axis=1))
    print(np.std(data_cand , axis=1))
    print(np.amin(data_cand, axis=1))
    print(np.amax(data_cand, axis=1))

    res = []
    for n_samples in np.unique(np.concatenate([np.arange(10, 66*2, 5), np.arange(66*2, 66*10, 66)], axis=0)):
    # for n_samples in np.unique(np.concatenate([np.arange(10, 6*2, 5), np.arange(66*2, 66*2, 66)], axis=0)):
        ## ------------------------ UQRA Surrogate model----------------- ###
        orth_poly = uqra.poly.orthogonal(solver.ndim, model_params.degs, model_params.basis)
        pce_model = uqra.PCE(orth_poly)
        dist_u    = model_params.dist_u 
        dist_xi   = orth_poly.weight
        dist_x    = solver.distributions
        pce_model.info()

        data    = uqra.Data()
        data.n  = n_samples
        data.xi = []
        data.x  = []
        data.y  = []
        data.model  = []
        data.beta = beta
        data.beta_hat = [] 
        error = []
        for i in tqdm(range(50)):
            uqra.blockPrint()
            optimal_idx = getattr(data_optimal[i], doe_params.doe_nickname())
            xi = data_cand[:solver.ndim, optimal_idx[:n_samples]]
            x  = solver.map_domain(xi, dist_xi)
            y  = solver.run(x)
            data.xi.append(xi)
            data.x.append(x)
            data.y.append(y)
            # data.y  = data.y + observation_error(data.y)
            # print('   2. Training with {} '.format(model_params.fitting))
            weight  = doe_params.sampling_weight()   ## weight function
            pce_model.fit(model_params.fitting, xi, y, w=weight, 
                    n_jobs=model_params.n_jobs)#, n_splits=model_params.n_splits) #
            sparsity = len(pce_model.active_index)
            # print('     - {:<32s} : ({},{}),    Alpha: {:.2f}'.format('X train', data.x.shape[1], pce_model.num_basis, 
                            # data.x.shape[1]/pce_model.num_basis))
            # print('     - {:<32s} : {}'.format('Y train'    , data.y.shape))

            data.model.append(pce_model)
            beta_hat = np.zeros((orth_poly.num_basis))
            for i, beta_i in zip(pce_model.active_index, pce_model.coef):
                beta_hat[i] = beta_i
            data.beta_hat.append(beta_hat)
            print('     - {:<32s} : {}'.format('Sparsity'   , len(pce_model.active_index)))
            print(' {:.2f}'.format(np.linalg.norm(beta-beta_hat)/np.linalg.norm(beta))  )
            # print(beta)
            # print(beta_hat)
            error.append(np.linalg.norm(beta-beta_hat)/np.linalg.norm(beta))
            uqra.enablePrint()

        res.append(data)
        error = np.array(error)
        print(n_samples, len(error), np.sum(error<0.1), np.sum(error<0.1)/len(error))
    filename = '{:s}_{:d}{:s}{:d}_{:s}.npy'.format('OrthPoly', solver.ndim, model_params.basis[:3], model_params.degs, doe_params.doe_nickname())
    print(os.path.join(data_dir_test, filename))
    np.save(os.path.join(data_dir_test, filename), res, allow_pickle=True)

 

if __name__ == '__main__':
    main()
