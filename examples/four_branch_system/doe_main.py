#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import context
import museuq
import numpy as np, chaospy as cp, os, sys
import warnings
from tqdm import tqdm
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()

import time
import multiprocessing as mp

def main():
    ## ------------------------ Parameters set-up ----------------------- ###
    ndim        = 2
    dist_x      = cp.Iid(cp.Normal(),ndim) 
    dist_zeta   = cp.Iid(cp.Normal(),ndim) 
    simparams   = museuq.simParameters('four_branch_system', dist_zeta)
    solver      = museuq.four_branch_system()
    plim        = (2,10)
    porder_opt  = plim[1]
    n_budget    = 100
    simparams.set_adaptive_parameters(n_budget=n_budget, plim=plim, r2_bound=0.9, q_bound=0.05)
    simparams.info()

    ### 1. Random Design
    for r in tqdm(range(1)):
        doe = museuq.RandomDesign('MCS', n_samples=n_budget, ndim=ndim, dist_names= ['normal',]*ndim, dist_theta=[(0,1),]*ndim)
        u_doe = doe.samples()
        x_doe = u_doe 
        y_doe = solver.run(x_doe)
        data = np.concatenate((u_doe, x_doe, y_doe.reshape(1,-1)), axis=0)
        filename = os.path.join(simparams.data_dir, doe.filename+'R{:d}'.format(r))
        np.save(filename, data)

    ### 2. Latin HyperCube Design
    for r in tqdm(range(1)):
        doe = museuq.LHS(n_samples=n_budget,dist_names=['normal']*ndim,ndim=ndim,dist_theta=[(0,1)]*ndim)
        u_doe, x_doe = doe.samples() ## u in [0,1], x in N(0,1)
        u_doe = x_doe 
        y_doe = solver.run(x_doe)
        data = np.concatenate((u_doe, x_doe, y_doe.reshape(1,-1)), axis=0)
        filename = os.path.join(simparams.data_dir, doe.filename+'R{:d}'.format(r))
        np.save(filename, data)


    ## 3. Quadrature Design
    quad_orders = range(10,11)
    for iquad_orders in tqdm(quad_orders):
        doe = museuq.QuadratureDesign(iquad_orders, ndim = ndim, dist_names=['normal', ] *ndim)
        u_doe, w_doe = doe.samples()
        x_doe = u_doe 
        y_doe = solver.run(x_doe)
        data = np.concatenate((u_doe, x_doe, w_doe.reshape(1,-1), y_doe.reshape(1,-1)), axis=0)
        filename = os.path.join(simparams.data_dir, doe.filename)
        np.save(filename, data)


    ### 4. Optimal Design
    # data_dir = simparams.data_dir 
    # data_dir = 'E:\Run_MUSEUQ'

    # filename  = 'DoE_McsE6R0.npy'
    # data_set  = np.load(os.path.join(data_dir, filename))
    # u_samples = data_set[0:ndim, :]
    # x_samples = data_set[ndim:2*ndim, :]
    # y_samples = data_set[-1  , :].reshape(1,-1)
    # basis     = cp.orth_ttr(porder_opt, dist_zeta)
    # print('Calculating Design matrix...')
    # start     = time.time()
    # design_matrix = basis(*u_samples).T
    # done      = time.time()
    # print('Candidate samples filename: {:s}'.format(filename))
    # print('   >> Candidate sample set shape: {}'.format(u_samples.shape))
    # print('   >> Candidate Design matrix shape: {}'.format(design_matrix.shape))
    # print('   >> Candidate Design matrix time elapsed: {}'.format(done - start))
    

    # start     = time.time()
    # doe = museuq.OptimalDesign('S', n_samples = n_budget)
    # doe_index = doe.samples(design_matrix, is_orth=True)
    # done      = time.time()
    # print('   >> OPT-S (n={:d}) time elapsed: {}'.format(n_budget, done - start))
    # start     = time.time()
    # y_samples = solver.run(x_samples[:, doe_index])
    # done      = time.time()
    # print('   >> Solver (n={:d}) time elapsed: {}'.format(n_budget, done - start))
    # data = np.concatenate((u_samples[:,doe_index],x_samples[:,doe_index], doe_index.reshape(1,-1), y_samples.reshape(1,-1)), axis=0)
    # filename = os.path.join(data_dir, 'DoE_McsE6R0_q{:d}_OptS{:d}'.format(porder_opt,n_budget))
    # np.save(filename, data)
            
    # start     = time.time()
    # doe = museuq.OptimalDesign('D', n_samples = n_budget)
    # doe_index = doe.samples(design_matrix, is_orth=True)
    # done      = time.time()
    # print('   >> OPT-D (n={:d}) time elapsed: {}'.format(n_budget, done - start))
    # start     = time.time()
    # y_samples = solver.run(x_samples[:, doe_index])
    # done      = time.time()
    # print('   >> Solver (n={:d}) time elapsed: {}'.format(n_budget, done - start))
    # data = np.concatenate((u_samples[:,doe_index],x_samples[:,doe_index], doe_index.reshape(1,-1), y_samples.reshape(1,-1)), axis=0)
    # filename = os.path.join(data_dir, 'DoE_McsE6R0_q{:d}_OptD{:d}'.format(porder_opt,n_budget))
    # np.save(filename, data)
            


if __name__ == '__main__':
    main()
