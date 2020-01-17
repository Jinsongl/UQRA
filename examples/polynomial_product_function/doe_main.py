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
def cal_design_matrix(basis, x):
    return basis(*x)

def main():
    ## ------------------------ Parameters set-up ----------------------- ###
    ndim        = 5
    n_samples   = 1E6
    dist_x      = cp.Iid(cp.Normal(),ndim) 
    dist_zeta   = cp.Iid(cp.Normal(),ndim) 
    simparams   = museuq.simParameters('polynomial_product_function', dist_zeta)
    solver      = museuq.polynomial_product_function()

    n_eval      = 2
    plim        = (2,15)
    n_budget    = 1000
    simparams.set_adaptive_parameters(n_budget=n_budget, plim=plim, r2_bound=0.9, q_bound=0.05)
    simparams.info()

    # ### 1. Random Design
    # for r in tqdm(range(10)):
        # doe = museuq.RandomDesign('MCS', n_samples=n_samples, ndim=ndim, dist_names= ['normal',]*ndim, dist_theta=[(0,1),]*ndim)
        # u_doe = doe.samples()
        # x_doe = u_doe 
        # y_doe = solver.run(x_doe)
        # data = np.concatenate((u_doe, x_doe, y_doe.reshape(1,-1)), axis=0)
        # filename = os.path.join(simparams.data_dir, doe.filename+'R{:d}_ndim{:d}'.format(r, ndim))
        # np.save(filename, data)

    ### 2. Latin HyperCube Design
    for r in tqdm(range(1)):
        doe = museuq.LHS(n_samples=38257,dist_names=['normal']*ndim,ndim=ndim,dist_theta=[(0,1)]*ndim)
        u_doe, x_doe = doe.samples() ## u in [0,1], x in N(0,1)
        u_doe = x_doe
        y_doe = solver.run(x_doe)
        data  = np.concatenate((u_doe, x_doe, y_doe.reshape(1,-1)), axis=0)
        filename = os.path.join(simparams.data_dir, doe.filename+'R{:d}_ndim{:d}'.format(r, ndim))
        np.save(filename, data)


    # ## 3. Quadrature Design
    # quad_orders = range(3,17)
    # for iquad_orders in tqdm(quad_orders):
        # doe = museuq.QuadratureDesign(iquad_orders, ndim = ndim, dist_names=['normal', ] *ndim)
        # u_doe, w_doe = doe.samples()
        # x_doe = u_doe 
        # y_doe = solver.run(x_doe)
        # data  = np.concatenate((u_doe, x_doe, w_doe.reshape(1,-1), y_doe.reshape(1,-1)), axis=0)
        # filename = os.path.join(simparams.data_dir, doe.filename+'_ndim{:d}'.format(ndim))
        # np.save(filename, data)


    ### 4. Optimal Design
    # data_dir = '/Users/jinsongliu/External/MUSE_UQ_DATA/Ishigami/Data'
    # # data_dir = 'E:\Run_MUSEUQ'
    # # data_dir = simparams.data_dir
    # np.random.seed(100)

    # filename  = 'DoE_McsE6R0.npy'
    # data_set  = np.load(os.path.join(data_dir, filename))
    # u_samples = data_set[0:ndim, :]
    # x_samples = data_set[ndim:2*ndim, :]
    # # y_samples = data_set[-1  , :].reshape(1,-1)
    # basis     = cp.orth_ttr(plim[1], dist_zeta)
    # print('Calculating Design matrix...')
    # start     = time.time()
    # # pool = mp.Pool(processes=mp.cpu_count())
    # # design_matrix = pool.starmap(cal_design_matrix,  [(basis, iu) for iu in u_samples.T])
    # # pool.close()
    # design_matrix = basis(*u_samples).T
    # done      = time.time()
    # print('Candidate samples filename: {:s}'.format(filename))
    # print('   >> Candidate sample set shape: {}'.format(u_samples.shape))
    # print('   >> Candidate Design matrix shape: {}'.format(design_matrix.shape))
    # print('   >> Candidate Design matrix time elapsed: {}'.format(done - start))
    # filename = os.path.join(data_dir, 'DoE_McsE6R0_Leg{:d}_design_matrix'.format(plim[1]))
    # np.save(filename, design_matrix)
    
    # # for ia in alpha:
    # #     print('   >> Oversampling rate : {:.2f}'.format(ia))
    # #     doe_size = min(int(len(basis)*ia), int(n_samples))
    # #     doe = museuq.OptimalDesign('S', n_samples = doe_size )
    # #     doe_index = doe.samples(design_matrix, is_orth=True)
    # #     data = np.concatenate((doe_index.reshape(1,-1),u_samples[:,doe_index],x_samples[:,doe_index], y_samples[:,doe_index]), axis=0)
    # #     filename = os.path.join(data_dir, 'DoE_McsE6R{:d}_q{:d}_OptS{:d}'.format(r,iquad_orders,doe_size))
    # #     np.save(filename, data)

    # start     = time.time()
    # doe = museuq.OptimalDesign('D', n_samples = n_eval)
    # doe_index = doe.samples(design_matrix, is_orth=True)
    # done      = time.time()
    # print('   >> OPT-D (n={:d}) time elapsed: {}'.format(n_eval, done - start))
    # start     = time.time()
    # y_samples = solver.run(x_samples[:, doe_index])
    # done      = time.time()
    # print('   >> Solver (n={:d}) time elapsed: {}'.format(n_eval, done - start))
    # data = np.concatenate((doe_index.reshape(1,-1),u_samples[:,doe_index],x_samples[:,doe_index], y_samples.reshape(1,-1)), axis=0)
    # filename = os.path.join(data_dir, 'DoE_McsE6R0_q{:d}_OptD{:d}'.format(plim[1],n_eval))
    # np.save(filename, data)
            


if __name__ == '__main__':
    main()
