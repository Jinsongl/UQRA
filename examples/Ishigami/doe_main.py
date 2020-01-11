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




def main():
    ## ------------------------ Parameters set-up ----------------------- ###
    ndim        = 3
    n_samples   = 1E6
    model_name  = 'Ishigami'
    ## 1. Choose Wiener-Askey scheme random variable
    dist_x      = cp.Iid(cp.Uniform(-np.pi, np.pi),ndim) 
    dist_zeta   = cp.Iid(cp.Uniform(-1, 1),ndim) 
    ## 2. If transformation needed, like Rosenblatt, need to be done here
    ## Perform Rosenblatt etc
    simparams = museuq.simParameters(model_name, dist_zeta)
    # simparams.set_error()  # no observation error for sdof
    simparams.info()

    solver = museuq.Ishigami()

    # ### 1. Random Design
    # for r in tqdm(range(10)):
        # doe = museuq.RandomDesign('MCS', n_samples=n_samples, ndim=ndim, dist_names= ['uniform',]*ndim, dist_theta=[(-1,2),]*ndim)
        # doe.samples()
        # doe.x = np.pi*(doe.u+1) - np.pi
        # solver.run(doe.x)
        # data = np.concatenate((doe.u, doe.x, solver.y.reshape(1,-1)), axis=0)
        # filename = os.path.join(simparams.data_dir, doe.filename+'R{:d}'.format(r))
        # np.save(filename, data)

    # ### 2. Latin HyperCube Design
    # for r in tqdm(range(1)):
        # doe = museuq.LHS(n_samples=1000,dist_names=['uniform']*ndim,ndim=ndim,dist_theta=[(-1,2)]*ndim)
        # doe.samples()
        # doe.x = -np.pi + 2*np.pi * doe.u
        # doe.u = -1 + 2 * doe.u
        # solver.run(doe.x)
        # data = np.concatenate((doe.u, doe.x, solver.y.reshape(1,-1)), axis=0)
        # filename = os.path.join(simparams.data_dir, doe.filename+'R{:d}'.format(r))
        # np.save(filename, data)


    ### 3. Quadrature Design
    quad_orders = range(3,17)
    for iquad_orders in tqdm(quad_orders):
        doe = museuq.QuadratureDesign(iquad_orders, ndim = ndim, dist_names=['uniform', ] *ndim)
        doe.samples()
        doe.x = np.pi*(doe.u+1) - np.pi
        solver.run(doe.x)
        data = np.concatenate((doe.u, doe.x, doe.w.reshape(1,-1), solver.y.reshape(1,-1)), axis=0)
        filename = os.path.join(simparams.data_dir, doe.filename)
        np.save(filename, data)


    # ### 4. Optimal Design
    # # data_dir = '/Users/jinsongliu/External/MUSE_UQ_DATA/linear_oscillator/Data'
    # # data_dir = 'E:\Run_MUSEUQ'
    # data_dir = simparams.data_dir
    # np.random.seed(100)
            

    # filename  = 'DoE_McsE6R0.npy'
    # data_set  = np.load(os.path.join(data_dir, filename))
    # samples_u = data_set[0:ndim, :]
    # samples_x = data_set[ndim:2*ndim, :]
    # samples_y = data_set[-1  , :].reshape(1,-1)
    # print('Candidate samples filename: {:s}'.format(filename))
    # print('   >> Candidate sample set shape: {}'.format(samples_u.shape))
    # design_matrix = basis(*samples_u).T
    # print('   >> Candidate Design matrix shape: {}'.format(design_matrix.shape))
    
    # # for ia in alpha:
    # #     print('   >> Oversampling rate : {:.2f}'.format(ia))
    # #     doe_size = min(int(len(basis)*ia), int(n_samples))
    # #     doe = museuq.OptimalDesign('S', n_samples = doe_size )
    # #     doe.samples(design_matrix, u=samples_u, is_orth=True)
    # #     data = np.concatenate((doe.I.reshape(1,-1),doe.u,samples_x[:,doe.I], samples_y[:,doe.I]), axis=0)
    # #     filename = os.path.join(data_dir, 'DoE_McsE6R{:d}_q{:d}_OptS{:d}'.format(r,iquad_orders,doe_size))
    # #     np.save(filename, data)

    # for ia in alpha:
        # print('   >> Oversampling rate : {:.2f}'.format(ia))
        # doe_size = min(int(len(basis)*ia), int(n_samples))
        # doe = museuq.OptimalDesign('D', n_samples = doe_size )
        # doe.samples(design_matrix, u=samples_u, is_orth=True)
        # data = np.concatenate((doe.I.reshape(1,-1),doe.u,samples_x[:,doe.I], samples_y[:,doe.I]), axis=0)
        # filename = os.path.join(data_dir, 'DoE_McsE6R{:d}_q{:d}_OptD{:d}'.format(r,iquad_orders,doe_size))
        # np.save(filename, data)
                


if __name__ == '__main__':
    main()
