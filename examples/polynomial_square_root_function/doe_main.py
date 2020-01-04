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
    ndim        = 2
    n_samples   = 1E6
    model_name  = 'polynomial_square_root_function'
    ## 1. Choose Wiener-Askey scheme random variable
    dist_x      = cp.Iid(cp.Normal(),ndim) 
    dist_zeta   = cp.Iid(cp.Normal(),ndim) 
    ## 2. If transformation needed, like Rosenblatt, need to be done here
    ## Perform Rosenblatt etc
    simparams = museuq.simParameters(model_name, dist_zeta)
    # simparams.set_error()  # no observation error for sdof
    simparams.info()

    solver = museuq.polynomial_square_root_function()

    # ### 1. Random Design
    # for r in tqdm(range(10)):
        # doe = museuq.RandomDesign('MCS', n_samples=n_samples, ndim=ndim, dist_names= ['normal',]*ndim, dist_theta=[(0,1),]*ndim)
        # doe.samples()
        # doe.x = doe.u
        # solver.run(doe.x)
        # data = np.concatenate((doe.u, doe.x, solver.y.reshape(1,-1)), axis=0)
        # filename = os.path.join(simparams.data_dir, doe.filename+'R{:d}'.format(r))
        # np.save(filename, data)

    # ### 2. Latin HyperCube Design
    # for r in tqdm(range(10)):
        # doe = museuq.LHS(n_samples=1e3,dist_names=['normal']*ndim,ndim=ndim,dist_theta=[(0,1)]*ndim)
        # doe.samples()
        # data = np.concatenate((doe.u, doe.x), axis=0)
        # filename = os.path.join(simparams.data_dir, doe.filename+'R{:d}'.format(r))
        # np.save(filename, data)


    # ### 3. Quadrature Design
    # quad_orders = range(1,10)
    # for iquad_orders in tqdm(quad_orders):
    #     doe = museuq.QuadratureDesign(iquad_orders, ndim = ndim, dist_names=['normal', ] *ndim)
    #     doe.samples()
    #     doe.x = doe.u 
    #     print(doe.x.shape)
    #     solver.run(doe.x)
    #     data = np.concatenate((doe.u, doe.x, doe.w.reshape(1,-1), solver.y.reshape(1,-1)), axis=0)
    #     filename = os.path.join(simparams.data_dir, doe.filename)
    #     np.save(filename, data)


    ### 4. Optimal Design
    # data_dir = '/Users/jinsongliu/External/MUSE_UQ_DATA/linear_oscillator/Data'
    # data_dir = 'E:\Run_MUSEUQ'
    data_dir ='E:\polynomial_square_root\'
    np.random.seed(100)

    quad_orders = range(1,10)
    alpha       = [1.0, 1.1, 1.3, 1.5, 2.0,2.5, 3.0,3.5, 5]
    for iquad_orders in tqdm(quad_orders):
        basis = cp.orth_ttr(iquad_orders-1,dist_zeta)
        for r in range(4):
            
            filename  = 'DoE_McsE6R{:d}.npy'.format(r)
            data_set  = np.load(os.path.join(data_dir, filename))
            samples_u = data_set[0:2, :]
            samples_x = data_set[2:4, :]
            samples_y = data_set[5  , :].reshape(1,-1)
            print('Quadrature Order: {:d}'.format(iquad_orders))
            print('Candidate samples filename: {:s}'.format(filename))
            print('   >> Candidate sample set shape: {}'.format(samples_u.shape))
            design_matrix = basis(*samples_u).T
            print('   >> Candidate Design matrix shape: {}'.format(design_matrix.shape))
            
            # for ia in alpha:
            #     print('   >> Oversampling rate : {:.2f}'.format(ia))
            #     doe_size = min(int(len(basis)*ia), int(n_samples))
            #     doe = museuq.OptimalDesign('S', n_samples = doe_size )
            #     doe.samples(design_matrix, u=samples_u, is_orth=True)
            #     data = np.concatenate((doe.I.reshape(1,-1),doe.u,samples_x[:,doe.I], samples_y[:,doe.I]), axis=0)
            #     filename = os.path.join(data_dir, 'DoE_McsE6R{:d}_q{:d}_OptS{:d}'.format(r,iquad_orders,doe_size))
            #     np.save(filename, data)

            for ia in alpha:
                print('   >> Oversampling rate : {:.2f}'.format(ia))
                doe_size = min(int(len(basis)*ia), int(n_samples))
                doe = museuq.OptimalDesign('D', n_samples = doe_size )
                doe.samples(design_matrix, u=samples_u, is_orth=True)
                data = np.concatenate((doe.I.reshape(1,-1),doe.u,samples_x[:,doe.I], samples_y[:,doe.I]), axis=0)
                filename = os.path.join(data_dir, 'DoE_McsE6R{:d}_q{:d}_OptD{:d}'.format(r,iquad_orders,doe_size))
                np.save(filename, data)
                


if __name__ == '__main__':
    main()
