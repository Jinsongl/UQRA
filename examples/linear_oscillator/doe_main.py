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
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel,WhiteKernel)
from tqdm import tqdm
from museuq.utilities import helpers as uqhelpers
from museuq.utilities import metrics_collections as museuq_metrics
from museuq.utilities import dataIO 
from museuq.environment import Kvitebjorn
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()




def main():
    ## ------------------------ Parameters set-up ----------------------- ###
    model_name  = 'linear_oscillator'
    ## 1. Choose Wiener-Askey scheme random variable
    dist_normal = cp.Normal()
    dist_zeta = cp.Iid(cp.Normal(),2) 

    ## 2. If transformation needed, like Rosenblatt, need to be done here
    ## Perform Rosenblatt etc
    simparams = museuq.simParameters(model_name, dist_zeta)
    # simparams.set_error()  # no observation error for sdof
    simparams.info()

    solver = museuq.Solver(model_name)
    ### ------------------------------ Run DOE ----------------------------- ###

    ### 1. Random Design
    for r in range(10):
        doe = museuq.RandomDesign('MCS', n_samples=1e6, ndim=2, dist_names= ['normal',]*2, dist_theta=[(0,1),]*2)
        doe.samples()
        doe.x = doe.u
        solver.run(doe.x)
        doe_data = np.concatenate((doe.u, doe.x, solver.y.reshape(1,-1)), axis=0)
        filename = os.path.join(simparams.data_dir, doe.filename+'R{:d}'.format(r))
        np.save(filename, doe_data)

    # ### 2. Latin HyperCube Design
    # for r in range(10):
        # doe = museuq.LHS(n_samples=1e3,dist_names=['normal']*2,ndim=2,dist_theta=[(0,1)]*2)
        # doe.samples()
        # doe_data = np.concatenate((doe.u, doe.x), axis=0)
        # filename = os.path.join(simparams.data_dir, doe.filename+'R{:d}'.format(r))
        # np.save(filename, doe_data)


    ### 3. Quadrature Design
    quad_orders = range(3,20)
    for iquad_orders in quad_orders:
        doe = museuq.QuadratureDesign(iquad_orders, ndim = 2, dist_names=['normal', 'normal'])
        doe.samples()
        print(' > {:<15s} : {}'.format('Experimental Design', doe))
        doe.x = doe.u 
        solver.run(doe.x)
        doe_data = np.concatenate((doe.u, doe.x, doe.w.reshape(1,-1), solver.y.reshape(1,-1)), axis=0)
        filename = os.path.join(simparams.data_dir, doe.filename)
        np.save(filename, doe_data)



    ### 4. Optimal Design
    # data_dir = '/Users/jinsongliu/External/MUSE_UQ_DATA/linear_oscillator/Data'
    # data_dir = 'E:\Run_MUSEUQ'
    data_dir = os.getcwd()
    np.random.seed(100)

    quad_orders = range(4,11)
    alpha = [1.0, 1.1, 1.3, 1.5, 2.0,2.5, 3.0,3.5, 5]
    for iquad_orders in quad_orders:
        poly_order = iquad_orders - 1
        basis = cp.orth_ttr(poly_order,dist_zeta)
        for r in range(10):

            filename  = 'DoE_McsE6R{:d}_stats.npy'.format(r)
            data_set  = np.load(os.path.join(data_dir, filename))
            samples_y = np.squeeze(data_set[:,4,:]).T
            
            filename  = 'DoE_McsE6R{:d}.npy'.format(r)
            data_set  = np.load(os.path.join(data_dir, filename))
            samples_u = data_set[0:2, :]
            samples_x = data_set[2:4, :]

            print('Quadrature Order: {:d}'.format(iquad_orders))
            print('Candidate samples filename: {:s}'.format(filename))
            print('   >> Candidate sample set shape: {}'.format(samples_u.shape))
            design_matrix = basis(*samples_u).T
            print('   >> Candidate Design matrix shape: {}'.format(design_matrix.shape))

            for ia in alpha:
                print('   >> Oversampling rate : {:.2f}'.format(ia))
                doe_size = min(int(len(basis)*ia), 10000)
                doe = museuq.OptimalDesign('S', n_samples = doe_size )
                doe.samples(design_matrix, u=samples_u, is_orth=True)
                data = np.concatenate((doe.I.reshape(1,-1),doe.u,samples_x[:,doe.I], samples_y[:,doe.I]), axis=0)
                filename = os.path.join(data_dir,'DoE_McsE6R{:d}_q{:d}_OptS{:d}'.format(r,iquad_orders,doe_size))
                np.save(filename, data)

            for ia in alpha:
                print('   >> Oversampling rate : {:.2f}'.format(ia))
                doe_size = min(int(len(basis)*ia), 10000)
                doe = museuq.OptimalDesign('D', n_samples = doe_size )
                doe.samples(design_matrix, u=samples_u, is_orth=True)
                data = np.concatenate((doe.I.reshape(1,-1),doe.u,samples_x[:,doe.I], samples_y[:,doe.I]), axis=0)
                filename = os.path.join(data_dir,'DoE_McsE6R{:d}_q{:d}_OptD{:d}'.format(r,iquad_orders,doe_size))
                np.save(filename, data)
                


if __name__ == '__main__':
    main()
