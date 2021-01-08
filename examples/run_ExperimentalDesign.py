#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import numpy as np
import uqra
import os
import itertools
from tqdm import tqdm
import scipy.stats as stats
import math

def test_OptD():
    """
    Test D-Optimality 
    """
    np.random.seed(100)

    ndim= 1
    deg = int(30)
    n   = int(1e5)
    print(' Asymptotic distribution (ndim,deg, n)=({:d},{:d}, {:d}) check against Chevyshev distribution'.format(ndim, deg, n))
    fname_cand = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/OED/DoE_McsE6R0_uniform.npy' 
    candidate_samples = np.arange(n)
    x = np.load(fname_cand)[:ndim, candidate_samples]
    poly = uqra.Legendre(ndim, deg)
    X = poly.vandermonde(x)
    doe = uqra.OptimalDesign(X)
    SOptimal_samples0 = doe.samples('S', poly.num_basis * 2, initialization='AFP')
    doe = uqra.OptimalDesign(X)
    SOptimal_samples1 = doe.samples('S', poly.num_basis * 2, initialization='TSM')
    data= np.array([x[:,SOptimal_samples0], x[:, SOptimal_samples1]])
    np.save('test', data)

    # data = Data()
    # data.ndim = ndim
    # data.deg  = deg
    # data.candidate_data = fname_cand
    # data.candidate_samples = np.array(candidate_samples)
    # data.OptS_QR = np.array(SOptimal_samples0)
    # data.OptS_TSM= np.array(SOptimal_samples1)
    # with open(os.path.join(data_dir, out_fname), "wb") as output_file:
        # pickle.dump(data, output_file)

    # # data_dir = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/MCS/Uniform'
    # data_dir = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/MCS/Normal'
    # filename = 'DoE_McsE6R0.npy'
    # data_set = np.load(os.path.join(data_dir, filename))

    # np.random.seed(100)
    # ndim= 2
    # p   = np.array([2])

    # curr_set = []
    # for p in np.array([10]):
        # orth_poly = uqra.Hermite(d=ndim,deg=p)
        # n_cand    = int(1e5)
        # u_samples = data_set[0:ndim, :n_cand]
        # design_matrix = orth_poly.vandermonde(u_samples)
        # # n_budget  = 10 * design_matrix.shape[1]
        # n_budget  = 2048 
        # # n_budget  =  int(np.exp2(math.ceil(np.log2(design_matrix.shape[1]))))
        
        # start    = time.time()
        # doe      = uqra.OptimalDesign('D', curr_set=curr_set)
        # doe_index= doe.samples(design_matrix, n_samples=n_budget, orth_basis=True)
        # print(doe_index)
        # done     = time.time()
        # print('   >> OED-{:s} (n={:d}) time elapsed: {}'.format('S', n_cand, done - start))
        # # np.save('DoE_McsE6R0_d2_p{:d}_D.npy'.format(p), doe_index)

def run_CLS():

    # d = 2
    # print('Testing: Random Sampling from Pluripotential Equilibrium ...')
    # print('testing: d={:d}, theta: default'.format(d))
    # for i in range(1,6):
        # try:
            # doe = uqra.RandomDesign([sp.stats.uniform,]*d, 'CLS{:d}'.format(i))
            # doe_x = doe.get_samples(n_samples=1e5)
            # print('CLS{:d}'.format(i))
            # print('x shape: {}'.format(doe_x.shape))
            # print('x mean: {}'.format(np.mean(doe_x, axis=1)))
            # print('x std : {}'.format(np.std(doe_x, axis=1)))
            # print('x min : {}'.format(np.min(doe_x, axis=1)))
            # print('x max : {}'.format(np.max(doe_x, axis=1)))
        # except NotImplementedError:
            # pass
    data_dir = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/ExperimentalDesign'
    ndim = 3
    doe_method = 'CLS4'
    print('{:s}, d={:d}'.format(doe_method, ndim))
    doe = uqra.CLS(doe_method,ndim)
    for r in range(1):
        np.random.seed(None)
        doe_x = doe.samples(size=1e6)
        print('   - {:<25s} : {}'.format(' Dataset (U)', doe_x.shape))
        print('   - {:<25s} : {}, {}'.format(' U [min(U), max(U)]', np.amin(doe_x, axis=1), np.amax(doe_x, axis=1)))
        filename = 'DoE_{:s}E6D{:d}R{:d}.npy'.format(doe_method.capitalize(), ndim, r)
        np.save(os.path.join(data_dir, filename), doe_x)
    # print('Testing: Random Sampling from Pluripotential Equilibrium ...')
    # print('testing: d=2, theta: default')
    # doe = uqra.RandomDesign( [sp.stats.uniform,] * 2, 'CLS')
    # doe_x = doe.samples(n_samples=1e5)
    # print(doe_x.shape)
    # print(np.mean(doe_x, axis=1))
    # print(np.std(doe_x, axis=1))
    # print(np.min(doe_x, axis=1))
    # print(np.max(doe_x, axis=1))

    # print('Testing: Random Sampling from Pluripotential Equilibrium ...')
    # ndim = 1
    # print('testing: d={:d}, theta: default'.format(ndim))
    # doe = uqra.RandomDesign( [sp.stats.norm,] *ndim, 'CLS')
    # doe_x = doe.get_samples(n_samples=1e5)
    # print(doe_x.shape)
    # print(np.mean(doe_x, axis=1))
    # print(np.std(doe_x, axis=1))
    # print(np.min(doe_x, axis=1))
    # print(np.max(doe_x, axis=1))
    # # np.save('cls_norm_d2', doe_x)


    # print('\nTesting: Random Sampling from Pluripotential Equilibrium ...')
    # ndim = 3
    # print('testing: d={:d}, theta: default'.format(ndim))
    # doe = uqra.RandomDesign( [sp.stats.norm,] *ndim, 'CLS')
    # doe_x = doe.get_samples(n_samples=1e7)
    # print(doe_x.shape)
    # print(np.mean(doe_x, axis=1))
    # print(np.std(doe_x, axis=1))
    # print(np.min(doe_x, axis=1))
    # print(np.max(doe_x, axis=1))
    # # np.save('cls_norm_d2', doe_x)

    # data = np.load('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/CLS/Norm/DoE_ClsE6d3R0.npy')
    # print(data.shape)
    # print(np.mean(data, axis=1))
    # print(np.std(data, axis=1))
    # print(np.min(data, axis=1))
    # print(np.max(data, axis=1))


def test_Soptimality(slef):
    
    np.random.seed(100)
    ndim= 1
    deg = int(30)
    n   = int(1e5)
    print(' Asymptotic distribution (ndim,deg, n)=({:d},{:d}, {:d}) check against Chevyshev distribution'.format(ndim, deg, n))
    data_dir   = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/OED'
    fname_cand = 'DoE_McsE6R0_uniform.npy' 
    data_mcs   = np.load(os.path.join(data_dir, fname_cand))[:ndim, :]
    candidate_samples= np.arange(data_mcs.shape[1])
    random.seed(100)
    # random.shuffle(candidate_samples)
    candidate_samples= candidate_samples[:n]
    print(candidate_samples[:5])
    print(max(candidate_samples))
    print(len(set(candidate_samples)))
    x   = data_mcs[:ndim, candidate_samples]
    poly= uqra.Legendre(ndim, deg)
    X   = poly.vandermonde(x)
    print(X[:3,:3])
    doe = uqra.OptimalDesign(X, optimal_samples=[])
    SOptimal_samples0 = doe.get_samples('D', poly.num_basis, algorithm='RRQR')
    doe = uqra.OptimalDesign(X, optimal_samples=[])
    SOptimal_samples1 = doe.get_samples('S', poly.num_basis, algorithm='TSM')
    # print(idx)


    np.save(os.path.join(data_dir, 'test_S0.npy'), data_mcs[:,SOptimal_samples0])
    np.save(os.path.join(data_dir, 'test_S1.npy'), data_mcs[:,SOptimal_samples1])

    # data = Data()
    # data.ndim = ndim
    # data.deg  = deg
    # data.candidate_data = fname_cand
    # data.candidate_samples = np.array(candidate_samples)
    # data.OptS_QR = np.array(SOptimal_samples0)
    # data.OptS_TSM= np.array(SOptimal_samples1)
    # with open(os.path.join(data_dir, 'test.pkl'), "wb") as output_file:
        # pickle.dump(data, output_file)

    # x = np.linspace(-1,1,1000)
    # y = cdf_chebyshev(x)
    # ndim    = 2
    # n_cand  = int(1e7)

    # data_dir= r'/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/MCS/Uniform'
    # filename= r'DoE_McsE6R0.npy'
    # mcs_data_set  = np.load(os.path.join(data_dir, filename))
    # x_cand  = mcs_data_set[:ndim,:n_cand].reshape(ndim, -1)

    # for i, p in enumerate([5, ]):
        # mean_kappa = []
        # for _ in range(1):
            # np.random.seed(100)
            # orth_poly = uqra.Legendre(d=ndim,deg=p)
            # # orth_poly = uqra.Hermite(d=ndim,deg=p, hem_type='physicists')
            # doe     = uqra.OptimalDesign('S', selected_index=[3284,])
            # X       = orth_poly.vandermonde(x_cand)
            # idx     = doe.get_samples(X, n=math.ceil(1.2 * orth_poly.num_basis), orth_basis=True)
            # print('adding:')
            # print(idx)
            # print('current:')
            # print(doe.selected_index)
            # x_samples = x_cand[:,idx]
            # X_train = orth_poly.vandermonde(x_samples)
            # _, s, _ = np.linalg.svd(X_train)
            # ## condition number, kappa = max(svd)/min(svd)
            # kappa = max(abs(s)) / min(abs(s)) 
            # mean_kappa.append(kappa)
            # print('mean condition number: {}'.format(np.mean(mean_kappa)))

def test_gauss_quadrature():
    """
    https://keisan.casio.com/exec/system/1329114617
    """

    print('========================TESTING: 1D GAUSS QUADRATURE=======================')
    dists2test = [cp.Uniform(-1,1), cp.Normal(), cp.Gamma(1,1), cp.Beta(1,1)]
    rules2test = ['leg', 'hem', 'lag', 'jacobi']
    order2test = [2,3,4,5,6,7,8]
    for idist2test, irule2test in zip(dists2test, rules2test):
        print('-'*50)
        print('>>> Gauss Quadrature with polynominal: {}'.format(const.DOE_RULE_FULL_NAMES[irule2test.lower()]))
        uqra.blockPrint()
        quad_doe = uqra.DoE('QUAD', irule2test, order2test, idist2test)
        uqra_samples = quad_doe.get_samples()
        # quad_doe.disp()
        uqra.enablePrint()
        if irule2test == 'hem':
            for i, iorder in enumerate(order2test):
                print('>>> order : {}'.format(iorder))
                coord1d_e, weight1d_e = np.polynomial.hermite_e.hermegauss(iorder)
                print('{:<15s}: {}'.format('probabilist', np.around(coord1d_e,2)))
                coord1d, weight1d = np.polynomial.hermite.hermgauss(iorder)
                print('{:<15s}: {}'.format('physicist', np.around(coord1d,2)))
                print('{:<15s}: {}'.format('uqra', np.around(np.squeeze(uqra_samples[i][:-1,:]),2)))

        elif irule2test == 'leg':
            for i, iorder in enumerate(order2test):
                print('>>> order : {}'.format(iorder))
                coord1d, weight1d = np.polynomial.legendre.leggauss(iorder)
                print('{:<15s}: {}'.format('numpy ', np.around(coord1d,2)))
                print('{:<15s}: {}'.format('uqra', np.around(np.squeeze(uqra_samples[i][:-1,:]),2)))
        elif irule2test == 'lag':
            for i, iorder in enumerate(order2test):
                print('>>> order : {}'.format(iorder))
                coord1d, weight1d = np.polynomial.laguerre.laggauss(iorder)
                print('{:<15s}: {}'.format('numpy ', np.around(coord1d,2)))
                print('{:<15s}: {}'.format('uqra', np.around(np.squeeze(uqra_samples[i][:-1,:]),2)))
        elif irule2test == 'jacobi':
            print('NOT TESTED YET')

def run_LHS():
    alpha1 = np.linspace(1, 2, 6)
    alpha2 = np.linspace(2, 5, 7)
    # alpha3 = np.linspace(4,10, 7)
    alpha  = np.unique(np.hstack((alpha1, alpha2)))
    print('Running UQRA Latin Hypercube...')
    data_dir = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/ExperimentalDesign/LHS'
    # ndim_deg = [(2,10), (2,15), (4,4), (10, 2), (15,2), (2,2), (2,6)] 
    ndim_deg = [(3,i) for i in np.arange(2,11)] 
    for ndim, deg in ndim_deg:
        poly = uqra.poly.orthogonal(ndim, deg, 'leg') ## just need total number of basis, either leg or hem works

        print(' LHS (Uniform): ndim={}, deg={}'.format(ndim, deg))
        ### LHS samples in uniform
        doe = uqra.LHS([stats.uniform(-1,2), ] *ndim)
        for ialpha in tqdm(alpha, ascii=True, ncols=80):
            n = math.ceil(ialpha * poly.num_basis)
            filename = 'DoE_Lhs{:d}_{:d}uniform.npy'.format(n, ndim)
            if  os.path.isfile(os.path.join(data_dir, filename)):
                tqdm.write(filename+ ' exist')
                pass
            else:
                tqdm.write('running '+ filename)
                np.random.seed(None)
                doe_x = np.array([doe.samples(size=n) for _ in range(50)])
                np.save(os.path.join(data_dir, filename), doe_x)

        print(' LHS (Norm): ndim={}, deg={}'.format(ndim, deg))
        ### LHS samples in uniform
        doe = uqra.LHS([stats.norm(0,1), ] *ndim)
        for ialpha in tqdm(alpha, ascii=True, ncols=80):
            n = math.ceil(ialpha * poly.num_basis)
            filename = 'DoE_Lhs{:d}_{:d}norm.npy'.format(n, ndim)
            if  os.path.isfile(os.path.join(data_dir, filename)):
                pass
            else:
                np.random.seed(None)
                doe_x = np.array([doe.samples(size=n) for _ in range(50)])
                np.save(os.path.join(data_dir, filename), doe_x)

def run_UniformBall():

    data_dir = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/ExperimentalDesign/UniformBall'
    ndim = 3
    print(' UQRA: drawing samples uniformly from {:d}-dimensional ball...'.format(ndim))
    doe = uqra.CLS(doe_method,ndim)
    for r in range(10):
        np.random.seed(None)
        x = uqra.experiment.utilities.Uniform_ball(ndim, 1e6,r=1)
        doe_x = doe.samples(size=1e6)
        print('   - {:<25s} : {}'.format(' Dataset (U)', doe_x.shape))
        print('   - {:<25s} : {}, {}'.format(' U [min(U), max(U)]', np.amin(doe_x, axis=1), np.amax(doe_x, axis=1)))
        filename = 'DoE_{:s}E6D{:d}R{:d}.npy'.format(doe_method.capitalize(), ndim, r)
        np.save(os.path.join(data_dir, filename), doe_x)
    
def main():
    run_CLS()
    # run_LHS()


if __name__ == '__main__':
    main()
