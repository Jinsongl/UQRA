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
# warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()

class Data():
    pass

def OED_data_dir():
    if platform.system() == 'Darwin':       ## Mac
        data_dir= r'/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/OED'
    elif platform.system() == 'Windows':    ## Windows
        data_dir= r'G:\My Drive\MUSE_UQ_DATA\Samples\OED' 
    elif platform.system() == 'Linux':      ## Ubuntu
        data_dir= r'/home/jinsong/Documents/MUSE_UQ_DATA/Samples/OED'
    else:
        raise ValueError
    return data_dir

def main():

    ## ------------------------ Displaying set up ------------------- ###
    print('\n#################################################################################')
    print(' >>>  Running UQRA '.format(__file__))
    print('#################################################################################\n')
    np.random.seed(100)
    np.set_printoptions(precision=4)
    np.set_printoptions(threshold=8)
    np.set_printoptions(suppress=True)
    np.random.seed(100)
    data_dir_oed = OED_data_dir()
    ## ------------------------ Define solver ----------------------- ###

    u_distname    = 'norm'
    doe_candidate = 'cls4'
    doe_optimality= 'D'

    u_distname    = 'uniform'
    doe_candidate = 'cls1'
    doe_optimality= 'D'
    ### >>> Define solver 
    if u_distname.lower() == 'uniform':
        print(' Legendre polynomial')
        orth_poly = museuq.Legendre(d=ndim,deg=p)
        solver    = museuq.SparsePoly(orth_poly, sparsity='full', seed=100)
        print(' Model: {}'.format(solver.name))

    elif u_distname.lower().startswith('norm'):
        if doe_candidate.startswith('cls') or doe_candidate == 'reference':
            print(' Physicist Hermite polynomial')
            orth_poly = museuq.Hermite(d=ndim,deg=p, hem_type='physicists')
        else:
            print(' Probabilists Hermite polynomial')
            orth_poly = museuq.Hermite(d=ndim,deg=p, hem_type='probabilists')

        solver    = museuq.SparsePoly(orth_poly, sparsity='full', seed=100)
        print(' Model: {}'.format(solver.name))

    else:
        raise NotImplementedError

    ## ------------------------ Simulation Parameters ----------------- ###
    solver = uqra.Solver('SparsePoly', ndim=2)
    simparams = uqra.Parameters(solver, doe_method=['CLS4', 'D'], fit_method='LASSOLARS')
    simparams.set_udist(u_distname)
    simparams.x_dist     = solver.distributions 
    simparams.pce_degs   = np.array(range(2,11))
    simparams.n_cand     = int(1e5)
    simparams.n_test     = int(2e5)
    simparams.n_pred     = int(1e7)
    simparams.n_splits   = 50
    simparams.alphas     = 2
    n_initial = 20
    simparams.info()
    ## ------------------------ Simulation Parameters ----------------- ###
    n_cand = int(1e5)
    ndim   = 2
    alphas = [1.2, 2.0]
    pce_degs = np.arange(2,4)
    u_dist = [stats.uniform(),]*ndim
    # u_dist = [stats.norm(),]*ndim
    data   = Data()
    for ialpha in alphas:
        for deg in pce_degs:
            kappa = []
            if u_dist[0].dist.name == 'uniform':
                pce = uqra.Legendre(d=ndim, deg=deg)
                fname_cand = 'DoE_McsE5R0_uniform.npy'
                fname_opt  = 'DoE_McsE5R0_{:d}Leg{:d}_OPT.npy'.format(ndim, deg)
            elif u_dist[0].dist.name == 'norm':
                pce = uqra.Hermite(d=ndim, deg=deg, hem_type='prob')
                fname_cand = 'DoE_McsE5R0_norm.npy'
                fname_opt  = 'DoE_McsE5R0_{:d}Hem{:d}_OPT.npy'.format(ndim, deg)
            else:
                raise ValueError

            ## MCS
            x = uqra.MCS(u_dist).samples(size=int(ialpha*pce.num_basis))
            X = pce.vandermonde(x)
            ## condition number, kappa = max(svd)/min(svd)
            _, sig_value, _ = np.linalg.svd(X)
            kappa.append(max(abs(sig_value)) / min(abs(sig_value)))

            ## MCS-D
            x_cand     = np.load(os.path.join(data_dir_oed, fname_cand))
            fname_optD = fname_opt.replace('OPT', 'D')
            oed_idx    = np.load(os.path.join(data_dir_oed,fname_optD))
            x = x_cand[:, oed_idx[:int(ialpha*pce.num_basis)]] 
            X = pce.vandermonde(x)
            ## condition number, kappa = max(svd)/min(svd)
            _, sig_value, _ = np.linalg.svd(X)
            kappa.append(max(abs(sig_value)) / min(abs(sig_value)))


            ## MCS-S
            x_cand = np.load(os.path.join(data_dir_oed, fname_cand))
            fname_optS = fname_opt.replace('OPT', 'S')
            oed_idx= np.load(os.path.join(data_dir_oed,fname_optS))
            x = x_cand[:, oed_idx[:int(ialpha*pce.num_basis)]] 
            X = pce.vandermonde(x)
            ## condition number, kappa = max(svd)/min(svd)
            _, sig_value, _ = np.linalg.svd(X)
            kappa.append(max(abs(sig_value)) / min(abs(sig_value)))

            ## CLS
            if u_dist[0].dist.name == 'uniform':
                pce = uqra.Legendre(d=ndim, deg=deg)
                fname_cand = 'DoE_Cls1E5D{:d}R0.npy'.format(ndim)
                fname_opt  = 'DoE_Cls1E5R0_{:d}Leg{:d}_OPT.npy'.format(ndim, deg)
            elif u_dist[0].dist.name == 'norm':
                pce = uqra.Hermite(d=ndim, deg=deg, hem_type='phy')
                fname_cand = 'DoE_Cls4E5D{:d}R0.npy'.format(ndim)
                fname_opt  = 'DoE_Cls4E5R0_{:d}Hem{:d}_OPT.npy'.format(ndim, deg)
            else:
                raise ValueError

            x = uqra.CLS('CLS1', ndim).samples(size=int(ialpha*pce.num_basis))
            X = pce.vandermonde(x)
            ## condition number, kappa = max(svd)/min(svd)
            _, sig_value, _ = np.linalg.svd(X)
            kappa.append(max(abs(sig_value)) / min(abs(sig_value)))
            ## CLS-D
            x_cand     = np.load(os.path.join(data_dir_oed, fname_cand))
            fname_optD = fname_opt.replace('OPT', 'D')
            oed_idx    = np.load(os.path.join(data_dir_oed,fname_optD))
            x = x_cand[:, oed_idx[:int(ialpha*pce.num_basis)]] 
            X = pce.vandermonde(x)
            ## condition number, kappa = max(svd)/min(svd)
            _, sig_value, _ = np.linalg.svd(X)
            kappa.append(max(abs(sig_value)) / min(abs(sig_value)))

            ## CLS-S

            print(kappa)

if __name__ == '__main__':
    main()
