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
import itertools, copy, math
import multiprocessing as mp
# warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()
class Data():
    pass

def observation_error(y, mu=0, cov=0.03, random_state=100):
    e = stats.norm(0, cov * abs(y)).rvs(size=len(y), random_state=random_state)
    return e

def main(s=0):

    ## ------------------------ Displaying set up ------------------- ###
    print('\n#################################################################################')
    print(' >>>  Start UQRA : {:d}'.format(s), __file__)
    print('#################################################################################\n')
    np.random.seed(100)
    np.set_printoptions(precision=4)
    np.set_printoptions(threshold=8)
    np.set_printoptions(suppress=True)
    ## ------------------------ Define solver ----------------------- ###
    # solver      = uqra.ExpAbsSum(stats.uniform(-1,2),d=2,c=[-2,1],w=[0.25,-0.75])
    # solver      = uqra.ExpSquareSum(stats.uniform(-1,2),d=2,c=[1,1],w=[1,0.5])
    # solver      = uqra.CornerPeak(stats.uniform(-1,2), d=2)
    # solver      = uqra.ProductPeak(stats.uniform(-1,2), d=2,c=[-3,2],w=[0.5,0.5])
    # solver      = uqra.Franke()
    # solver      = uqra.Ishigami()
    # poly_name   = 'Leg'

    # solver      = uqra.ExpAbsSum(stats.norm(0,1),d=2,c=[-2,1],w=[0.25,-0.75])
    # solver      = uqra.ExpSquareSum(stats.norm(0,1),d=2,c=[1,1],w=[1,0.5])
    # solver      = uqra.CornerPeak(stats.norm(0,1), d=3, c=np.array([1,2,3]), w=[0.5,]*3)
    # solver      = uqra.ProductPeak(stats.norm(0,1), d=2, c=[-3,2], w=[0.5,]*2)
    # solver      = uqra.ExpSum(stats.norm(0,1), d=3)
    solver      = uqra.FourBranchSystem()
    uqra_env = solver.distributions[0]
    poly_name  = 'Hem'

    model_dir  = os.path.join('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/UQRA_Examples', solver.nickname)
    data_dir   = os.path.join(model_dir, 'Data')
    figure_dir = os.path.join(model_dir, 'Figures')
    data_dir_test = os.path.join(model_dir, 'TestData')

    n_pred = int(1e8)
    pf = np.array([1e-6])
    ## ----------- predict data set ----------- ###
    filename = '{:s}_{:d}{:s}.npy'.format(solver.nickname, solver.ndim, poly_name.capitalize())
    data_pce = np.load(os.path.join(data_dir, filename), allow_pickle=True)
    doe_sampling = ['Mcs','McsD','McsS', 'Cls4', 'Cls4S', 'Cls4D', 'Lhs']

    mcs_cdf = stats.uniform(0,1).rvs(size=(2,n_pred))
    u = []
    x = []
    xi= []
    y = []
    for r in range(10):
        filename   = '{:s}_CDF_McsE6R{:d}.npy'.format(solver.nickname, r)
        print(filename)
        data_test_ = np.load(os.path.join(data_dir_test, filename), allow_pickle=True).tolist()
        u.append(data_test_.u)
        xi.append(data_test_.xi)
        x.append(data_test_.x)
        y.append(data_test_.y)
    data_test = uqra.Data()
    data_test.u = np.concatenate(u, axis=-1)
    data_test.x = np.concatenate(x, axis=-1)
    data_test.xi= np.concatenate(xi,axis=-1)
    data_test.y = np.concatenate(y, axis=-1)

    y_boots = np.squeeze(uqra.bootstrapping(data_test.y.reshape(-1,1), 10, bootstrap_size=n_pred)).T
    print(y_boots.shape)


    data2plot = uqra.Data()
    for idoe_sampling in doe_sampling:
        idata2plot = uqra.Data()
        idata2plot.deg    = np.array([idata.deg for idata in data_pce])
        idata2plot.kappa  = np.array([getattr(idata, idoe_sampling).kappa for idata in data_pce])
        idata2plot.rmse_y = np.array([getattr(idata, idoe_sampling).rmse_y for idata in data_pce])
        idata2plot.y0_hat = np.array([getattr(idata, idoe_sampling).y0_hat for idata in data_pce])
        idata2plot.cv_err = np.array([getattr(idata, idoe_sampling).cv_error for idata in data_pce])
        idata2plot.model  = [getattr(idata, idoe_sampling).model for idata in data_pce]
        idata2plot.y0_ecdf= [uqra.ECDF(y_boots, alpha=pf, compress=True)]
        idata2plot.y0_pf  = uqra.metrics.mquantiles(y_boots, 1-pf, multioutput='raw_values')
        idata2plot.y0_hat_ecdf= []
        idata2plot.y0_hat_pf  = []

        print(idata2plot.deg.shape)
        print(idata2plot.kappa.shape)
        print(idata2plot.rmse_y.shape)
        print(idata2plot.y0_hat.shape)
        print(idata2plot.cv_err.shape)
        print(len(idata2plot.model))


        for ideg_model in idata2plot.model:
            print('>> PCE model: ndim={}, deg={}, DoE={}'.format(
                ideg_model.ndim, ideg_model.deg, idoe_sampling))
            if idoe_sampling.lower().startswith('cls'):
                u_pred = data_test.xi
            else:
                u_pred = data_test.u
            print('   - U[mean, std]: {} {}'.format(np.mean(u_pred, axis=-1), np.std(u_pred, axis=-1)))
            parallel_batch_size = int(1e6)
            parallel_batch_num  = math.ceil(u_pred.shape[1]/parallel_batch_size)
            with mp.Pool(processes=mp.cpu_count()) as p:
                y_pred = list(tqdm(p.imap(ideg_model.predict, 
            [(u_pred[:, i*parallel_batch_size: (i+1)*parallel_batch_size]) for i in range(parallel_batch_num)]),
            ncols=80, total=parallel_batch_num))
            y_pred = np.concatenate(y_pred, axis=0)
            # y_pred = ideg_model.predict(u_pred)
            idata2plot.y0_hat_ecdf.append(uqra.ECDF(y_pred, alpha=pf, compress=True))
            idata2plot.y0_hat_pf.append(uqra.metrics.mquantiles(y_pred, 1-pf))
        idata2plot.y0_hat_pf = np.array(idata2plot.y0_hat_pf)
        print('   - y0_hat_pf: {}'.format(idata2plot.y0_hat_pf))
        attr_name = ''.join([i for i in idoe_sampling if not i.isdigit() ])
        setattr(data2plot, attr_name, idata2plot)
    np.save(os.path.join(data_dir, '{:s}_{:d}{:s}_plot.npy'.format(solver.nickname, solver.ndim, poly_name.capitalize())), data2plot, allow_pickle=True)


if __name__ == '__main__':
    main(0)
