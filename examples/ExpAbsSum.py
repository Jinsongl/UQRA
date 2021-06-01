#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import numpy as np
import uqra
from tqdm import tqdm
import scipy.stats as stats
import math
def main():
    pf = 1e-4
    dist_x = stats.uniform(-1,2)
    solver = uqra.ExpAbsSum(dist_x)
    mcs_data = uqra.Data()
    mcs_data.x = dist_x.rvs(size=(2,100000))
    mcs_data.y = solver.run(mcs_data.x)
    mcs_y_ecdf = uqra.ECDF(mcs_data.y, 1-pf)


    ndim, deg = 2, 5
    orth_poly = uqra.poly.orthogonal(ndim, deg, 'Leg')
    pce_model = uqra.PCE(orth_poly)
    dist_u    = stats.uniform(0,1) 
    dist_xi   = orth_poly.weight
    dist_x    = solver.distributions
    pce_model.info()

    alphas = np.arange(1,5,0.5)

    data = []
    for ialpha in alphas:
        data_ialpha = uqra.Data()
        data_ialpha.ndim      = ndim
        data_ialpha.deg       = deg 
        data_ialpha.y0_hat_   = []
        data_ialpha.cv_err_   = []
        data_ialpha.rmse_y_   = []
        data_ialpha.model_    = []
        data_ialpha.score_    = []
        data_ialpha.yhat_ecdf_= []
        data_ialpha.xi_train_ = []
        data_ialpha.x_train_  = []
        data_ialpha.y_train_  = []

        print('alpha = {}'.format(ialpha))
        for _ in tqdm(range(50)):
            
            train_x = dist_x[0].rvs(size=(ndim,math.ceil(ialpha * pce_model.num_basis)))
            train_y = solver.run(train_x)
            pce_model.fit('OLS', train_x, train_y )

            y_test_hat = pce_model.predict(mcs_data.x, n_jobs=4)
            data_ialpha.model_.append(pce_model)
            data_ialpha.rmse_y_.append(uqra.metrics.mean_squared_error(mcs_data.y, y_test_hat, squared=False))
            data_ialpha.y0_hat_.append(uqra.metrics.mquantiles(y_test_hat, 1-pf))
            data_ialpha.score_.append(pce_model.score)
            data_ialpha.cv_err_.append(pce_model.cv_error)
            data_ialpha.yhat_ecdf_.append(uqra.ECDF(y_test_hat, pf, compress=True))
        data.append(data_ialpha)
    filename = 'ExpAbsSum.npy'
    np.save(filename, data, allow_pickle=True)
if __name__ == '__main__':
    main()
