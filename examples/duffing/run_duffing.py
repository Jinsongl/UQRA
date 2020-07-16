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

def main():

    f = lambda t: 8 * np.cos(0.5 *t)
    np.random.seed(100)
    dt =0.12566370614359174 
    out_responses = [1,2]
    nsim = 1
    out_stats = ['mean', 'std', 'skewness', 'kurtosis', 'absmax', 'absmin']
    method = 'LSODA'
    solver = uqra.duffing_oscillator(m=1,c=0.02,k=1,s=5,out_responses=out_responses, excitation=f,out_stats=out_stats, tmax=18000, dt=dt,y0=[0,0], method=method)
    print(solver)
    data_dir_src    = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/Kvitebjorn/Normal/'
    data_dir_destn  = r'/Volumes/External/MUSE_UQ_DATA/Duffing/Data/' 
    batch_size = int(10000)
    for i in range(14,15):
        batch_start = (i)*batch_size
        batch_end   = (i+1)*batch_size
        data = []
        filename = 'DoE_McsE6R{:d}.npy'.format(0)
        x = np.load(os.path.join(data_dir_src, filename))[:solver.ndim, 0]
        # x = np.load(os.path.join(data_dir_src, filename))[:solver.ndim, batch_start:batch_end]
        # x = solver.map_domain(u, [stats.norm(0,1),] * solver.ndim) 
        # y_raw, y_QoI = zip(*[solver.run(x.T) for _ in range(nsim)]) 
        y_raw, y_QoI = solver.run(x.T)
        np.save('duffing_time_series', y_raw)
        # filename = 'DoE_McsE6R0_{:d}_stats'.format(i)
        # np.save(os.path.join(data_dir_destn, filename), y_QoI)

    # np.random.seed(100)
    # out_responses = [1,2]
    # out_stats = ['mean', 'std', 'skewness', 'kurtosis', 'absmax', 'absmin']
    # solver = uqra.linear_oscillator(out_responses=out_responses, out_stats=out_stats)
    # print()
    # data_dir = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/Kvitebjorn/Normal/'
    # for r in range(10):
        # data = []
        # filename = 'DoE_McsE6R{:d}.npy'.format(r)
        # x = np.load(os.path.join(data_dir, filename))
        # # x = solver.map_domain(u, [stats.norm(0,1),] * solver.ndim) 
        # y_raw, y_QoI = solver.run(x.T) 
        # print(np.array(y_QoI).shape)

if __name__ == '__main__':
    main()
