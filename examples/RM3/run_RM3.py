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
import numpy as np, os, sys, io
import scipy.stats as stats
from tqdm import tqdm
import itertools, copy, math, collections
import multiprocessing as mp
import random
import scipy
import matlab.engine
# warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()



def main(solver, r=0, random_state=None):
    random.seed(random_state)

    RM3_T50 = np.load('RM3T50_Cls4S.npy', allow_pickle=True).tolist()
    yscales = [1,1e6,1e7,1e6]
    for iqoi, iyscale in zip([2,23,24,30], yscales):
        x_train = np.array(RM3_T50[iqoi].x0).T
        print(x_train)
        
        ## get train data, if not available, return training samples to run
        ## set matlabengine workspace variables
        # eng.workspace['deg'] = float(deg)
        # eng.workspace['phaseSeed'] = float(theta)
        y_train = []
        itheta   = 0
        for itheta,(iHs, iTp) in enumerate(tqdm(x_train.T, ncols=80, desc='   [WEC-SIM]' )):
            eng.workspace['phaseSeed'] = float(itheta)
            eng.workspace['Hs'] = float(iHs)
            eng.workspace['Tp'] = float(iTp)
            eng.wecSim(nargout=0,stdout=out,stderr=err)
            y = np.squeeze(eng.workspace['maxima'])
            y_train.append(y)
            y_hat = RM3_T50[iqoi].y0[itheta]
            y = y_train[-1][iqoi+2]/iyscale
            print('\n')
            print(r'    Hs={:.2f}, Tp={:.2f}'.format(iHs, iTp))
            print(r'    y0: True={:.2f}, PCE={:.2f}, epsilon={:.2f}%'.format(y, y_hat,(y_hat-y)/y*100))
        y_train = np.array(y_train)
        RM3_T50[iqoi].y = y_train 

    return RM3_T50

if __name__ == '__main__':
    ## ------------------------ Displaying set up ------------------- ###
    r, theta= 0, 10
    np.random.seed(100)
    random.seed(100)
    np.set_printoptions(precision=4)
    np.set_printoptions(threshold=1000)
    np.set_printoptions(suppress=True)
    uqra_env = uqra.environment.NDBC46022()

    eng = matlab.engine.start_matlab()
    out = io.StringIO()
    err = io.StringIO()
    ## ------------------------ Define solver ----------------------- ###
    # solver = uqra.FPSO(random_state=theta, distributions=uqra_env)
    solver = uqra.Solver('RM3', 2, distributions=uqra_env)
    ## ------------------------ UQRA Modeling Parameters ----------------- ###

    res = []
    for i, irepeat in enumerate(range(1)):
        print('\n#################################################################################')
        print(' >>>  File: ', __file__)
        print(' >>>  Start UQRA : Theta: {:d}, -{:d}'.format(theta,  i))
        print(' >>>  Test data R={:d}'.format(r))
        print('#################################################################################\n')
        res.append(main(solver, r=r, random_state=irepeat))
    if len(res) == 1:
        res = res[0]
    # filename = '{:s}_T50_R{:d}S{:d}'.format(solver.nickname, r, theta)
    filename = 'RM3T50_Cls4S2.npy'
    eng.quit()
    # ## ============ Saving QoIs ============
    try:
        np.save(os.path.join(data_dir_result, filename), res, allow_pickle=True)
        print(' >> Simulation Done! Data saved to {:s}'.format(os.path.join(data_dir_result, filename)))
    except:
        np.save(filename, res, allow_pickle=True)
        print(' >> Simulation Done! Data saved to {:s}'.format(os.path.join(os.getcwd(), filename)))
