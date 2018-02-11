#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import chaospy as cp
import numpy as np
import numpy.random as rn
import scipy as sp
from scipy.stats import norm
from scipy.stats import gamma
import matplotlib.pyplot as plt
from sklearn import svm
import os.path
# import Norway5 as envi
# from SDOF import * 
# from genVar import *
# from getStats import *
# import csv
# from mpl_toolkits.mplot3d import Axes3D

# data = np.genfromtxt("Norway5EC2D_HT_50.csv",delimiter=',')
# with open("testData.csv", 'wb') as fileid:
    # writer = csv.writer(fileid)
    # for hstp in data:
        # f_obs = SDOF(hstp[0],hstp[1], 1000, seed=[1,1000])
        # f_stats = getStats(f_obs, outputs=[1,2], stats=[1,1,1,1,1,1,0])
        # f_stats = np.array(f_stats)
        # v = [hstp[0],hstp[1],f_stats[1,4]]
        # writer.writerow(['{:8.4e}'.format(float(x)) for x in v])

def model_solver(q):
    return [q[0]*q[1], q[0]**3-q[1]+1]
dist = cp.Iid(cp.Normal(0,1),2)
cor, w = cp.generate_quadrature(8,dist,rule='G')

solves = np.array([model_solver(c) for c in cor.T ])
print solves


for order in xrange(1,10):

    orthPoly,norms = cp.orth_ttr(order,dist,retall=True)
    f_hat = cp.fit_quadrature(orthPoly,cor, w,solves[:,1],norms=norms)
# print cp.decompose(f_hat)
    f_fit = []
    for c in cor.T:
        f_fit.append(f_hat(*c))
    f_fit = np.array(f_fit) 
    print 'order:', order, np.sqrt(np.mean((f_fit - solves[:,1])**2))
        # f_fit.append(f.tolist())

