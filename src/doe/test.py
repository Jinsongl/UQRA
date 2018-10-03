#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""

import numpy as np
import numpy.random as nrand
import numpy.linalg as nla
import matplotlib.pyplot as plt
from doe_quasi_optimal import get_quasi_optimal
from test_func import gaussian_peak 
from test_func import plot_func

def test_f(a, *args):
    c = args[0]
    print(c)

# a = 1
# b = [1,2]
# test_f(a,b)
# subset_size_= np.arange(21,110,5)
# error = np.load('gaussian_peak.npy')
# print(error.shape)

# fig = plt.figure()
# plt.semilogy(subset_size_, error,'-*')
# plt.legend(['Averaged MC','Quasi MC', 'Quasi Optimal Design','All'])
# plt.show()


l = '4341534353331332254433'
l1 ='........4....42......4' 
for a,b in enumerate(l1):
    if b is not '.':
        print(a+1, 'correct:', l[a])
    
