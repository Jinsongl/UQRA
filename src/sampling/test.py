#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import numpy as np
import matplotlib.pyplot as plt

def test(size,base=2.0):
    stop = int(np.log(size) / np.log(base)) + 1
    ind = np.logspace(0,stop,num = stop+1, base=base, dtype=int)
    return ind
ind = test(1E7,base=10.0)
ind2 = 1E7 - ind
print ind
print ind2


plt.plot(ind[:-1],'o-')
plt.plot(ind2[:-1],'o-')
plt.show()


