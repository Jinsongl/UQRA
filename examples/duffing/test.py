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
from test1 import nextpow2

print(nextpow2(3))

def func(x, *args, plus=0):
    res = np.cos(x) + plus

    for i in args:
        res = res * i 
    return res 

# def ff(f1,n, *args, **kwargs):
    # return n*f1(*args, **kwargs)

x = 1
print(func(x))
print(func(x,-1))
print(func(x,plus=1))
print(func(x,-1,plus=1))
# nf = ff(func,2)
# print(nf(x, plus=10))

# def use_logging(level):
def decorator(func,level):
    def wrapper(*args, **kwargs):
        return func(*args,**kwargs) / level
    return wrapper

# return decorator



# def normalize(func):

    # def wrapper(*args, **kwargs):
            # # args是一个数组，kwargs一个字典
            # logging.warn("%s is running" % func.__name__)
            # return func(*args, **kwargs)
        # return wrapper

# @normalize
# ff = use_logging(level=2)
ff = decorator(func,1)
print(ff(x))
print(ff(x,-1))
print(ff(x,plus=1))
print(ff(x,-1,plus=1))


