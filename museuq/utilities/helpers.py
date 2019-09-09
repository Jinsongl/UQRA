#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
def num2print(n):
    if n<100:
        return '{:2d}'.format(n)
    else:
        __str ='{:.0E}'.format(n) 
        return __str[0]+'E'+__str[-1] 

