#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import chaospy as cp
import numpy as np

from doe_generator import samplegen

dist = cp.J(cp.Uniform(), cp.Uniform())
x,w  = samplegen('GQ',5,dist)
print(x.shape)
print(w.shape)
