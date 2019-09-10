#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import context
import museuq
import numpy as np, chaospy as cp
from museuq.doe.ExperimentDesign import ExperimentDesign

def main():
    doe_method, doe_rule, doe_orders = 'QUAD', 'hem', [4,5]
    dist1 = cp.Uniform()

    quad_doe = ExperimentDesign(doe_method, doe_rule, doe_orders, dist1)
    quad_doe.get_samples()

if __name__ == '__main__':
    main()

