#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
class EnvBase(object):

    def __init__(self):
        pass

    def pdf(self):
        raise NotImplementedError

    def samples(self, n):
        raise NotImplementedError

    def iid_joint(self):
        raise NotImplementedError
