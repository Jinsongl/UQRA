#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import numpy as np

class Error(object):
    """
    Abstract class for observation error 
    """

    def __init__(self, random_seed=None):
        self.random_seed = random_seed

    def samples(self):
        """
        Return error samples
        """
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

class NullError(Error):
    """
    return 0 error
    """
    def __init__(self, random_seed=None):
        super().__init__(random_seed=random_seed)
        self.name = 'Null'
    def samples(self):
        return 0

    def __repr__(self):
        return "Null Error"

    def __str__(self):
        return "Null Error"

class IidError(Error):
    """
    Generate IID error samples from specified distribution
    """

    def __init__(self, name, size=None, theta=None, random_seed=None):
        super().__init__(random_seed=random_seed)
        self.name  = name 
        self.theta = theta
        self.size  = size

    def samples(self):

        if self.name.upper() == 'NORMAL':
            loc, scale = (0.0,1.0) if self.theta is None else self.theta
            errors = np.random.normal(loc=loc, scale=scale, size=size) 

        elif self.name.upper() == 'GUMBEL':
            loc, scale = (0.0,1.0) if self.theta is None else self.theta
            errors = np.random.gumbel(loc=loc, scale=scale, size=size)

        # elif self.name.upper() == 'WEIBULL':
            # errors = scale * np.random.weibull(shape, size=size)
        else:
            raise NotImplementedError("{:s} error is not implemented".format(self.name))
        return errors

    def __repr__(self):
        return "I.I.D Error ({:s} ({}))".format(self.name, self.theta)

    def __str__(self):
        return "I.I.D Error ({:s} ({}))".format(self.name, self.theta)


class CovError(Error):
    """
    Generate errors based on coefficient of variation (CoV) of observed data points. 
    e ~ distribution(0, std=cov * |y(x)|)
    """

    def __init__(self, name, cov=0.1, size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.cov  = cov

    def samples(self):

        if self.name.upper() == 'NORMAL':
            loc, scale = (0.0,self.cov) 
            errors = np.random.normal(loc=loc, scale=scale, size=size) 

        elif self.name.upper() == 'GUMBEL':
            loc, scale = (0.0,self.cov) 
            errors = np.random.gumbel(loc=loc, scale=scale, size=size)

        # elif self.name.upper() == 'WEIBULL':
            # errors = scale * np.random.weibull(shape, size=size)
        else:
            raise NotImplementedError("{:s} error is not implemented".format(self.name))
        return errors


    def __repr__(self):
        return "Cov Error ({:s}(mu=0, cov={:.2f}))".format(self.name, self.cov)

    def __str__(self):
        return "Cov Error ({:s}(mu=0, cov={:.2f}))".format(self.name, self.cov)

