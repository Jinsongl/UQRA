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
import museuq
from ._envbase import EnvBase
import scipy.stats as stats

class Environment(EnvBase):
    """
    Environment class based on a list of iid marginal distributions
    """

    def __init__(self, spectrum, *args):
        self.spectrum   = spectrum
        self.is_arg_rand, self.name = self._check_args(args)
        self.rvs_args   = [idist for (idist, is_rand) in zip(self._dist_args, self.is_arg_rand) if is_rand]
        self.ndim       = np.size(self.rvs_args)


    def rvs(self, size=None):
        samples = np.array([idist.rvs(size=size) for idist in self._dist_args])
        return samples

    def pdf(self, x):
        x = np.array(x, copy=False, ndmin=2) ## empty array will return another empty array with shape (1,0) 

        size = max(1, x.shape[1])
        idim = 0
        res  = []
        for is_rand, idist in zip(self.is_arg_rand, self._dist_args):
            if is_rand:
                ix = x[idim]
                idim += 1
                ### make sure u is valid cdf values
                res.append(idist.pdf(ix))
            else:
                res.append(np.ones((1,size)))
        res = np.vstack(res)
        res = np.prod(res, axis=0)
        return res

    def cdf(self, x):
        x = np.array(x, copy=False, ndmin=2) ## empty array will return another empty array with shape (1,0) 
        size = max(1, x.shape[1])
        idim = 0
        res  = []
        for is_rand, idist in zip(self.is_arg_rand, self._dist_args):
            if is_rand:
                ix = x[idim]
                idim += 1
                ### make sure u is valid cdf values
                res.append(idist.cdf(ix))
            else:
                res.append(np.ones((1,size)))
        res = np.vstack(res)
        res = np.prod(res, axis=0)
        return res


    def ppf(self, u):
        u = np.array(u, copy=False, ndmin=2) ## empty array will return another empty array with shape (1,0) 
        size = max(1, u.shape[1])
        idim = 0
        res  = []
        for is_rand, idist in zip(self.is_arg_rand, self._dist_args):
            if is_rand:
                iu = u[idim]
                idim += 1
                ### make sure u is valid cdf values
                if np.amin(u) < 0 or np.amax(u) > 1:
                    raise ValueError('values for ppf function must be in range[0,1], but min:{}, max{} was given'.format(np.amin(u), np.amax(u)))
                res.append(idist.ppf(iu))
            else:
                res.append(idist.rvs(size=(1,size)))
        res = np.vstack(res)
        return res

    def _check_args(self, args):
        self._dist_args = []
        is_arg_rand     = []
        names           = []
        if isinstance(args, (list, tuple)):
            for iarg in args:
                if museuq.isfromstats(iarg):
                    self._dist_args.append(iarg)
                    is_arg_rand.append(True)
                    names.append(self._get_dist_name(iarg))
                elif np.ndim(iarg) == 0 :
                    self._dist_args.append(stats.uniform(loc=iarg, scale=0))
                    is_arg_rand.append(False)
                    names.append('C')
                else:
                    raise ValueError('Environment: args type {} for Environment object is not defined'.format(type(args)))
        elif museuq.isfromstats(args):
            self._dist_args.append(args)
            is_arg_rand.append(True)
            names.append(self._get_dist_name(args))
        elif np.ndim(args) == 0 :
            self._dist_args.append(stats.uniform(loc=args, scale=0))
            is_arg_rand.append(False)
            names.append('C')
        else:
            raise ValueError('Environment: args type {} for Environment object is not defined'.format(type(args)))
        return is_arg_rand, names

    def _get_dist_name(self, dist):
        try:
            name = dist.name
        except AttributeError:
            name = dist.dist.name

        return name


