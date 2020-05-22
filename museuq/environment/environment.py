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

    def __init__(self, args):
        self.rvs_args, self.name = self._check_args(args)
        self.ndim       = len(self.rvs_args)

    def pdf(self, x):
        x = np.array(x, copy=False, ndmin=2)
        assert x.shape[0] == self.ndim
        pdf_joint = np.array([idist.pdf(ix) for ix, idist in zip(x, self._dist_args)])
        pdf_joint = np.prod(pdf_joint, axis=0)
        return pdf_joint

    def rvs(self, size=None):
        n = int(size)
        samples = np.array([idist.rvs(size=n) for idist in self._dist_args])
        return samples

    def cdf(self, x):
        x = np.array(x, copy=False, ndmin=2)
        assert x.shape[0] == self.ndim
        cdf_joint = np.array([idist.cdf(ix) for ix, idist in zip(x, self._dist_args)])
        cdf_joint = np.prod(cdf_joint, axis=0)
        return cdf_joint

    def ppf(self, u):
        u = np.array(u, copy=False, ndmin=2)
        ### make sure u is valid cdf values
        if np.amin(u) < 0 or np.amax(u) > 1:
            raise ValueError('values for ppf function must be in range[0,1], but min:{}, max{} was given'.format(np.amin(u), np.amax(u)))
        assert u.shape[0] == self.ndim
        ppf_joint = np.array([idist.ppf(iu) for iu, idist in zip(u, self._dist_args)])
        return ppf_joint

    def _check_args(self, args):
        self._dist_args = []
        rvs_args        = []
        names           = []
        if isinstance(args, list):
            for iarg in args:
                if museuq.isfromstats(iarg):
                    self._dist_args.append(iarg)
                    rvs_args.append(iarg)
                    names.append(self._get_dist_name(iarg))
                elif np.ndim(iarg) == 0 :
                    self._dist_args.append(stats.uniform(loc=iarg, scale=0))
                    names.append('const')
                else:
                    raise ValueError('Environment: args type {} for Environment object is not defined'.format(type(args)))
        elif museuq.isfromstats(args):
            self._dist_args.append(args)
            rvs_args.append(args)
            names.append(self._get_dist_name(args))
        elif np.ndim(args) == 0 :
            self._dist_args.append(stats.uniform(loc=args, scale=0))
            names.append('const')
        else:
            raise ValueError('Environment: args type {} for Environment object is not defined'.format(type(args)))
        return rvs_args, names

    def _get_dist_name(self, dist):
        try:
            name = dist.name
        except AttributeError:
            name = dist.dist.name

        return name


