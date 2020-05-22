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

    def __init__(self, marginals):
        self.marginals  = self._check_marginals(marginals)
        self.ndim       = len(self.marginals)
        self.name       = self._get_dist_names(self.marginals)

    def pdf(self, x):
        x = np.array(x, copy=False, ndmin=2)
        assert x.shape[0] == self.ndim
        pdf_joint = np.array([idist.pdf(ix) for ix, idist in zip(x, self.marginals)])
        pdf_joint = np.prod(pdf_joint, axis=0)
        return pdf_joint

    def rvs(self, size=None):
        n = int(size)
        samples = np.array([idist.rvs(size=n) for idist in self.marginals])
        return samples

    def cdf(self, x):
        x = np.array(x, copy=False, ndmin=2)
        assert x.shape[0] == self.ndim
        cdf_joint = np.array([idist.cdf(ix) for ix, idist in zip(x, self.marginals)])
        cdf_joint = np.prod(cdf_joint, axis=0)
        return cdf_joint

    def ppf(self, u):
        u = np.array(u, copy=False, ndmin=2)
        ### make sure u is valid cdf values
        assert np.amin(u) >= 0
        assert np.amax(u) <= 1
        assert u.shape[0] == self.ndim
        ppf_joint = np.array([idist.ppf(iu) for iu, idist in zip(u, self.marginals)])
        return ppf_joint

    def _check_marginals(self, dists):
        marginals = []
        if isinstance(dists, list):
            for idist in dists:
                if museuq.isfromstats(idist):
                    marginals.append(idist)
                elif np.ndim(idist) == 0 :
                    marginals.append(stats.uniform(loc=idist, scale=0))
                else:
                    raise ValueError('Environment: marginal distribution must from scipy.stats')
        elif museuq.isfromstats(dists):
            marginals.append(dists)
        elif np.ndim(dists) == 0 :
            marginals.append(stats.uniform(loc=dists, scale=0))
        else:
            raise ValueError('Environment: marginal distribution must from scipy.stats')
        return marginals

    def _get_dist_names(self, dists):
        names = []
        for idist in dists:
            try:
                names.append(idist.name)
            except AttributeError:
                names.append(idist.dist.name)

        return names 


