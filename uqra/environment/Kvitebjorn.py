#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""
Long-term extreme response analysis of offshore structures by combining importance sampling with subset simulation Ying Min Low, Xiaoxu Huang

Haver S. On the prediction of extreme wave crest heights. In: Proceedings of the 7th international workshop on wave hindcasting and forecasting, Meteorological Service of Canada. 2002.

Johannessen, K. and Nygaard, E. (2000): “Metocean Design Criteria for Kvitebjørn”, Statoil Report , C193-KVB-N-FD-0001, Rev. date: 2000-12-14, Stavanger, 2000.

"""
import numpy as np
import scipy.stats as stats
import uqra
from ._envbase import EnvBase
#### Hs distribution Class
##################################################
##################################################
class DistHs(object):

    def __init__(self):

        self.name     = 'lognorm_weibull'
        self.mu_Hs    = 0.77
        self.sigma_Hs = 0.6565
        self.Hs_shape = 1.503
        self.Hs_scale = 2.691
        self.h0       = 2.9
        self.dist1    = stats.lognorm(s=self.sigma_Hs, scale=np.exp(self.mu_Hs))
        self.dist2    = stats.weibull_min(c=self.Hs_shape, scale=self.Hs_scale)

    def ppf(self, u):
        """
        Return Hs samples corresponding ppf values u
        """

        assert np.logical_and(u >=0, u <=1).all(), 'CDF values should be in range [0,1]'
        hs1 = self.dist1.ppf(u)
        hs2 = self.dist2.ppf(u)
        hs  = np.where(hs1 < self.h0, hs1, hs2)
        return hs 

    def cdf(self, hs):
        """
        Return Hs cdf 
        """
        hs_cdf1 = self.dist1.cdf(hs)
        hs_cdf2 = self.dist2.cdf(hs)
        hs_cdf  = np.where(hs < self.h0, hs_cdf1, hs_cdf2)
        return hs_cdf

    def rvs(self, size=1):
        hs1 = self.dist1.rvs(size=size)
        hs2 = self.dist2.rvs(size=size)
        hs  = np.where(hs1 < self.h0, hs1, hs2)
        return hs

    def pdf(self, hs):
        hs_pdf1 = self.dist1.pdf(hs)
        hs_pdf2 = self.dist2.pdf(hs)
        hs_pdf  = np.where(hs < self.h0, hs_pdf1, hs_pdf2)
        return hs_pdf

    def get_distribution(self, x, key='value'):
        """
        Return Hs distribution based on x value
        For Kvitebjorn, Hs distribution is a piecewise distribution 
        connected at Hs = H0 or ppf_Hs = ppf_h0 (icdf_h0)

        key = 'value'
        dist_Hs = dist1 if x < h0
                = dist2 if x > h0
        or 

        dist_hs = dist1 if cdf_x < ppf_h0
                = dist2 if cdf_x > ppf_h0

        key = 'ppf'
           - value: physical value of Hs
           - ppf: point percentile value of Hs
        """
        dist1    = stats.lognorm(s=self.sigma_Hs, scale=np.exp(self.mu_Hs))
        dist2    = stats.weibull_min(c=self.Hs_shape, scale=self.Hs_scale)
        ppf_h0   = dist1.cdf(self.h0)

        if key.lower() == 'value':
            dist = dist1 if x <= h0 else dist2
        elif key.lower() == 'ppf':
            dist = dsit1 if x <= ppf_h0 else dist2

        else:
            raise ValueError('Key value: {} is not defined'.format(key))

##################################################
##################################################

class DistTp(object):

    def __init__(self, hs):

        self.a1 = 1.134
        self.a2 = 0.892
        self.a3 = 0.225
        self.b1 = 0.005
        self.b2 = 0.120
        self.b3 = 0.455
        self.hs = hs
        self.dist = stats.lognorm(s=1)

    def rvs(self, size=1):
        mu_tp    = self.a1 + self.a2* self.hs**self.a3 
        sigma_tp = np.sqrt(self.b1 + self.b2*np.exp(-self.b3*self.hs))
        tp       = stats.lognorm.rvs(sigma_tp, loc=0, scale=np.exp(mu_tp), size=[size,self.hs.size])
        tp       = np.squeeze(tp)
        assert self.hs.shape == tp.shape
        return tp 

    def ppf(self, u):
        """
        Generate Tp sample values based on given Hs values:
        """
        mu_tp    = self.a1 + self.a2* self.hs**self.a3 
        sigma_tp = np.sqrt(self.b1 + self.b2*np.exp(-self.b3*self.hs))
        tp       = stats.lognorm.ppf(u, sigma_tp, loc=0, scale=np.exp(mu_tp))
        return tp 

    def cdf(self, tp):
        mu_tp    = self.a1 + self.a2* self.hs**self.a3 
        sigma_tp = np.sqrt(self.b1 + self.b2*np.exp(-self.b3*self.hs))
        tp_cdf   = stats.lognorm.cdf(tp, sigma_tp, loc=0, scale=np.exp(mu_tp))
        return tp_cdf

    def pdf(self, tp):
        mu_tp    = self.a1 + self.a2* self.hs**self.a3 
        sigma_tp = np.sqrt(self.b1 + self.b2*np.exp(-self.b3*self.hs))
        tp_pdf   = stats.lognorm.pdf(tp, sigma_tp, loc=0, scale=np.exp(mu_tp))
        return tp_pdf

##################################################
##################################################
class Kvitebjorn(EnvBase):
    """
    Environment class for site "Kvitebjorn"
    Sequence of conditional distributions based on Rosenblatt transformation 
    """

    def __init__(self, spectrum='jonswap'):
        self.spectrum = spectrum
        self.is_arg_rand = [True, True] 
        self.ndim = int(2)
        self.name = ['lognorm_weibull','lognorm']

    def dist_hs(self):
        return DistHs()

    def dist_tp(self, hs):
        return DistTp(hs)

    def pdf(self, x):
        """
        Return pdf values for given random variables x
        parameters:
            x, ndarray of shape (2, n)
        Return:
            y, ndarray of shape(2, n)
        """
        self._check_input(x)
        hs, tp = x
        hs_pdf = self.dist_hs().pdf(hs)
        tp_pdf = self.dist_tp(hs).pdf(tp)
        x_pdf  = np.array([hs_pdf, tp_pdf])
        return x_pdf

    def jpdf(self, x):
        """
        Return pdf values for given random variables x
        parameters:
            x, ndarray of shape (2, n)
        Return:
            y, ndarray of shape(2, n)
        """
        self._check_input(x)
        
        hs, tp = x
        hs_pdf = self.dist_hs().pdf(hs)
        tp_pdf = self.dist_tp(hs).pdf(tp)
        x_pdf  = hs_pdf * tp_pdf
        return x_pdf

    def cdf(self, x):
        """
        Return cdf values for given random variables x
        parameters:
            x, ndarray of shape (2, n)
        Return:
            y, ndarray of shape(2, n)
        """
        self._check_input(x)
        hs, tp = x
        hs_cdf = self.dist_hs().cdf(hs)
        tp_cdf = self.dist_tp(hs).cdf(tp)
        y      = np.array([hs_cdf, tp_cdf])
        return y

    def jcdf(self, x):
        """
        Return cdf values for given random variables x
        parameters:
            x, ndarray of shape (2, n)
        Return:
            y, ndarray of shape(n,)
        """
        self._check_input(x)
        hs, tp = x
        hs_cdf = self.dist_hs().cdf(hs)
        tp_cdf = self.dist_tp(hs).cdf(tp)
        y      = hs_cdf * tp_cdf
        return y

    def ppf(self, u):
        """
        Return Percent point function (inverse of cdf — percentiles) corresponding to u.

        """
        u = np.array(u, ndmin=2)
        self._check_input(u)
        ### make sure u is valid cdf values
        assert np.amin(u).all() >= 0
        assert np.amax(u).all() <= 1

        hs = self.dist_hs().ppf(u[0])
        tp = self.dist_tp(hs).ppf(u[1])
        return np.array([hs, tp])

    def rvs(self, size=None):
        """
        Generate random sample for Kvitebjørn

        """
        ### generate n random Hs
        hs= self.dist_hs().rvs(size=size)
        ### generate n random Tp given above Hs
        tp= self.dist_tp(hs).rvs(size=1)
        res = np.array([hs, tp])
        return res

    def support(self):
        return ((0, np.inf), (0, np.inf))

    def environment_contour(self, P ,T=1000,n=100):
        """za
        Return samples for Environment Contours method
        
        arguments:
            P: return period in years
            T: simulation duration in seconds
            n: no. of samples on the contour

        Returns:
            ndarray of shape (4, n)
        """
        print(r'Calculating Environment Contour samples for Kvitebjørn:')
        print(r' - {:<25s}: {}'.format('Return period (years)', P))
        print(r' - {:<25s}: {}'.format('Simulation duration (s)', T))
        prob_fail   = 1.0/(P * 365.25*24*3600/T)
        beta        = -stats.norm().ppf(prob_fail) ## reliability index
        print(r' - {:<25s}: {:.2e}'.format('Failure probability', prob_fail))
        print(r' - {:<25s}: {:.2f}'.format('Reliability index', beta))
        U = self._make_circles(beta,n=n)
        X = self.ppf(stats.norm.cdf(U))
        return U, X


    def target_contour(self, hs, P, T=1000, n=100):
        """
        Return EC points for specified points Uw
        """
        prob_fail   = T/(P * 365.25*24*3600)
        beta        = -stats.norm().ppf(prob_fail) ## reliability index
        u1 = stats.norm().ppf(self.dist_hs().cdf(hs))
        u2 = np.sqrt(beta**2 - u1**2)
        tp = self.dist_tp(hs).ppf(stats.norm.cdf(u2))
        res = np.array([hs, tp])
        return res


# Sequence of conditional distributions based on Rosenblatt transformation 


    def _make_circles(self, r,n=100):
        """
        return coordinates of points on a 2D circle with radius r

        Parameters:
            r: radius
            n: number of points on circle
        Return:
            ndarray of shape(2,n)

        """
        t = np.linspace(0, np.pi * 2.0, n)
        t = t.reshape((len(t), 1))
        x = r * np.cos(t)
        y = r * np.sin(t)
        return np.hstack((x, y)).T

    def _check_input(self, x):
        if x.shape[0] != self.ndim:
            raise ValueError('Kvitebjørn site expects two random variables (Hs, Tp), but {:d} were given'.format(x.shape[0]))
