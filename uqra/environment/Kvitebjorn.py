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

    def pdf(self, x):
        """
        Return pdf values for given random variables x
        parameters:
            x, ndarray of shape (2, n)
        Return:
            y, ndarray of shape(2, n)
        """
        if x.shape[0] != 2:
            raise ValueError('Kvitebjørn site expects two random variables (Hs, Tp), but {:d} were given'.format(x.shape[0]))
        
        hs = np.squeeze(x[0,:])
        tp = np.squeeze(x[1,:])
        hs_pdf = np.squeeze(self._hs_pdf(hs))
        tp_pdf = np.squeeze(self._tp_pdf(tp, hs))
        y = np.array(hs_pdf * tp_pdf)
        return y

    def cdf(self, x):
        """
        Return cdf values for given random variables x
        parameters:
            x, ndarray of shape (2, n)
        Return:
            y, ndarray of shape(2, n)
        """
        if x.shape[0] != 2:
            raise ValueError('Kvitebjørn site expects two random variables (Hs, Tp), but {:d} were given'.format(x.shape[0]))
        
        hs = np.squeeze(x[0,:])
        tp = np.squeeze(x[1,:])
        hs_cdf = np.squeeze(self._hs_cdf(hs))
        tp_cdf = np.squeeze(self._tp_cdf(tp, hs))
        y = np.array([hs_cdf, tp_cdf])
        return y

    def ppf(self, u):
        """
        Return Percent point function (inverse of cdf — percentiles) corresponding to u.

        """
        u = np.array(u, ndmin=2)
        ### make sure u is valid cdf values
        assert np.amin(u) >= 0
        assert np.amax(u) <= 1

        tp_cdf = None if u.shape[0] == 1 else u[1,:]
        hs = self.samples_hs_ppf(u[0,:])
        tp = self.samples_tp(hs, tp_cdf=tp_cdf)
        return np.array([hs, tp])

    def rvs(self, size=None):
        """
        Generate random sample for Kvitebjørn

        """
        n = int(size)
        u = np.random.uniform(0,1,n)
        ### generate n random Hs
        hs= self.samples_hs_ppf(u)
        ### generate n random Tp given above Hs
        tp= self.samples_tp(hs)
        res = np.array([hs, tp]).reshape(2,-1)
        return res

    def get_environment_contour(self, P,T=1000,n=100):
        """
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
        beta        = stats.norm.ppf(1-prob_fail) ## reliability index
        print(r' - {:<25s}: {:.2e}'.format('Failure probability', prob_fail))
        print(r' - {:<25s}: {:.2f}'.format('Reliability index', beta))
        U_samples   = self._make_circles(beta,n=n)
        U_cdf       = stats.norm.cdf(U_samples)
        Hs          = self.samples_hs_ppf(U_cdf[0])
        Tp          = self.samples_tp(Hs, tp_cdf=U_cdf[1])
        X_samples   = np.array([Hs, Tp])
        res         = np.vstack((U_samples, X_samples))
        return res 

# Sequence of conditional distributions based on Rosenblatt transformation 
    def dist_Hs(self, x, key='value'):
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
        mu_Hs    = 0.77
        sigma_Hs = 0.6565
        dist1    = stats.lognorm(sigma_Hs, scale=np.exp(mu_Hs))
        Hs_shape = 1.503
        Hs_scale = 2.691
        dist2    = stats.Weibull(Hs_shape, scale=Hs_scale)
        h0       = 2.9
        ppf_h0   = dist1.cdf(h0)

        if key.lower() == 'value':
            dist = dist1 if x <= h0 else dist2
        elif key.lower() == 'ppf':
            dist = dsit1 if x <= ppf_h0 else dist2

        else:
            raise ValueError('Key value: {} is not defined'.format(key))

    def dist_Tp(self, Hs):
        a1 = 1.134
        a2 = 0.892
        a3 = 0.225
        b1 = 0.005
        b2 = 0.120
        b3 = 0.455
        mu_tp = a1 + a2* Hs**a3 
        sigma_tp = np.sqrt(b1 + b2*np.exp(-b3*Hs))
        return stats.lognorm(sigma_tp, scale=np.exp(mu_tp))

    def samples_hs_ppf(self, u):
        """
        Return Hs samples corresponding ppf values u
        """

        mu_Hs    = 0.77
        sigma_Hs = 0.6565
        Hs_shape = 1.503
        Hs_scale = 2.691
        h0       = 2.9

        u = np.array(u,ndmin=1)
        assert (min(u) >=0).all() and (max(u) <=1).all(), 'CDF values should be in range [0,1]'
        samples1 = stats.lognorm.ppf(u, s=sigma_Hs, loc=0, scale=np.exp(mu_Hs))
        samples2 = stats.weibull_min.ppf(u, c=Hs_shape, loc=0, scale=Hs_scale) #0 #Hs_scale * (-np.log(1-u)) **(1/Hs_shape)
        samples_hs = np.where(samples1<=h0,samples1, samples2)

        return np.squeeze(samples_hs)

    def samples_tp(self, hs,tp_cdf=None):
        """
        Generate Tp sample values based on given Hs values:
        two steps:
            1. get the conditional distributions of Tp given Hs
            2. If tp_cdf is given, return the corresponding ppf values
                else, return a random sample from distribution Tp|Hs
        """

        a1 = 1.134
        a2 = 0.892
        a3 = 0.225
        b1 = 0.005
        b2 = 0.120
        b3 = 0.455
        mu_tp   = a1 + a2* hs**a3 
        sigma_tp= np.sqrt(b1 + b2*np.exp(-b3*hs))

        if tp_cdf is None:
            samples = np.random.lognormal(mean=mu_tp, sigma=sigma_tp, size=np.size(hs)) 
        else:
            samples = stats.lognorm.ppf(tp_cdf, s=sigma_tp, loc=0, scale=np.exp(mu_tp)) 

        return np.squeeze(samples)

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
    def _hs_pdf(self, hs):
        """
        Return pdf value for give hs
        """
        mu_Hs    = 0.77
        sigma_Hs = 0.6565
        Hs_shape = 1.503
        Hs_scale = 2.691
        h0       = 2.9

        hs1_pdf = stats.lognorm.pdf(hs, s=sigma_Hs, loc=0, scale=np.exp(mu_Hs)) 
        hs2_pdf = stats.weibull_min.pdf(hs, c=Hs_shape, scale=Hs_scale) 
        pdf  = np.where(hs<=h0, hs1_pdf, hs2_pdf)
        return pdf

    def _tp_pdf(self, tp, hs):
        res = np.array([self.dist_Tp(ihs).pdf(itp) for itp, ihs in zip(tp, hs)])
        # res = np.where(res==np.NaN, 1, res)
        return res

    def _hs_cdf(self, hs):
        """
        Return pdf value for give hs
        """
        mu_Hs    = 0.77
        sigma_Hs = 0.6565
        Hs_shape = 1.503
        Hs_scale = 2.691
        h0       = 2.9

        hs1_cdf = stats.lognorm.cdf(hs, s=sigma_Hs, loc=0, scale=np.exp(mu_Hs)) 
        hs2_cdf = stats.weibull_min.cdf(hs, c=Hs_shape, scale=Hs_scale) 
        hs_cdf  = np.where(hs<=h0, hs1_cdf, hs2_cdf)
        return hs_cdf

    def _tp_cdf(self, tp, hs):
        res = np.array([self.dist_Tp(ihs).cdf(itp) for itp, ihs in zip(tp, hs)])
        res = np.where(res==np.NaN, 1, res)
        return res


