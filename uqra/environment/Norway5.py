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

Johannessen, K. and Nygaard, E. (2000): “Metocean Design Criteria for Norway5”, Statoil Report , C193-KVB-N-FD-0001, Rev. date: 2000-12-14, Stavanger, 2000.

"""
import numpy as np
import scipy.stats as stats
import uqra
from ._envbase import EnvBase


class DistUw(object):
    def __init__(self):
        self.name = 'weibull'
        self.shape= 2.029
        self.loc  = 0
        self.scale= 9.409
        self.dist = stats.weibull_min(c=self.shape, loc=self.loc, scale=self.scale) #0 #Hs_scale * (-np.log(1-u)) **(1/Hs_shape)

    def ppf(self, u):
        """
        Percent point function (inverse of cdf — percentiles)
        """
        assert np.logical_and(u >=0, u <=1).all(), 'CDF values should be in range [0,1]'

        x = self.dist.ppf(u)
        return x

    def cdf(self, x):
        """
        Cumulative distribution function.
        """
        u = self.dist.cdf(x)
        return u

    def rvs(self, size=1, random_state=None):
        """
        Random variates.
        """
        x = self.dist.rvs(size=size, random_state=random_state)
        return x
    
    def pdf(self, x):
        """
        Probability density function.
        """
        y = self.dist.pdf(x)
        return y

class DistHs_Uw(object):
    def __init__(self, uw):
        self.name = 'weibull'
        self.a1, self.a2, self.a3 = 2.136, 0.013, 1.709
        self.b1, self.b2, self.b3 = 1.816, 0.024, 1.787
        self.shape= self.a1 + self.a2 * uw ** self.a3
        self.loc  = 0
        self.scale= self.b1 + self.b2 * uw ** self.b3
        self.dist = stats.weibull_min(c=self.shape, loc=self.loc, scale=self.scale)

    def ppf(self, u):
        """
        Percent point function (inverse of cdf — percentiles)
        """
        assert np.logical_and(u >=0, u <=1).all(), 'CDF values should be in range [0,1]'

        x = self.dist.ppf(u)
        return x

    def cdf(self, x):
        """
        Cumulative distribution function.
        """
        u = self.dist.cdf(x)
        return u

    def rvs(self, size=1, random_state=None):
        """
        Random variates.
        """
        x = self.dist.rvs(size=size, random_state=random_state)
        return x
    
    def pdf(self, x):
        """
        Probability density function.
        """
        y = self.dist.pdf(x)
        return y

class DistTp_HsUw(object):
    def __init__(self, Uw, Hs):
        """
        Conditional distribution of Tp given var

        """
        self.name = 'lognorm'
        theta, gamma = -0.255, 1.0
        e1, e2, e3 = 8.0, 1.938, 0.486
        f1, f2, f3 = 2.5, 3.001, 0.745
        k1, k2, k3 = -0.001, 0.316, -0.145 

        Tp_bar     = e1 + e2 * Hs**e3 
        u_bar      = f1 + f2 * Hs**f3
        niu_Tp     = k1 + k2 * np.exp(Hs*k3)
        mu_Tp      = Tp_bar * (1 + theta * ((Uw - u_bar)/u_bar)**gamma)
        mu_lnTp    = np.log(mu_Tp / (np.sqrt(1 + niu_Tp**2)))
        sigma_lnTp = np.sqrt(np.log(niu_Tp**2 + 1))
        self.shape = sigma_lnTp
        self.loc   = 0
        self.scale = np.exp(mu_lnTp)
        self.dist  = stats.lognorm(self.shape, loc=self.loc, scale=self.scale)

    def ppf(self, u):
        """
        Percent point function (inverse of cdf — percentiles)
        """
        assert np.logical_and(u >=0, u <=1).all(), 'CDF values should be in range [0,1]'

        x = self.dist.ppf(u)
        return x

    def cdf(self, x):
        """
        Cumulative distribution function.
        """
        u = self.dist.cdf(x)
        return u

    def rvs(self, size=1, random_state=None):
        """
        Random variates.
        """
        x = self.dist.rvs(size=size, random_state=random_state)
        return x
    
    def pdf(self, x):
        """
        Probability density function.
        """
        y = self.dist.pdf(x)
        return y


    def dist_tp(self, Hs, Uw):
        # if len(var) == 1: 
            # c1, c2, c3 = 1.886, 0.365, 0.312
            # d1, d2, d3 = 0.001, 0.105, -0.264
            # h = var[0][0]
            # mu_LTC = c1 + c2 * h ** c3

            # sigma_LTC = (d1 + d2 * np.exp(d3 * h))** 0.5

            # dist = cp.Lognormal(mu_LTC, sigma_LTC)
            # return dist
        # elif len(var) == 2:
        return dist

class DistHs(object):
    """
    Hybrid lognormal and Weibull distribution, i.e., the Lonowe model
    """
    def __init__(self):
        self.name     = 'Lonowe'
        self.mu_Hs    = 0.871
        self.sigma_Hs = 0.506
        self.Hs_shape = 1.433
        self.Hs_scale = 2.547
        self.h0       = 5.0
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

class DistTp_Hs(object):
    def __init__(self, hs):
        self.a1 = 1.886
        self.a2 = 0.365
        self.a3 = 0.312
        self.b1 = 0.001
        self.b2 = 0.105
        self.b3 = 0.264
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

class Norway5(EnvBase):
    """
    Reference: 
    Norway 5:
    Li L, Gao Z, Moan T. Joint environmental data at five european offshore sites for design of combined wind and wave
    energy concepts. 32nd International Conference on Ocean, Offshore, and Arctic Engineering, Nantes, France, Paper
    No. OMAE2013-10156, 2013.
    """

    def __init__(self, spectrum='jonswap', ndim=3):
        self.spectrum = spectrum
        self.site = 'Norway5'
        self.ndim = int(ndim)
        self.is_arg_rand = [True, ] * self.ndim
        if self.ndim == 3:
            self.dist_name = ['weibull','weibull','lognorm']
        elif self.ndim == 2:
            self.dist_name = ['Lonowe','lognorm']

    def dist_uw(self):
        return DistUw()

    def dist_hs(self, uw=None):
        if self.ndim==2:
            return DistHs()
        elif self.ndim==3:
            return DistHs_Uw(uw)

    def dist_tp(self, hs, uw=None):
        if self.ndim==2:
            return DistTp_Hs(hs)
        elif self.ndim==3:
            return DistTp_HsUw(uw, hs)

    def pdf(self, x):
        """
        Return pdf values for given random variables x
        parameters:
            x, ndarray of shape (3, n)
        Return:
            y, ndarray of shape(3, n)
        """
        if x.shape[0] == 3:
            uw, hs, tp = x
            uw_pdf = self.dist_uw().pdf(uw)
            hs_pdf = self.dist_hs(uw).pdf(hs)
            tp_pdf = self.dist_tp(uw, hs).pdf(tp)
            pdf_y  = np.array([uw_pdf, hs_pdf, tp_pdf])
        elif x.shape[0] == 2:
            hs, tp = x
            hs_pdf = self.dist_hs().pdf(hs)
            tp_pdf = self.dist_tp(hs).pdf(tp)
            pdf_y  = np.array([hs_pdf, tp_pdf])
        else:
            raise ValueError('uqra.environment.{:s} expecting 2 or 3 random variables but {:d} are given'.format(self.site,x.shape[0]))

        return pdf_y

    def jpdf(self, x):
        """
        Return joint pdf values for given random variables x
        parameters:
            x, ndarray of shape (3, n)
        Return:
            y, ndarray of shape(n,)
        """
        if x.shape[0] == 3:
            uw, hs, tp = x
            uw_pdf = self.dist_uw().pdf(uw)
            hs_pdf = self.dist_hs(uw).pdf(hs)
            tp_pdf = self.dist_tp(uw, hs).pdf(tp)
            pdf_y  = uw_pdf * hs_pdf * tp_pdf
        elif x.shape[0] == 2:
            hs, tp = x
            hs_pdf = self.dist_hs().pdf(hs)
            tp_pdf = self.dist_tp(hs).pdf(tp)
            pdf_y  = hs_pdf * tp_pdf
        else:
            raise ValueError('Norway5 site expects 2 or 3 random variables [(Uw), Hs, Tp], but {:d} were given'.format(x.shape[0]))
        return pdf_y

    def cdf(self, x):
        """
        Return cdf values for given random variables x
        parameters:
            x, ndarray of shape (3, n)
        Return:
            y, ndarray of shape(3, n)
        """
        if x.shape[0] == 3:
            uw, hs, tp = x
            uw_cdf = self.dist_uw().cdf(uw)
            hs_cdf = self.dist_hs(uw).cdf(hs)
            tp_cdf = self.dist_tp(uw, hs).cdf(tp)
            cdf_y  = np.array([uw_cdf , hs_cdf , tp_cdf])
        elif x.shape[0] == 2:
            hs, tp = x
            hs_cdf = self.dist_hs().cdf(hs)
            tp_cdf = self.dist_tp(hs).cdf(tp)
            cdf_y  = np.array([hs_cdf , tp_cdf])
        else:
            raise ValueError('Norway5 site expects 2 or 3 random variables [(Uw), Hs, Tp], but {:d} were given'.format(x.shape[0]))
        return cdf_y

    def jcdf(self, x):
        """
        Return cdf values for given random variables x
        parameters:
            x, ndarray of shape (3, n)
        Return:
            y, ndarray of shape(n,)
        """
        if x.shape[0] == 3:
            uw, hs, tp = x
            uw_cdf = self.dist_uw().cdf(uw)
            hs_cdf = self.dist_hs(uw).cdf(hs)
            tp_cdf = self.dist_tp(uw, hs).cdf(tp)
            cdf_y  = uw_cdf * hs_cdf * tp_cdf
        elif x.shape[0] == 2:
            hs, tp = x
            hs_cdf = self.dist_hs().cdf(hs)
            tp_cdf = self.dist_tp(hs).cdf(tp)
            cdf_y  = hs_cdf * tp_cdf
        else:
            raise ValueError('Norway5 site expects 2 or 3 random variables [(Uw), Hs, Tp], but {:d} were given'.format(x.shape[0]))
        return cdf_y

    def ppf(self, u):
        """
        Return Percent point function (inverse of cdf — percentiles) corresponding to u.

        """
        u = np.array(u, ndmin=2)
        if u.shape[0] == 3:
            ### make sure u is valid cdf values
            assert np.amin(u).all() >= 0
            assert np.amax(u).all() <= 1

            uw = self.dist_uw().ppf(u[0])
            hs = self.dist_hs(uw).ppf(u[1])
            tp = self.dist_tp(hs, uw).ppf(u[2])
            res = np.array([uw, hs, tp])
        elif u.shape[0] == 2:
            ### make sure u is valid cdf values
            assert np.amin(u).all() >= 0
            assert np.amax(u).all() <= 1

            hs = self.dist_hs().ppf(u[0])
            tp = self.dist_tp(hs).ppf(u[1])
            res = np.array([hs, tp])
        else:
            raise ValueError('Norway5 site expects 2 or 3 random variables [(Uw), Hs, Tp], but {:d} were given'.format(x.shape[0]))
        return res 

    def rvs(self, size=None):
        """
        Generate random sample for Norway5

        """
        n  = int(size)
        if self.ndim == 3: 
            ### generate n random Uw
            uw = self.dist_uw().rvs(size=(n,))
            ### generate n random Hs
            hs = self.dist_hs(uw).rvs(size=(n,))
            ### generate n random Tp given above Hs
            tp = self.dist_tp(hs, uw).rvs(size=(n,))
            res = np.array([uw, hs, tp])
        elif self.ndim ==2:
            hs = self.dist_hs().rvs(size=(n,))
            ### generate n random Tp given above Hs
            tp = self.dist_tp(hs).rvs(size=(n,))
            res = np.array([hs, tp])
        return res

    def support(self):
        return ((0, np.inf),) * self.ndim

    def environment_contour(self, P, T=1000, n=100, q=0.5):
        """
        Return samples for Environment Contours method
        
        arguments:
            P: return period in years
            T: simulation duration in seconds
            n: no. of samples on the contour
            q: fractile for the response variable. q=0.5 corresponds the median response

        Returns:
            ndarray of shape (4, n)
        """
        print(r'Calculating Environment Contour samples for Norway5: {}-D'.format(self.ndim))
        print(r' - {:<25s}: {}'.format('Return period (years)', P))
        print(r' - {:<25s}: {}'.format('Simulation duration (sec)', T))
        print(r' - {:<25s}: {}'.format('Response fractile ', q))
        prob_fail   = 1.0/(P * 365.25*24*3600/T)
        beta        = -stats.norm().ppf(prob_fail) ## reliability index
        r           = np.sqrt(beta**2-stats.norm(0,1).ppf(q)**2)
        print(r' - {:<25s}: {:.2e}'.format('Failure probability', prob_fail))
        print(r' - {:<25s}: {:.2f}'.format('Reliability index', beta))
        print(r' - {:<25s}: {:.2f}'.format('Circle radius', r))
        if self.ndim == 2:
            U = self._create_circle(r, n=n)
        elif self.ndim ==3:
            U = self._create_sphere(r, n=n)
        else:
            raise NotImplementedError

        X = self.ppf(stats.norm().cdf(U)) 
        return U, X

    def target_contour(self, uw, P, T=1000, n=100):
        """
        Return EC points for specified points Uw
        """
        prob_fail   = T/(P * 365.25*24*3600)
        beta        = -stats.norm().ppf(prob_fail) ## reliability index
        u1 = stats.norm().ppf(self.dist_uw().cdf(uw))
        u2 = np.sqrt(beta**2 - u1**2)
        u3 = u2 * 0
        hs = self.dist_hs(uw).ppf(u2)
        tp = self.dist_tp(hs, uw).ppf(u3)
        res = np.array([uw, hs, tp])
        return res


    # ===========================================================  
    # Sequence of conditional distributions based on Rosenblatt transformation 
    # ===========================================================  

    def _create_circle(self, r, n=100):
        """
        return coordinates of points on a 2D circle with radius r

        Parameters:
            r: radius
            n: number of points on circle
        Return:
            ndarray of shape(2,n)

        """
        t = np.linspace(0, np.pi * 2.0, n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        res = np.array([x, y])
        return res 

    def _create_sphere(self, r, n=10):
        lst = []
        for phi in [(pi*i)/(n-1) for i in range(n)]:
            M = int(sin(phi)*(n-1))+1
            for theta in [(2*pi*i)/M for i in range(M)]:
                x = r * sin(phi) * cos(theta)
                y = r * sin(phi) * sin(theta)
                z = r * cos(phi)
                lst.append((x, y, z))
        return np.array(lst).T
