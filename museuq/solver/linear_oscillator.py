#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""
Single degree of freedom with time series external loads
"""

import museuq
from museuq.solver._solverbase import SolverBase
import os, numpy as np, scipy as sp
# from scipy.integrate import odeint
# from scipy.optimize import brentq
from .PowerSpectrum import PowerSpectrum
from museuq.environment import Kvitebjorn 
from tqdm import tqdm
import scipy.stats as stats


class linear_oscillator(SolverBase):
    """
    Solving linear oscillator in frequency domain
        m x'' + c x' + k x = f 
    =>    x'' + 2*zeta*omega_n x' + omega_n**2 x = 1/m * f
    where, omega_n = sqrt(k/m), zeta = c/(2*sqrt(m*k))
    default value: omega_n = 1/pi Hz (2 rad/s), zeta = 0.05

    """
    def __init__(self, m=1, c=0.1/np.pi, k=1.0/np.pi/np.pi, excitation=None, environment=None,**kwargs):
        """
        excitation:
            1. None: free vibration case, equivalent to set excitation=0
            2. np.scalar: constant external force
            3. np.array of shape (2,): external force time series,[ t, f = excitation ], may need to interpolate
            4. string   : name of spectrum to be used
            5. callable() function
            
        f0: part of external force
        kwargs, dictionary, spectrum definitions for the input excitation functions

        """
        super().__init__()
        self.name       = 'linear oscillator'
        self.nickname   = 'SDOF'
        self.random_params = {}
        self.m          = m
        self.c          = c
        self.k          = k
        self.params_is_rand = []
        self.ndim_sys   = self._validate_mck()
        self.excitation = excitation
        self.environment= environment
        self.ndim_env   = self._validate_env()
        self.ndim       = self.ndim_sys + self.ndim_env
        assert self.ndim == sum(self.params_is_rand)
        self.nparams    = np.size(self.params_is_rand)
        self.dist_name  = 'None'

        self.tmax       = kwargs.get('time_max', 100)
        self.tmax       = kwargs.get('tmax', 100)
        self.dt         = kwargs.get('dt', 0.01)
        self.t_transit  = kwargs.get('t_transit', 0)
        self.out_stats  = kwargs.get('out_stats', ['mean', 'std', 'skewness', 'kurtosis', 'absmax', 'absmin', 'up_crossing'])
        self.seeds_st   = kwargs.get('seeds_st', [100,])
        self.n_short_term= len(self.seeds_st) 
        self.out_responses= kwargs.get('out_responses', 'ALL')
        self.t          = np.arange(0,int((self.tmax + self.t_transit)/self.dt) +1) * self.dt
        self.f_hz       = np.arange(len(self.t)+1) *0.5/self.t[-1]

    def __str__(self):
        message1 = 'Single Degree of Fredom Oscillator: \n'
        keys   = list(self.random_params.keys())
        value_names = [] 
        for ivalue in self.random_params.values():
            try:
                value_names.append(ivalue.name)
            except AttributeError:
                value_names.append(ivalue.dist.name)
        message2 = '   - {} : {}\n'.format(keys, value_names)
        message = message1 + message2
        return message

    def run(self, x, save_raw=False, seeds_st=None, save_qoi=False,  **kwargs):
        """
        run linear_oscillator:
        Arguments:
            x, power spectrum parameters, ndarray of shape (n_parameters, nsamples)

        """
        seeds_st        = kwargs.get('seeds_st', self.seeds_st)
        n_short_term    = len(seeds_st) 
        out_responses   = kwargs.get('out_responses', self.out_responses)
        data_dir        = kwargs.get('data_dir', os.getcwd())
        x = np.array(x.T, copy=False, ndmin=2)
        y_QoI = []
        for ishort_term in range(n_short_term):
            pbar_x  = tqdm(x, ascii=True, ncols=80, desc="    - {:d}/{:d} ".format(ishort_term, n_short_term))
            ### Note that xlist and ylist will be tuples (since zip will be unpacked). 
            ### If you want them to be lists, you can for instance use:
            if save_raw:
                y_raw_, y_QoI_ = map(list, zip(*[self._linear_oscillator(ix, random_seed=seeds_st[ishort_term], out_responses=out_responses, ret_raw=True) for ix in pbar_x]))
                filename = '{:s}_yRaw_R{:d}'.format(self.nickname,ishort_term)
                np.save(os.path.join(data_dir, filename), np.array(y_raw_))
            else:
                y_QoI_ = [self._linear_oscillator(ix, random_seed=seeds_st[ishort_term], out_responses=out_responses, ret_raw=False) for ix in pbar_x]

            if save_qoi:
                filename = '{:s}_yQoI_R{:d}'.format(self.nickname,ishort_term)
                np.save(os.path.join(data_dir, filename), np.array(y_QoI_))
                y_QoI=[]

            y_QoI.append(y_QoI_)
        y_QoI = np.array(y_QoI)
        return y_QoI

    def _linear_oscillator(self, x, random_seed=None, ret_raw=False, out_responses='ALL'):
        """
        Solving linear oscillator in frequency domain
        m x'' + c x' + k x = f => 
        x'' + 2*zeta*omega_n x' + omega_n**2 x = 1/m f, where, omega_n = sqrt(k/m), zeta = c/(2*sqrt(m*k))
        default value: omega_n = 0.15 Hz, zeta = 0.01

        f: frequency in Hz
        t: array, A sequence of time points for which to solve for y. 
        args, tuple, oscillator arguments in order of (mass, damping, stiffness) 
        kwargs, dictionary, spectrum definitions for the input excitation functions
        """
        ndim = int(3) + self.ndim_env
        assert len(x) == ndim, "Expecting {:d} variables but {:d} given".format(ndim, len(x))
        ##--------- oscillator properties -----------
        params_mck, params_env = x[:3], x[3:]
        m,c,k = params_mck 
        spectrum_x = self.external_force_psd(params_env, self.f_hz)
        spectrum_y = self.response_psd(params_mck, spectrum_x)

        np.random.seed(random_seed)
        theta_x = np.random.uniform(-np.pi, np.pi, np.size(self.f_hz)*2)
        t0, x_t, f_hz_x, theta_x = spectrum_x.gen_process(t=self.t, phase=theta_x)
        omega   = 2*np.pi*f_hz_x
        A, B    = k - m * omega**2 , c * omega
        theta_y =  np.arctan(-(A * np.tan(theta_x) + B) / (A - B * np.tan(theta_x)))
        t1, y_t, f_hz_y, theta_y = spectrum_y.gen_process(t=self.t, phase=theta_y)
        assert np.array_equal(t0,t1)
        assert np.array_equal(f_hz_x,f_hz_y)
        assert np.array_equal(self.t,t1)
        t_transit_idx = int(self.t_transit/(t0[1]-t0[0]))
        t0 = t0[t_transit_idx:]
        x_t = x_t[t_transit_idx:]
        y_t = y_t[t_transit_idx:]

        y_raw = np.vstack((t0, x_t, y_t)).T
        museuq.blockPrint()
        y_QoI = museuq.get_stats(y_raw, out_responses=out_responses, out_stats=self.out_stats, axis=0) 
        museuq.enablePrint()
        if ret_raw:
            return y_raw, y_QoI
        else:
            return y_QoI
            
    def map_domain(self, u, u_cdf, only_rand=False):
        """
        mapping random variables u from distribution dist_u (default U(0,1)) to self.distributions 
        Argument:
            u and dist_u
        """

        if not isinstance(u_cdf, np.ndarray):
            u, dist_u = super().map_domain(u, u_cdf) ## check if dist from stats and change to list [dist,]
            u_cdf     = np.array([idist.cdf(iu) for iu, idist in zip(u, dist_u)])

        assert (u_cdf.shape[0] == self.ndim), '{:s} expecting {:d} random variables, {:s} given'.format(self.name, self.ndim, u_cdf.shape[0])
        x = []
        i = 0
        ### maping mck values
        for ikey in ['m', 'c', 'k']:
            try:
                x_ = self.random_params[ikey].ppf(u_cdf[i])
                x.append(x_)
                i += 1
            except KeyError:
                x_ = getattr(self, ikey) * np.ones((1,u_cdf.shape[1]))
                x.append(x_)
        ### maping env values
        if len(u_cdf[i:]) != 0:
            try:
                x_ = self.random_params['env'].ppf(u_cdf[i:])
                x.append(x_)
            except KeyError:
                pass
        else:
            pass
        x = np.vstack(x)
        assert x.shape[0] == int(3) + self.ndim_env
        return x

    # def random_params_mean

    def _validate_mck(self):
        """
        mck vould either be sclars or distributions from scipy.stats
        """
        ndim = int(0)

        if museuq.isfromstats(self.m):
            ndim += int(1)
            self.random_params['m'] = self.m
            self.params_is_rand.append(True)
        elif np.isscalar(self.m):
            self.params_is_rand.append(False)
        else:
            raise ValueError('SDOF: mass value type error: {}'.format(self.m))

        if museuq.isfromstats(self.c):
            ndim += int(1)
            self.random_params['c'] = self.c
            self.params_is_rand.append(True)
        elif np.isscalar(self.c):
            self.params_is_rand.append(False)
        else:
            raise ValueError('SDOF: dampling value type error: {}'.format(self.m))

        if museuq.isfromstats(self.k):
            ndim += int(1)
            self.random_params['k'] = self.k
            self.params_is_rand.append(True)
        elif np.isscalar(self.k):
            self.params_is_rand.append(False)
        else:
            raise ValueError('SDOF: stiffness value type error: {}'.format(self.m))

        return ndim

    def _validate_env(self):
        """
        env must be an object of museuq.Environemnt or None
        """
        # self.distributions= kwargs.get('environment', Kvitebjorn)
        # self.dist_name   = self.distributions.__name__.split('.')[-1]
        if self.environment is None:
            ndim = 0
            self.dist_name = 'None'
        elif isinstance(self.environment, museuq.EnvBase):
            ndim = self.environment.ndim
            self.random_params['env'] = self.environment
            self.dist_name = '_'.join(self.environment.name)
            for iname in self.environment.name:
                self.params_is_rand.append(iname.lower()!='const')
        else:
            raise ValueError('SDOF: environment type {} not defined'.format(type(self.environment)))
        return ndim

    def generate_samples(self, n, random_seed=None):
        n = int(n)
        x = []
        np.random.seed(random_seed)
        ### mck samples 
        i = 0
        for ikey in ['m', 'c', 'k']:
            try:
                x_ = self.random_params[ikey].rvs(size=(1,n))
                x.append(x_)
                i += 1
            except KeyError:
                x_ = getattr(self, ikey) * np.ones((1,n))
                x.append(x_)

        ### env samples 
        try:
            x_ = self.random_params['env'].rvs(size=n)
            x_ = np.array(x_, ndmin=2)
            x.append(x_)
        except KeyError:
            pass

        x= np.vstack(x)
        return x 

    def external_force_psd(self, x, f_hz):
        """
        Return the psd estimator of excitation function on the right hand side 
        parameters:
            x: power spectrum parameters
            f: frequency in Hz
            
        Returns:
            a function take scalar argument

        """


        if self.excitation is None:
            f = lambda t: t * 0 
            self.ndim_env = 0
            raise ValueError('SDOF is sovled in frequency domain, None type external force need to be transformed into frequency domain first') 

        elif isinstance(self.excitation, (int, float, complex)) and not isinstance(x, bool):
            f = lambda t: t/t * self.excitation 
            self.ndim_env = 0
            raise ValueError('SDOF is sovled in frequency domain, constant external force need to be transformed into frequency domain first') 

        elif isinstance(self.excitation, np.ndarray):
            assert np.ndim(self.excitation) == 2
            assert self.excitation.shape[0] == 2
            t, y= self.excitation
            f   = sp.interpolate.interp1d(t, y,kind='cubic')
            self.ndim_env= 0
            raise ValueError('SDOF is sovled in frequency domain, time series external force need to be transformed into frequency domain first') 

        elif isinstance(self.excitation, str):
            spectrum= PowerSpectrum(self.excitation, *x)
            density = spectrum.cal_density(f_hz)

        else:
            raise NotImplementedError

        return spectrum

    def response_psd(self, mck, spectrum_x):
        """
        Return the psd estimator of both input and output signals at frequency f for specified PowerSpectrum with given parameters x

        Arguments:
            f: frequency in Hz
            x: PowerSpectrum parameters
        Returns:
            PowerSpectrum object of input and output signal
        """
        m,c,k = mck
        f_hz  = spectrum_x.f_hz  ## f_hz in Hz
        H_square = 1.0/np.sqrt( (k-m*f_hz**2)**2 + (c*f_hz)**2)
        psd_y = H_square * spectrum_x.pxx
        spectrum_y = PowerSpectrum()
        spectrum_y.set_density(spectrum_x.f_hz, psd_y, sides=spectrum_x.sides)
        return spectrum_y 
