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


class duffing_oscillator(SolverBase):
    """
    Solve the Duffing oscillator in time domain solved by Runge-Kutta RK4
        m x'' + c x' + k x + s x^3 = f 
    =>    x'' + 2*zeta*omega_n*x' + omega_n^2*x  + s/m x^3 = 1/m * f
    where, omega_n = sqrt(k/m), zeta = c/(2*sqrt(m*k))
    Default value: 
       - mcks     : [1. 0.01  0.1 0.02]
       - zeta     : 0.01
       - omega_n  : 2 rad/s

    >> Nonlinear spring term: omega_n^2*x  + s/m x^3 
        1. Hardening spring: s > 0
        2. Softening spring: s < 0
        |(s/m) / (omega_n^2)| => |s / k| ~ 0.1, reference Doostan:[0.25, 0.75]
        default: s = 0.2 k, => s/m = 0.2*omega_n^2 = 0.0045 

    f   : frequency in Hz
    dt  : scalar, time step

    args, tuple, oscillator arguments in order of (mass, damping, stiffness) 
    kwargs, dictionary, spectrum definitions for the input excitation functions
    """

    def __init__(self, m=1, c=0.02/np.pi, k=1.0/np.pi/np.pi, s=0.2/np.pi**2, excitation=None, environment=None,**kwargs):
        super().__init__()
        self.name        = 'Duffing oscillator'
        self.nickname    = 'Duffing'
        self.random_params = {}
        self.m          = m
        self.c          = c
        self.k          = k
        self.s          = s
        self.ndim_sys   = self._validate_mcks()
        self.excitation = excitation
        self.environment= environment
        self.ndim_env   = self._validate_env()
        self.ndim       = self.ndim_sys + self.ndim_env

        self.qoi2analysis= kwargs.get('qoi2analysis', 'ALL')
        self.stats2cal   = kwargs.get('stats2cal', ['mean', 'std', 'skewness', 'kurtosis', 'absmax', 'absmin', 'up_crossing'])
        self.tmax        = kwargs.get('time_max', 1000 )
        self.tmax        = kwargs.get('tmax'    , 1000 )
        self.dt          = kwargs.get('dt'      , 0.01 )
        self.y0          = kwargs.get('y0'      , [1,0]) ## initial condition
        self.n_short_term= kwargs.get('n_short_term', 10)  ## number of short term simulations
        self.method      = kwargs.get('method', 'RK45')

    def __str__(self):
        message1 = 'Duffing Oscillator: \n'    
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

    def run(self, x, return_all=False, random_seed=None, **kwargs):
        """
        solving duffing equation:
        Arguments:
            x, power spectrum parameters, ndarray of shape (nsamples, n_parameters)

        """
        n_short_term = kwargs.get('n_short_term', self.n_short_term)
        qoi2analysis = kwargs.get('qoi2analysis', self.qoi2analysis)
        x = np.array(x.T, copy=False, ndmin=2)
        np.random.seed(random_seed)
        seeds = np.random.randint(0, int(2**32-1), size=n_short_term) 
        y_QoI = []
        for ishort_term in range(n_short_term):
            pbar_x  = tqdm(x, ascii=True, ncols=80,desc="    - {:d}/{:d} ".format(ishort_term, self.n_short_term))
            y_raw_, y_QoI_ = map(list, zip(*[self._duffing_oscillator(ix, seed=seeds[ishort_term], qoi2analysis=qoi2analysis) for ix in pbar_x]))
            y_QoI.append(y_QoI_)
            if return_all:
                np.save('{:s}_raw{:d}'.format(self.nickname,ishort_term), np.array(y_raw_))

        return np.array(y_QoI)


    def map_domain(self, u, dist_u):
        """
        Mapping random variables u from distribution dist_u (default U(0,1)) to self.distributions 
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
        for ikey in ['m', 'c', 'k', 's']:
            try:
                x_ = self.random_params[ikey].ppf(u_cdf[i])
                x.append(x_)
                i += 1
            except KeyError:
                x_ = getattr(self, ikey) * np.ones((1,u_cdf.shape[1]))
                x.append(x_)

        ### maping env values
        try:
            x_ = self.random_params['env'].ppf(u_cdf[i:])
            x.append(x_)
        except KeyError:
            pass
        x = np.vstack(x)
        assert x.shape[0] == int(4) + self.ndim_env
        return x

    def _duffing_oscillator(self, x, seed=None, qoi2analysis='ALL'):

        ndim = int(4) + self.ndim_env
        assert len(x) == ndim, "Expecting {:d} variables but {:d} given".format(ndim, len(x))
        t = np.arange(0,int(self.tmax/self.dt) +1) * self.dt
        force_func = self._external_force_func(x, seed=seed)
        x_t = force_func(t)
        solution = sp.integrate.solve_ivp(self._rhs_odes, [0,self.tmax], self.y0, t_eval=t, args=[force_func,], method=self.method)
        y_raw = np.vstack((t, x_t, solution.y)).T

        museuq.blockPrint()
        y_QoI = museuq.get_stats(y_raw, qoi2analysis=qoi2analysis, stats2cal=self.stats2cal, axis=0) 
        museuq.enablePrint()
        return y_raw, y_QoI

    def _rhs_odes(self, t, y, f):
        """
        Reformulate 2nd order ODE to a system of ODEs.
            dy / dt = f(t, y)
        Here t is a scalar, 
        Let u = y, v = y', then:
            u' = v
            v' = -2*zeta*omega_n*v - omega_n^2*u - s/m u^3 + 1/m * f
        Arguments: 
            t: scalar
            y: ndarray has shape (n,), e.g. [y, y']
            f: callable function taken t as argument
        Return:
            (u', v')

        """
        y0, y1 = y
        vdot =1.0/self.m * (-self.c *y1 - self.k*y0 - self.s * y0**3 + f(t))
        return y1, vdot 

    def _external_force_func(self, x, seed=None):
        """
        Return the excitation function f on the right hand side 

        Returns:
            a function take scalar argument

        """

        if self.excitation is not None:
            f = lambda t: t * 0 
            self.ndim_env = 0
        elif isinstance(self.excitation, (int, float, complex)) and not isinstance(x, bool):
            f = lambda t: t/t * self.excitation 
            self.ndim_env = 0

        elif isinstance(self.excitation, np.ndarray):
            assert np.ndim(self.excitation) == 2
            assert self.excitation.shape[0] == 2
            t, y= self.excitation
            f   = sp.interpolate.interp1d(t, y,kind='cubic')
            self.ndim_env= 0
        elif isinstance(self.excitation, str):
            t    = np.arange(0,int(1.10* self.tmax/self.dt)) * self.dt
            tmax = t[-1]
            df   = 0.5/tmax
            freq = np.arange(len(t)+1) * df
            psd_x = PowerSpectrum(self.spec_name, *x)
            x_pxx = psd_x.get_pxx(freq)
            t0, x_t = psd_x.gen_process(seed=seed)
            f = sp.interpolate.interp1d(t0, x_t,kind='cubic')
        elif callable(self.excitation):
            f = self.excitation
        else:
            ValueError('Duffing: Excitation function type error: {}'.format(self.excitation))
        return f

    def _validate_mcks(self):
        """
        mck vould either be sclars or distributions from scipy.stats
        """
        ndim = int(0)

        if museuq.isfromstats(self.m):
            ndim += int(1)
            self.random_params['m'] = self.m
        elif np.isscalar(self.m):
            pass
        else:
            raise ValueError('SDOF: mass value type error: {}'.format(self.m))

        if museuq.isfromstats(self.c):
            ndim += int(1)
            self.random_params['c'] = self.c
        elif np.isscalar(self.c):
            pass
        else:
            raise ValueError('SDOF: damping value type error: {}'.format(self.m))

        if museuq.isfromstats(self.k):
            ndim += int(1)
            self.random_params['k'] = self.k
        elif np.isscalar(self.k):
            pass
        else:
            raise ValueError('SDOF: stiffness value type error: {}'.format(self.m))

        if museuq.isfromstats(self.s):
            ndim += int(1)
            self.random_params['s'] = self.s
        elif np.isscalar(self.s):
            pass
        else:
            raise ValueError('SDOF: nonlinear damping value type error: {}'.format(self.m))

        return ndim

    def _validate_env(self):
        """
        env must be an object of museuq.Environemnt or None
        """
        if self.environment is None:
            ndim = 0

        else:
            assert isinstance(self.environment, museuq.EnvBase)
            ndim = self.environment.ndim
            self.random_params['env'] = self.environment
        return ndim
