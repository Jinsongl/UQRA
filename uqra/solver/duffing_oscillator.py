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

import uqra
from uqra.solver._solverbase import SolverBase
import os, numpy as np, scipy as sp
# from scipy.integrate import odeint
# from scipy.optimize import brentq
from .PowerSpectrum import PowerSpectrum
from uqra.environment import Kvitebjorn 
from tqdm import tqdm
import multiprocessing as mp

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

    def __init__(self, m=1, c=0.1/np.pi, k=1.0/np.pi/np.pi, s=0.2/np.pi**2, excitation='JONSWAP', environment=None,**kwargs):
        super().__init__()
        self.name       = 'Duffing oscillator'
        self.nickname   = 'Duffing'
        self.dist_name  = 'None'
        self.dict_rand_params = {}
        self.is_param_rand = self._validate_mcks(m,c,k,s)
        self.excitation = excitation
        self.environment= self._validate_env(environment)
        self.ndim       = sum(self.is_param_rand)
        self.nparams    = np.size(self.is_param_rand)

        np.random.seed(100)
        RANDOM_SEEDS    = np.random.randint(0, int(2**31-1), size=10000)

        self.random_states  = kwargs.get('random_state', None )
        if self.random_states is None:
            self.random_states = [None,]
        elif np.ndim(self.random_states) == 0:
            self.random_states = [RANDOM_SEEDS[self.random_states],]
        elif np.ndim(self.random_states) == 1:
            self.random_states = [RANDOM_SEEDS[istate] for istate in self.random_states] 
        else:
            raise ValueError('random_state {} not defined'.format(self.random_states))
        self.n_short_term= len(self.random_states) 

        self.tmax       = kwargs.get('time_max'    , 100  )
        self.tmax       = kwargs.get('tmax'        , 100  )
        self.t_transit  = kwargs.get('t_transit'   , 0    )
        self.dt         = kwargs.get('dt'          , 0.01 )
        self.y0         = kwargs.get('y0'          , [1,0]) ## initial condition
        self.method     = kwargs.get('method'      ,'RK45')
        self.seeds_idx  = kwargs.get('phase'       , [0,] )
        self.t          = np.arange(0, int((self.tmax + self.t_transit)/self.dt) +1) * self.dt
        self.f_hz       = np.arange(len(self.t)+1) *0.5/self.t[-1]
        self.distributions  = kwargs.get('distributions', None)

        # self.out_responses=kwargs.get('out_responses', 'ALL')
        # self.out_stats  = kwargs.get('out_stats'   , ['mean', 'std', 'skewness', 'kurtosis', 'absmax', 'absmin', 'up_crossing'])
    def __str__(self):
        message1 =  '   > Duffing Oscillator: \n'     +\
                    '   - {:<25s} : {}\n'.format('tmax'  , self.tmax) +\
                    '   - {:<25s} : {}\n'.format('dt'    , self.dt) +\
                    '   - {:<25s} : {}\n'.format('y0'    , self.y0) +\
                    '   - {:<25s} : {}\n'.format('n_short_term'  , self.n_short_term) +\
                    '   - {:<25s} : {}\n'.format('method'  , self.method)
                    # +\
                    # '   - {:<25s} : {}\n'.format('out_responses'  , self.out_responses) +\
                    # '   - {:<25s} : {}\n'.format('out_stats'  , self.out_stats) 

        keys   = list(self.dict_rand_params.keys())
        if len(keys) == 0:
            message2 = ''
        else:
            value_names = [] 
            for ivalue in self.dict_rand_params.values():
                try:
                    value_names.append(ivalue.name)
                except AttributeError:
                    value_names.append(ivalue.dist.name)
            message2 = '   - {} : {}\n'.format(keys, value_names)
        message = message1 + message2
        return message

    def run(self, x, verbose=False, **kwargs):
        """
        solving duffing equation:
        Arguments:
            x, power spectrum parameters, ndarray of shape (nsamples, n_parameters)
        """

        random_states = kwargs.get('random_state', self.random_states)
        x = np.array(x.T, copy=False, ndmin=2)
        y = []
        for i, irandom_state in enumerate(random_states):
            if not verbose:
                uqra.blockPrint()

            with mp.Pool(processes=mp.cpu_count()) as p:
                y_QoI_ = np.array(list(tqdm(p.imap(self._duffing_oscillator, [(ix, irandom_state) for ix in x]),
                ncols=80, desc='   - [{:>2d}/{:<2d}: {:>10d}]'.format(i, self.n_short_term, irandom_state), 
                total=x.shape[0])))
            if not verbose: 
                uqra.enablePrint()
            y.append(y_QoI_)

            # pbar_x  = tqdm(x, ascii=True, ncols=80, desc="    - {:d}/{:d} ".format(i, n_short_term))
            # if save_raw:
                # y_raw_, y_QoI_ = map(list, zip(*[self._duffing_oscillator(ix, random_seed=irandom_state, ret_raw=True, 
                # out_responses=out_responses) for ix in pbar_x]))
                # filename = '{:s}_yRaw_nst{:d}'.format(self.nickname,i)
                # np.save(os.path.join(data_dir, filename), np.array(y_raw_))
            # else:
                # y_QoI_ = [self._duffing_oscillator(ix, random_seed=irandom_state, ret_raw=False, 
                # out_responses=out_responses) for ix in pbar_x]

        return np.squeeze(y)

    def generate_samples(self, n, seed=None):
        n = int(n)
        x = []
        np.random.seed(seed)
        ### mck samples 
        i = 0
        for ikey in ['m', 'c', 'k', 's']:
            try:
                x_ = self.dict_rand_params[ikey].rvs(size=(1,n))
                x.append(x_)
                i += 1
            except KeyError:
                x_ = getattr(self, ikey) * np.ones((1,n))
                x.append(x_)

        ### env samples 
        try:
            x_ = self.dict_rand_params['env'].rvs(size=n)
            x_ = np.array(x_, ndmin=2)
            x.append(x_)
        except KeyError:
            pass

        x= np.vstack(x)
        return x 

    def map_domain(self, u, u_cdf):
        """
        Mapping random variables u from distribution dist_u (default U(0,1)) to self.distributions 
        Argument:
            u and dist_u
        """
        if not isinstance(u_cdf, np.ndarray):
            u, dist_u = super().map_domain(u, u_cdf) ## check if dist from stats and change to list [dist,]
            u_cdf     = np.array([idist.cdf(iu) for iu, idist in zip(u, dist_u)])
            u_cdf[u_cdf>0.99999] = 0.99999
            u_cdf[u_cdf<0.00001] = 0.00001

        assert (u_cdf.shape[0] == self.ndim), '{:s} expecting {:d} random variables, {:s} given'.format(self.name, self.ndim, u_cdf.shape[0])
        x = []
        i = 0
        ### maping mck values
        for ikey in ['m', 'c', 'k', 's']:
            try:
                x_ = self.dict_rand_params[ikey].ppf(u_cdf[i])
                x.append(x_)
                i += 1
            except KeyError:
                x_ = getattr(self, ikey) * np.ones((1,u_cdf.shape[1]))
                x.append(x_)

        ### maping env values
        try:
            x_ = self.dict_rand_params['env'].ppf(u_cdf[i:])
            if x_.shape[1] ==1:
                x_ = np.repeat(x_, u_cdf.shape[1], axis=1)
            elif x_.shape[1] == u_cdf.shape[1]:
                pass
            else:
                raise ValueError('map_domain: returned x shape not match u shape: {}!= {}'.format(x.shape, u.shape))
            x.append(x_)
        except KeyError:
            pass
        x = np.vstack(x)
        if x.shape[0] != self.nparams:
            raise ValueError('map_domain: expecting {:d} parameters but only return {:d}'.format(self.nparams, x.shape[0]))
        return x

    def _duffing_oscillator(self, args):
        # x, random_seed=None, ret_raw=False, out_responses='ALL'
        params_env  = args[0]
        random_seed = args[1]
        # if len(params_env) != self.nparams:
            # raise ValueError("_duffing_oscillator: Expecting {:d} parameters but {:d} given".format(self.ndim, len(x)))
        params_mcks = np.array([self.m, self.c, self.k, self.s])
        force_func  = self._external_force_func(x=params_env, random_seed=random_seed)
        x_t         = force_func(self.t)
        t_span      = (0,self.tmax)
        args        = (force_func,params_mcks)
        solution    = sp.integrate.solve_ivp(self._rhs_odes, t_span, self.y0, t_eval=self.t, args=args, method=self.method)

        t_transit_idx = int(self.t_transit/(self.t[1]-self.t[0]))
        t   = self.t[t_transit_idx:]
        x_t = x_t[t_transit_idx:]
        y_t = solution.y[t_transit_idx:]
        return np.max(abs(y_t))

        # y_raw = np.vstack((t, x_t, y_t)).T

        # uqra.blockPrint()
        # y_QoI = uqra.get_stats(y_raw, out_responses=out_responses, out_stats=self.out_stats, axis=0)
        # uqra.enablePrint()
        # if ret_raw:
            # return y_raw, y_QoI
        # else:
            # return y_QoI

    def _rhs_odes(self, t, y, f, mcks):
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
        m,c,k,s = mcks
        y0, y1 = y
        vdot =1.0/m * (-c *y1 - k*y0 - s * y0**3 + f(t))
        return y1, vdot 

    def _external_force_func(self, x=None, random_seed=None):
        """
        Return the excitation function f on the right hand side 

        Returns:
            a function take scalar argument
        """

        if self.excitation is None:
            func = lambda t: t * 0 

        elif isinstance(self.excitation, (int, float, complex)) and not isinstance(self.excitation, bool):
            func = lambda t: t/t * self.excitation 

        elif isinstance(self.excitation, np.ndarray):
            assert np.ndim(self.excitation) == 2
            assert self.excitation.shape[0] == 2
            t, y = self.excitation
            func = sp.interpolate.interp1d(t, y,kind='cubic')

        elif isinstance(self.excitation, str):
            psd_x= PowerSpectrum(self.excitation, *x)
            density = psd_x.cal_density(self.f_hz)
            np.random.seed(random_seed)
            theta_x = np.random.uniform(-np.pi, np.pi, np.size(self.f_hz)*2)
            print('\n')
            print('self.t:', self.t.shape)
            print('self.theta:', theta_x.shape)
            t0, x_t, f_hz_x, theta_x = psd_x.gen_process(t=self.t, phase=theta_x)
            func = sp.interpolate.interp1d(t0, x_t,kind='cubic')
        elif callable(self.excitation):
            func = self.excitation

        else:
            ValueError('Duffing: Excitation function type error: {}'.format(self.excitation))

        return func

    def _validate_mcks(self,m,c,k,s):
        """
        mck vould either be sclars or distributions from scipy.stats
        """
        is_param_rand = []
        if uqra.isfromstats(m):
            self.dict_rand_params['m'] = m
            is_param_rand.append(True)
        elif np.isscalar(m):
            is_param_rand.append(False)
        else:
            raise ValueError('Duffing: mass value type error: {}'.format(m))

        if uqra.isfromstats(c):
            self.dict_rand_params['c'] = c
            is_param_rand.append(True)
        elif np.isscalar(c):
            is_param_rand.append(False)
        else:
            raise ValueError('Duffing: damping value type error: {}'.format(c))

        if uqra.isfromstats(k):
            self.dict_rand_params['k'] = k
            is_param_rand.append(True)
        elif np.isscalar(k):
            is_param_rand.append(False)
        else:
            raise ValueError('Duffing: stiffness value type error: {}'.format(k))

        if uqra.isfromstats(s):
            self.dict_rand_params['s'] = s
            is_param_rand.append(True)
        elif np.isscalar(s):
            is_param_rand.append(False)
        else:
            raise ValueError('Duffing: nonlinear damping value type error: {}'.format(s))
        self.m = m
        self.c = c
        self.k = k
        self.s = s

        return is_param_rand

    def _validate_env(self, env):
        """
        env must be an object of uqra.Environemnt or None
        """
        if env is None:
            self.dist_name = 'None'
        elif isinstance(env, uqra.EnvBase):
            self.dict_rand_params['env'] = env 
            self.dist_name = '_'.join(env.name)
            for is_rand in env.is_arg_rand:
                self.is_param_rand.append(is_rand)
        else:
            raise ValueError('SDOF: environment type {} not defined'.format(type(self.environment)))
        return env 
