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


class linear_oscillator(SolverBase):
    """
    Solving linear oscillator in frequency domain
        m x'' + c x' + k x = f 
    =>    x'' + 2*zeta*omega_n x' + omega_n**2 x = 1/m * f
    where, omega_n = sqrt(k/m), zeta = c/(2*sqrt(m*k))
    default value: omega_n = 1/pi Hz (2 rad/s), zeta = 0.01

    f: frequency in Hz
    t: array, A sequence of time points for which to solve for y. 
    args, tuple, oscillator arguments in order of (mass, damping, stiffness) 
    kwargs, dictionary, spectrum definitions for the input excitation functions
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.name        = 'linaer oscillator'
        self.nickname    = 'SDOF'
        self.spec_name   = kwargs.get('spec_name', 'JONSWAP')
        self.ndim        = PowerSpectrum(self.spec_name).ndim
        self.qoi2analysis= kwargs.get('qoi2analysis', 'ALL')
        self.stats2cal   = kwargs.get('stats2cal', ['mean', 'std', 'skewness', 'kurtosis', 'absmax', 'absmin', 'up_crossing'])
        self.axis        = kwargs.get('axis', 0)
        self.tmax        = kwargs.get('time_max', 1000)
        self.dt          = kwargs.get('dt', 0.1)
        self.distributions= kwargs.get('environment', Kvitebjorn)
        self.dist_name   = self.distributions.__name__.split('.')[-1]
        self.nsim        = kwargs.get('nsim', 10)  ## number of short term simulations
        # self.theta_m = [] 
        # self.theta_s = [] 
        ### two ways defining mck
        if 'k' in kwargs.keys() and 'c' in kwargs.keys():
            self.m      = kwargs.get('m', 1)
            self.k      = kwargs['k'] 
            self.c      = kwargs['c']
            self.zeta   = self.c/(2*np.sqrt(self.m*self.k))
            self.omega_n= 2*np.pi*np.sqrt(self.k/self.m) # rad/s 
        else:
            self.zeta   = kwargs.get('zeta', 0.01)
            self.omega_n= kwargs.get('omega_n', 2) # rad/s
            self.m      = kwargs.get('m', 1)
            self.k      = (self.omega_n/2/np.pi) **2 * self.m
            self.c      = self.zeta * 2 * np.sqrt(self.m * self.k)
        self.mck         = (self.m, self.c, self.k) 

    def __str__(self):
        message = 'Single Degree of Fredom Oscillator: \n'                              + \
                '   - {:<15s} : {}\n'.format('mck'      , np.around(self.mck, 2))       + \
                '   - {:<15s} : {}\n'.format('zeta'     , np.around(self.zeta, 2))      + \
                '   - {:<15s} : {}\n'.format('omega_n'  , np.around(self.omega_n, 2))   + \
                '   - {:<15s} : {}\n'.format('spec_name', self.spec_name)               + \
                '   - {:<15s} : {}\n'.format('qoi2analysis', self.qoi2analysis)         + \
                '   - {:<15s} : {}\n'.format('time_max' , self.tmax)                    + \
                '   - {:<15s} : {}\n'.format('dt' , self.dt)
        return message

    def run(self, x, return_all=False, random_seed=None, **kwargs):
        """
        run linear_oscillator:
        Arguments:
            x, power spectrum parameters, ndarray of shape (n_parameters, nsamples)

        """
        np.random.seed(random_seed)
        nsim = kwargs.get('nsim', self.nsim)
        x = np.array(x, copy=False, ndmin=2).T
        # x = x.reshape(-1,1) if x.ndim == 1 else x
        ## if x is just one set of input of shape (2, 1)
        y_raw = []
        y_QoI = []
        seeds = np.random.randint(0, int(2**32-1), size=nsim) 
        for insim in range(nsim):
        # Note that xlist and ylist will be tuples (since zip will be unpacked). If you want them to be lists, you can for instance use:
            pbar_x  = tqdm(x, ascii=True, desc="    - {:d}/{:d} ".format(insim, self.nsim))
            y_raw_, y_QoI_ = map(list, zip(*[self._linear_oscillator(ix, seed=seeds[insim]) for ix in pbar_x]))
            y_raw.append(y_raw_)
            y_QoI.append(y_QoI_)
        if return_all:
            return np.array(y_raw), np.array(y_QoI)
        else:
            return np.array(y_QoI)

    def x_psd(self, f, x, **kwargs):
        """
        Return the power spectral density (PSD) estimate, pxx, at frequency,f, for the given PowerSpectrum with given parameters x

        Returns:
            PowerSpectrum object of input signal
        """
        spec_name = kwargs.get('spec_name', self.spec_name)
        psd_x = PowerSpectrum(spec_name, *x)
        x_pxx = psd_x.get_pxx(f)
        return psd_x

    def psd(self,f,x):
        """
        Return the psd estimator of both input and output signals at frequency f for specified PowerSpectrum with given parameters x

        Arguments:
            f: frequency in Hz
            x: PowerSpectrum parameters
        Returns:
            PowerSpectrum object of input and output signal
        """
        H_square = 1.0/np.sqrt( (self.k-self.m*f**2)**2 + (self.c*f)**2 )
        psd_x = self.x_psd(f, x)
        y_pxx = H_square * psd_x.pxx
        psd_y = PowerSpectrum('SDOF')
        psd_y.set_psd(psd_x.f, y_pxx)

        return psd_x, psd_y 

    def _linear_oscillator(self, x, seed=None):
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

        t    = np.arange(0,int(self.tmax/self.dt) +1) * self.dt
        tmax = t[-1]
        df   = 0.5/tmax
        f    = np.arange(len(t)+1) * df
        ##--------- oscillator properties -----------
        psd_x, psd_y = self.psd(f, x)
        t0, x_t = psd_x.gen_process(seed=seed)
        t1, y_t = psd_y.gen_process(seed=seed)
        assert (t0==t1).all()

        y_raw = np.vstack((t0, x_t, y_t)).T
        museuq.blockPrint()
        y_QoI = museuq.get_stats(y_raw, qoi2analysis=self.qoi2analysis, stats2cal=self.stats2cal, axis=0) 
        museuq.enablePrint()
        return y_raw, y_QoI
            
    def map_domain(self, u, u_cdf):
        """
        mapping random variables u from distribution dist_u (default U(0,1)) to self.distributions 
        Argument:
            u and dist_u
        """

        if isinstance(u_cdf, np.ndarray):
            assert (u_cdf.shape[0] == self.ndim), '{:s} expecting {:d} random variables, {:s} given'.format(self.name, self.ndim, u_cdf.shape[0])
            if self.distributions.__name__ == 'museuq.environment.Kvitebjorn':
                x = Kvitebjorn.ppf(u_cdf)
            else:
                raise ValueError('Distribution name not defined: {:s}'.format(self.distributions.__name__))
        else:
            u, dist_u = super().map_domain(u, u_cdf) 
            u_cdf     = np.array([idist.cdf(iu) for iu, idist in zip(u, dist_u)])

            if self.distributions.__name__ == 'museuq.environment.Kvitebjorn':
                x = Kvitebjorn.ppf(u_cdf)
            else:
                raise ValueError('Distribution name not defined: {:s}'.format(self.distributions.__name__))
        return x
        return x


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

    def __init__(self, **kwargs):
        super().__init__()
        self.name        = 'Duffing oscillator'
        self.nickname    = 'Duffing'
        self.spec_name   = kwargs.get('spec_name', None)
        self.ndim        = PowerSpectrum(self.spec_name).ndim
        self.excitation  = kwargs.get('excitation', None)
        self.qoi2analysis= kwargs.get('qoi2analysis', 'ALL')
        self.stats2cal   = kwargs.get('stats2cal', ['mean', 'std', 'skewness', 'kurtosis', 'absmax', 'absmin', 'up_crossing'])
        self.axis        = kwargs.get('axis', 0)
        self.tmax        = kwargs.get('time_max', 1000)
        self.tmax        = kwargs.get('tmax', 1000)
        self.dt          = kwargs.get('dt', 0.1)
        self.y0          = kwargs.get('y0', [1,0]) ## initial condition
        self.distributions= kwargs.get('environment', Kvitebjorn)
        self.method      = kwargs.get('method', 'RK45')

        ### two ways defining mcks
        if 'k' in kwargs.keys() and 'c' in kwargs.keys():
            self.m      = kwargs.get('m', 1)
            self.k      = kwargs['k'] 
            self.c      = kwargs['c']
            self.s      = kwargs.get('s', 0.2* self.k)
            # self.zeta   = self.c/(2*np.sqrt(self.m*self.k))
            # self.omega_n= 2*np.pi*np.sqrt(self.k/self.m) # rad/s 
        else:
            self.zeta   = kwargs.get('zeta', 0.01)
            self.omega_n= kwargs.get('omega_n', 2) # rad/s
            self.m      = kwargs.get('m', 1)
            self.k      = (self.omega_n/2/np.pi) **2 * self.m
            self.c      = self.zeta * 2 * np.sqrt(self.m * self.k)
            self.s      = kwargs.get('s', 0.2 * self.k)
        self.mcks       = (self.m, self.c, self.k, self.s) 

    def __str__(self):
        message = 'Duffing Oscillator: \n'                                              + \
                '   - {:<15s} : {}\n'.format('mcks'     , np.around(self.mcks, 2))      + \
                '   - {:<15s} : {}\n'.format('excitation', self.excitation.__name__ if self.excitation else self.spec_name) + \
                '   - {:<15s} : {}\n'.format('qoi2analysis', self.qoi2analysis)         + \
                '   - {:<15s} : {}\n'.format('time_max' , self.tmax)                    + \
                '   - {:<15s} : {}\n'.format('dt' , self.dt)
        return message

    def run(self, x):
        """
        solving duffing equation:
        Arguments:
            x, power spectrum parameters, ndarray of shape (nsamples, n_parameters)

        """
        x = np.array(x, copy=False, ndmin=2)
        pbar_x  = tqdm(x, ascii=True, desc="   - ")
        y_raw, y_QoI = map(list, zip(*[self._duffing_oscillator(ix) for ix in pbar_x]))

        return np.array(y_raw), np.array(y_QoI)


    def map_domain(self, u, dist_u):
        """
        Mapping random variables u from distribution dist_u (default U(0,1)) to self.distributions 
        Argument:
            u and dist_u
        """
        ### convert dist_u to list and append to dist_u with U(0,1) if necessary to make sure the dimension matches
        u, dist_u = super().map_domain(u, dist_u)
        u_cdf     = np.array([idist.cdf(iu) for iu, idist in zip(u, dist_u)])
        assert (u_cdf.shape[0] == self.ndim), '{:s} expecting {:d} random variables, {:d} given'.format(self.name, self.ndim, u_cdf.shape[0])

        if isinstance(self.distributions, list):
            x = np.array([idist.ppf(iu_cdf)  for iu_cdf, idist in zip(u_cdf, self.distributions)])
        elif self.distributions.__name__ == 'museuq.environment.Kvitebjorn':
            x = Kvitebjorn.ppf(u_cdf)
        return x


    def _duffing_oscillator(self, x):

        t = np.arange(0,int(self.tmax/self.dt) +1) * self.dt
        self.excitation = self._excitation_func(x)
        x_t = self.excitation(t)
        solution = sp.integrate.solve_ivp(self._rhs_odes, [0,self.tmax], self.y0, t_eval=t, args=[self.excitation],method=self.method)
        y_raw = np.vstack((t, x_t, solution.y)).T

        museuq.blockPrint()
        y_QoI = museuq.get_stats(y_raw, qoi2analysis=self.qoi2analysis, stats2cal=self.stats2cal, axis=0) 
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


        # V = lambda x: beta/4 * x**4 - alpha/2 * x**2 
        # dVdx = lambda x: beta*x**3 - alpha*x

        # x, xdot = X
        # if source_interp is None:
            # xdotdot = -dVdx(x) -delta * xdot + gamma * np.cos(omega*t) 
        # else:
            # xdotdot = -dVdx(x) -delta * xdot + gamma * np.cos(omega*t) + source_interp(t)
        # return xdot, xdotdot

    def _excitation_func(self, x):
        """
        Return the excitation function f on the right hand side 

        Returns:
            a function take scalar argument

        """

        if self.excitation is not None:
            f = self.excitation
        else:
            if self.spec_name:
                t    = np.arange(0,int(1.10* self.tmax/self.dt)) * self.dt
                tmax = t[-1]
                df   = 0.5/tmax
                freq    = np.arange(len(t)+1) * df
                psd_x = PowerSpectrum(self.spec_name, *x)
                x_pxx = psd_x.get_pxx(freq)
                t0, x_t = psd_x.gen_process()
                f = sp.interpolate.interp1d(t0, x_t,kind='cubic')
            else:
                f = lambda t: t * 0 
        return f

# def _cal_normalize_values(zeta,omega0,source_kwargs, *source_args):
    # TF = lambda w : 1.0/np.sqrt((w**2-omega0**2)**2 + (2*zeta*omega0)**2)
    # spec_dict = psd.get_spec_dict() 
    # spec_name = source_kwargs.get('name','JONSWAP') if source_kwargs else 'JONSWAP'
    # spec_side = source_kwargs.get('sides', '1side') if source_kwargs else '1side'
    # spec_func = spec_dict[spec_name]

    # nquads = 100
    # if spec_side.upper() in ['2','2SIDE','DOUBLE','2SIDES']:
        # x, w = np.polynomial.hermite_e.hermegauss(nquads)
    # elif spec_side.upper() in ['1','1SIDE','SINGLE','1SIDES']:
        # x, w = np.polynomial.laguerre.laggauss(nquads)
    # else:
        # raise NotImplementedError("Spectrum side type '{:s}' is not defined".format(spec_side))
    # _,spec_vals = spec_func(x, *source_args)
    # spec_vals = spec_vals.reshape(nquads,1)
    # TF2_vals  = (TF(x)**2).reshape(nquads,1)
    # w = w.reshape(nquads,1)
    # norm_y = np.sum(w.T *(spec_vals*TF2_vals))/(2*np.pi)
    # norm_y = 1/norm_y**0.5
    # norm_t = omega0

    # return norm_t, norm_y


# def _normalize_source_func(source_func, norm_t, norm_y):
    # def wrapper(*args, **kwargs):
        # t, y = source_func(*args, **kwargs)
        # return  t*norm_t, y*norm_y/ norm_t**2
    # return wrapper 
        
# def duffing_oscillator(tmax,dt,x0,v0,zeta,omega0,mu,\
        # *source_args, source_func=None, source_kwargs=None,t_trans=0, normalize=False):
    # if normalize:
        # # norm_t, norm_y = normalize[1], normalize[2]
        # norm_t, norm_y = _cal_normalize_values(zeta, omega0, source_kwargs) 
        # # print('Normalizing value: [{:.2f}, {:.2f}]'.format(norm_t, norm_y))
        # assert norm_t!= 0
        # delta = 2 * zeta * omega0 / norm_t
        # alpha = omega0**2 / norm_t**2
        # beta = mu*omega0**2/(norm_y**2 * norm_t**2)
        # # print('delta:{:.2f}, alpha: {:.2f}, beta: {:.2f}'.format(delta, alpha, beta))
        # dt_per_period = int(2*np.pi/omega0/dt)
        # tmax = norm_t*tmax
        # dt = norm_t* dt
        # source_func = _normalize_source_func(source_func, norm_t, norm_y) if source_func else source_func
        # source_args = source_args/omega0 if source_args else source_args
        # gamma,omega = 0,1 # gamma ==0 with arbitrary omega
        # t, X, dt, pstep = duffing_equation(tmax,dt_per_period,x0,v0,gamma,delta,omega,\
                # *source_args, source_func=source_func, source_kwargs=source_kwargs,\
                # t_trans=t_trans, alpha=alpha, beta=beta)
    # else:
        # delta = 2 * zeta * omega0
        # alpha = omega0**2
        # beta = omega0**2 * mu
        # dt_per_period = int(2*np.pi/omega0/dt)
        # gamma,omega = 0,1 # gamma ==0 with arbitrary omega
        # t, X, dt, pstep = duffing_equation(tmax,dt_per_period,x0,v0,gamma,delta,omega,\
                # *source_args, source_func=source_func, source_kwargs=source_kwargs,\
                # t_trans=t_trans, alpha=alpha, beta=beta)

    # t = np.reshape(t,(len(t),1))
    # res = np.concatenate((t,X),axis=1)
    # return res, dt, pstep 


# def duffing_equation(tmax, dt_per_period, x0, v0,gamma,delta,omega,\
        # *source_args, source_func=None, source_kwargs=None,alpha=1,beta=1,t_trans=0):
    # """
    # Solve the Duffing equation for parameters gamma, delta, omega.
    # https://scipython.com/blog/the-duffing-oscillator/
    # Find the numerical solution to the Duffing equation using a suitable
    # time grid: 
        # - tmax is the maximum time (s) to integrate to; 
        # - t_trans is the initial time period of transient behaviour until the solution settles down (if it does) to some kind of periodic motion (these data
    # points are dropped) and 
        # - dt_per_period is the number of time samples (of duration dt) to include per period of the driving motion (frequency omega).

    # x'' + delta x' + alpha x + beta x^3 = gamma * cos(omega t)
    # x(0) = x'(0)

    # Returns the time grid, t (after t_trans), position, x, and velocity,
    # xdot, dt, and step, the number of array points per period of the driving
    # motion.

    # """
    # # Time point spacings and the time grid

    # period = 2*np.pi/omega
    # dt = int(2*np.pi/omega / dt_per_period * 1000)/1000
    # step = int(period / dt)
    # t = np.arange(0, tmax, dt)
    # # Initial conditions: x, xdot
    # X0 = [x0, v0]
    # if callable(source_func):
        # _t = np.arange(0, tmax+period, dt)
        # _, source = source_func(_t, *source_args, kwargs=source_kwargs)
        # source_interp = sp.interpolate.interp1d(_t, source,kind='cubic')

    # else:
        # source_interp = None

    # X = odeint(_deriv, X0, t, args=(gamma, delta, omega,alpha, beta, source_interp))
    # idx = int(t_trans / dt)
    # return t[idx:], X[idx:], dt, step

# def lin_oscillator(tmax,dt,x0,v0,zeta,omega0,source_func=None,t_trans=0, *source_args):
    # source_func = source_func if callable(source_func) else 0
    # x = duffing_oscillator(tmax,dt,x0,v0,zeta,omega0,0,source_func=source_func,t_trans=t_trans)
    # return x

