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

import os, numpy as np
from scipy import interpolate
from scipy.integrate import odeint, quad
from scipy.optimize import brentq
from ..utilities.PowerSpectrum import PowerSpectrum
from ..utilities import power_spectrum as psd
from ..utilities.psd2process import psd2process
from ..utilities.helpers import get_stats
data_dir = '/Users/jinsongliu/BoxSync/MUSELab/museuq/examples/JupyterNotebook'

def _cal_normalize_values(zeta,omega0,source_kwargs, *source_args):
    TF = lambda w : 1.0/np.sqrt((w**2-omega0**2)**2 + (2*zeta*omega0)**2)
    spec_dict = psd.get_spec_dict() 
    spec_name = source_kwargs.get('name','JONSWAP') if source_kwargs else 'JONSWAP'
    spec_side = source_kwargs.get('sides', '1side') if source_kwargs else '1side'
    spec_func = spec_dict[spec_name]

    nquads = 100
    if spec_side.upper() in ['2','2SIDE','DOUBLE','2SIDES']:
        x, w = np.polynomial.hermite_e.hermegauss(nquads)
    elif spec_side.upper() in ['1','1SIDE','SINGLE','1SIDES']:
        x, w = np.polynomial.laguerre.laggauss(nquads)
    else:
        raise NotImplementedError("Spectrum side type '{:s}' is not defined".format(spec_side))
    _,spec_vals = spec_func(x, *source_args)
    spec_vals = spec_vals.reshape(nquads,1)
    TF2_vals  = (TF(x)**2).reshape(nquads,1)
    w = w.reshape(nquads,1)
    norm_y = np.sum(w.T *(spec_vals*TF2_vals))/(2*np.pi)
    norm_y = 1/norm_y**0.5
    norm_t = omega0

    return norm_t, norm_y


def _normalize_source_func(source_func, norm_t, norm_y):
    def wrapper(*args, **kwargs):
        t, y = source_func(*args, **kwargs)
        return  t*norm_t, y*norm_y/ norm_t**2
    return wrapper 

def lin_oscillator(tmax,dt,x0,v0,zeta,omega0,source_func=None,t_trans=0, *source_args):
    source_func = source_func if callable(source_func) else 0
    x = duffing_oscillator(tmax,dt,x0,v0,zeta,omega0,0,source_func=source_func,t_trans=t_trans)
    return x

def linear_oscillator(t, x, **kwargs):
    """
    Solving linear oscillator in frequency domain
    m x'' + c x' + k x = f => 
    x'' + 2*zeta*w_n x' + w_n**2 x = 1/m f, where, w_n = sqrt(k/m), zeta = c/(2*sqrt(m*k))
    default value: omega_n = 0.15 Hz, zeta = 0.01

    f: frequency in Hz
    t: array, A sequence of time points for which to solve for y. 
    args, tuple, oscillator arguments in order of (mass, damping, stiffness) 
    kwargs, dictionary, spectrum definitions for the input excitation functions
    """

    # spec_dict = psd.get_spec_dict() 
    # spec_func = spec_dict.get(kwargs.get('spec_name', 'JONSWAP'))
    # f,x_pxx   = spec_func(f, *x)
    spec_name   = kwargs.get('spec_name', 'JONSWAP')
    return_all  = kwargs.get('return_all', False)
    m,c,k       = kwargs.get('mck', (100, 0.3, 2.25))


    tmax, dt    = t[-1], t[1] - t[0]
    df          = 0.5/tmax
    f           = np.arange(len(t)+1) * df
    ##--------- oscillator properties -----------
    w_n         = np.sqrt(k/m)          # natural frequency
    zeta        = c/(2*np.sqrt(m*k))    # damping ratio
    H_square    = 1.0/np.sqrt( (k-m*f**2)**2 + (c*f)**2 )

    input_psd   = PowerSpectrum(spec_name, *x)
    x_pxx       = input_psd.get_pxx(f)
    t, x_t      = input_psd.gen_process()
    output_psd  = PowerSpectrum('SDOF_out')
    y_pxx       = H_square * x_pxx
    output_psd.set_psd(f, y_pxx)
    t, y_t      = output_psd.gen_process()

    # t, x_t = psd2process(f, x_pxx)
    # t, y_t = psd2process(f, y_pxx)

    ##---- Reshape-----
    t   = np.array(t).reshape((-1,1))
    x_t = np.array(x_t).reshape((-1,1))
    y_t = np.array(y_t).reshape((-1,1))
    y1  = np.concatenate((t,x_t,y_t), axis=1)
    # np.save(os.path.join(data_dir, 'sdof_time'), y1)
    f   = np.array(f).reshape((-1,1))
    x_pxx=np.array(x_pxx).reshape((-1,1))
    y_pxx=np.array(y_pxx).reshape((-1,1))
    y2  = np.concatenate((f,x_pxx,y_pxx), axis=1)
    # np.save(os.path.join(data_dir, 'sdof_frequency'), y2)

    y  = np.concatenate((t,x_t,y_t, f,x_pxx,y_pxx), axis=1)
    if return_all:
        return y
    else:
        y_stats = get_stats(np.concatenate((x_t, y_t), axis=1)) 
        return y_stats
        
def duffing_oscillator(tmax,dt,x0,v0,zeta,omega0,mu,\
        *source_args, source_func=None, source_kwargs=None,t_trans=0, normalize=False):
    if normalize:
        # norm_t, norm_y = normalize[1], normalize[2]
        norm_t, norm_y = _cal_normalize_values(zeta, omega0, source_kwargs) 
        # print('Normalizing value: [{:.2f}, {:.2f}]'.format(norm_t, norm_y))
        assert norm_t!= 0
        delta = 2 * zeta * omega0 / norm_t
        alpha = omega0**2 / norm_t**2
        beta = mu*omega0**2/(norm_y**2 * norm_t**2)
        # print('delta:{:.2f}, alpha: {:.2f}, beta: {:.2f}'.format(delta, alpha, beta))
        dt_per_period = int(2*np.pi/omega0/dt)
        tmax = norm_t*tmax
        dt = norm_t* dt
        source_func = _normalize_source_func(source_func, norm_t, norm_y) if source_func else source_func
        source_args = source_args/omega0 if source_args else source_args
        gamma,omega = 0,1 # gamma ==0 with arbitrary omega
        t, X, dt, pstep = duffing_equation(tmax,dt_per_period,x0,v0,gamma,delta,omega,\
                *source_args, source_func=source_func, source_kwargs=source_kwargs,\
                t_trans=t_trans, alpha=alpha, beta=beta)
    else:
        delta = 2 * zeta * omega0
        alpha = omega0**2
        beta = omega0**2 * mu
        dt_per_period = int(2*np.pi/omega0/dt)
        gamma,omega = 0,1 # gamma ==0 with arbitrary omega
        t, X, dt, pstep = duffing_equation(tmax,dt_per_period,x0,v0,gamma,delta,omega,\
                *source_args, source_func=source_func, source_kwargs=source_kwargs,\
                t_trans=t_trans, alpha=alpha, beta=beta)

    t = np.reshape(t,(len(t),1))
    res = np.concatenate((t,X),axis=1)
    return res, dt, pstep 

def _deriv(X, t, gamma, delta, omega, alpha, beta, source_interp):
    """Return the derivatives dx/dt and d2x/dt2."""

    V = lambda x: beta/4 * x**4 - alpha/2 * x**2 
    dVdx = lambda x: beta*x**3 - alpha*x

    x, xdot = X
    if source_interp is None:
        xdotdot = -dVdx(x) -delta * xdot + gamma * np.cos(omega*t) 
    else:
        xdotdot = -dVdx(x) -delta * xdot + gamma * np.cos(omega*t) + source_interp(t)
    return xdot, xdotdot

def duffing_equation(tmax, dt_per_period, x0, v0,gamma,delta,omega,\
        *source_args, source_func=None, source_kwargs=None,alpha=1,beta=1,t_trans=0):
    """Solve the Duffing equation for parameters gamma, delta, omega.
    https://scipython.com/blog/the-duffing-oscillator/
    Find the numerical solution to the Duffing equation using a suitable
    time grid: 
        - tmax is the maximum time (s) to integrate to; 
        - t_trans is the initial time period of transient behaviour until the solution settles down (if it does) to some kind of periodic motion (these data
    points are dropped) and 
        - dt_per_period is the number of time samples (of duration dt) to include per period of the driving motion (frequency omega).

    x'' + delta x' + alpha x + beta x^3 = gamma * cos(omega t)
    x(0) = x'(0)

    Returns the time grid, t (after t_trans), position, x, and velocity,
    xdot, dt, and step, the number of array points per period of the driving
    motion.

    """
    # Time point spacings and the time grid

    period = 2*np.pi/omega
    dt = int(2*np.pi/omega / dt_per_period * 1000)/1000
    step = int(period / dt)
    t = np.arange(0, tmax, dt)
    # Initial conditions: x, xdot
    X0 = [x0, v0]
    if callable(source_func):
        _t = np.arange(0, tmax+period, dt)
        _, source = source_func(_t, *source_args, kwargs=source_kwargs)
        source_interp = interpolate.interp1d(_t, source,kind='cubic')

    else:
        source_interp = None

    X = odeint(_deriv, X0, t, args=(gamma, delta, omega,alpha, beta, source_interp))
    idx = int(t_trans / dt)
    return t[idx:], X[idx:], dt, step

