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

import numpy as np
from scipy import interpolate
from scipy.integrate import odeint, quad
from scipy.optimize import brentq
import utilities.spectrum_collections as spec_coll
import matplotlib.pyplot as plt

def _cal_norm_values(zeta,omega0,source_kwargs, *source_args):
    TF = lambda w : 1.0/np.sqrt((w**2-omega0**2)**2 + (2*zeta*omega0)**2)
    spec_dict = spec_coll.get_spec_dict() 
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


def duffing_oscillator(tmax,dt,x0,v0,zeta,omega0,mu,\
        *source_args, source_func=None, source_kwargs=None,t_trans=0, normalize=False):
    if normalize:
        # norm_t, norm_y = normalize[1], normalize[2]
        norm_t, norm_y = _cal_norm_values(zeta, omega0, source_kwargs) 
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
        ## Interpolate function will return [t, value interpolated at t], need only the value
        xdotdot = -dVdx(x) -delta * xdot + gamma * np.cos(omega*t) + source_interp(t)
    return xdot, xdotdot

def duffing_equation(tmax, dt_per_period, x0, v0,gamma,delta,omega,\
        *source_args, source_func=None, source_kwargs=None,alpha=1,beta=1,t_trans=0):
    """Solve the Duffing equation for parameters gamma, delta, omega.
    https://scipython.com/blog/the-duffing-oscillator/
    Find the numerical solution to the Duffing equation using a suitable
    time grid: tmax is the maximum time (s) to integrate to; t_trans is
    the initial time period of transient behaviour until the solution
    settles down (if it does) to some kind of periodic motion (these data
    points are dropped) and dt_per_period is the number of time samples
    (of duration dt) to include per period of the driving motion (frequency
    omega).

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

# def duffing_oscillator(tmax,dt,x0,v0,zeta,omega0,mu,source_func,norm=False):
    # """
    # Time domain solver for one dimentional duffing oscillator with Runge-Kutta method
    # One dimentinal duffing_oscillator dynamics.  Coefficients are normalized by mass: 2 zeta*omega0 = c/m, omega0^2 = k/m
    # where c is damping and k is stiffness 
        
    # x'' + 2*zeta*omega0*x' + omega0^2*x*(1+mu x^2) = source_func
    # with initial contidions
    # x(0) = x'(0) = 0

    # Arguments:
        # zeta, omega0: parameters for the dynamic system
        # mu: nonlinearity of the system. when mu=0, system is linear
        # spectrum_hz / correfunc:
            # exciting force process is defined either with spectrum_hz or correfunc
            # spectrum_hz = Fourier(correfunc)
        # init_conds: initial conditions for the dynamic system, default start from static
        # norm: For the purpose of perfomring a parametric study of the proposed technique, 
            # t[i] will prove expedient to normalize the equation of motion using nondimentional
            # time: tau = omega0 * t and nondimentional displacement of y = x/sigma_x
            # If norm is true, following equation is solved:
            # y'' + 2* zeta * y' + y * (1 + epsilon* y^2) = source_func/(sigma_x * omega0^2)
            # where epsilon = mu * sigma_x^2
            # Initial conditions are normalized accordingly.

    # """
    # # fig, axes = plt.subplots(1,2)
    # f1 = lambda t, x1, x2 : x2
    # f2 = lambda t, x1, x2 : -2 * zeta * omega0 * x2 - omega0**2 * x1 * (1+mu*x1**2) 
    # x1 = [x0,]
    # x2 = [v0,]
    # s0 = [source_func[0],]

    # t = np.arange(int(tmax/dt)+1) * dt
    # # Implement interpolation method for source_func 
    # source_func = np.array(source_func)
    # if source_func.shape[-1] == len(t):
        # interp_func = interpolate.interp1d(t,source_func,kind='cubic')
        # source_func = interp_func(np.arange(2*int(tmax/dt)+1) * 0.5*dt)
    # elif source_func.shape[-1] == 2*len(t) -1:
        # pass
    # else:
        # raise ValueError("System input signal array dimention doesn't match")
    # # axes[0].plot(t,source_func)

    # for i in np.arange(len(t)-1):
        # k0 = dt *  f1( t[i], x1[i], x2[i])
        # l0 = dt * (f2( t[i], x1[i], x2[i]) + source_func[2*i])

        # k1 = dt *  f1( t[i]+0.5*dt, x1[i]+0.5*k0, x2[i]+0.5*l0 )
        # l1 = dt * (f2( t[i]+0.5*dt, x1[i]+0.5*k0, x2[i]+0.5*l0) + source_func[2*i+1])

        # k2 = dt *  f1( t[i]+0.5*dt, x1[i]+0.5*k1, x2[i]+0.5*l1 )
        # l2 = dt * (f2( t[i]+0.5*dt, x1[i]+0.5*k1, x2[i]+0.5*l1) + source_func[2*i+1])

        # k3 = dt *  f1( t[i]+dt, x1[i]+k2, x2[i]+l2 )
        # l3 = dt * (f2( t[i]+dt, x1[i]+k2, x2[i]+l2) + source_func[2*i+1])

        # x1.append(x1[i] + 1/6 * (k0 + 2*k1 + 2*k2 + k3))
        # x2.append(x2[i] + 1/6 * (l0 + 2*l1 + 2*l2 + l3))
        # s0.append(source_func[2*i])

    # # axes[1].plot(t,x1)
    # # plt.grid()
    # # plt.show()
    # return np.array([t, s0, x1, x2]).T

# def transfer_func(f,f_n=0.15, zeta=0.1):
    # """
    # Transfer function of single degree freedom system.
    # f_n: natural frequency, Hz
    # zeta: dampling coefficient
    # """
    # fr = f/f_n
    # y1 = (1-fr**2)**2
    # y2 = (2*zeta*fr) **2

    # y = np.sqrt(y1 + y2)
    # y = 1./y 
    # return y

# def deterministic_lin_sdof(Hs, Tp, T=int(1e2), dt=0.1, seed=[0,100]):
    # """
    # Dynamics of deterministic linear sdof system with given inputs: Hs, Tp
    # (Hs,Tp): Environment variables
    # T: Simulation duration in seconds
    # dt: Simulation time step, default=0.1
    # seed=[bool, int], if seed[0]==True, seed is fixed with seed[1]
    # """
    # if seed[0]:
        # np.random.seed(int(seed[1]))
    # else:
        # np.random.seed() 

    # numPts_T = int(nextpow2(T/dt)) ## Number of points in Time domain
    # numPts_F = numPts_T
    # df      = 1.0/(numPts_T * dt)
    # f       = np.arange(1,numPts_F) * df

    # # H = np.ones(f.shape)
    # H = transfer_func(f)
    # t, etat, psd_f, eta_fft_coeffs = gen_surfwave('jonswap',f ,Hs, Tp)
    # t = t * dt
    # assert np.array_equal(f, psd_f)
    # psd_y  = eta_fft_coeffs * H 
    # y   = np.fft.ifft(psd_y).real * numPts_F 
    # # print("\tSystem response Done!")
    # # print "  > Significant wave height check:"
    # # print "     Area(S(f))/Hs: ",'{:04.2f}'.format(4 * np.sqrt(JS_area) /Hs)    
    # # print "     4*std/Hs:      ",'{:04.2f}'.format(4 * np.std(etat[int(100.0/dt):])/Hs)
    # t = t[:int(T/dt)]
    # etat = etat[:int(T/dt)]
    # y = y[:int(T/dt)]
    # res = np.array([t,etat, y]).T
    # np.random.seed() 
    # # return t, etat, y 
    # return res
