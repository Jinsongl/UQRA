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
import numpy.random as rn
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as sig

from scipy import interpolate
import matplotlib.pyplot as plt


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
    # # JS_area = np.sum(JS*df)

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

def lin_oscillator(tmax,dt,t_trans,x0,v0,zeta,omega0,f, retall=False):

    t, x = duffing_oscillator(tmax,dt,t_trans,x0,v0,zeta,omega0,0,f)
    Hw = lambda w: np.sqrt(1/((w**2 - omega0**2 )**2 + (2*zeta*omega0)**2 )) 

    return t, x, Hw if retall else t, x

def duffing_oscillator(tmax, dt, t_trans, x0, v0, zeta, omega0, mu, f, norm=False):
    """
    Time domain solver for one dimentional duffing oscillator with Runge-Kutta method
    One dimentinal duffing_oscillator dynamics.  Coefficients are normalized by mass: 2 zeta*omega0 = c/m, omega0^2 = k/m
    where c is damping and k is stiffness 
        
    x'' + 2*zeta*omega0*x' + omega0^2*x*(1+mu x^2) = f
    with initial contidions
    x(0) = x'(0) = 0

    Arguments:
        zeta, omega0: parameters for the dynamic system
        mu: nonlinearity of the system. when mu=0, system is linear
        spectrum_hz / correfunc:
            exciting force process is defined either with spectrum_hz or correfunc
            spectrum_hz = Fourier(correfunc)
        init_conds: initial conditions for the dynamic system, default start from static
        norm: For the purpose of perfomring a parametric study of the proposed technique, 
            t[i] will prove expedient to normalize the equation of motion using nondimentional
            time: tau = omega0 * t and nondimentional displacement of y = x/sigma_x
            If norm is true, following equation is solved:
            y'' + 2* zeta * y' + y * (1 + epsilon* y^2) = f/(sigma_x * omega0^2)
            where epsilon = mu * sigma_x^2
            Initial conditions are normalized accordingly.

    """
    # fig, axes = plt.subplots(1,2)
    f1 = lambda t, x1, x2 : x2
    f2 = lambda t, x1, x2 : -2 * zeta * omega0 * x2 - omega0**2 * x1 * (1+mu*x1**2) 
    x1 = [x0,]
    x2 = [v0,]

    t = np.arange(int(tmax/dt)+1) * dt
    # Implement interpolation method for f 
    f = np.array(f)
    if f.shape[-1] == len(t):
        interp_func = interpolate.interp1d(t,f,kind='cubic')
        f = interp_func(np.arange(2*int(tmax/dt)+1) * 0.5*dt)
    elif f.shape[-1] == 2*len(t) -1:
        pass
    else:
        raise ValueError("External force array dimention doesn't match")
    # axes[0].plot(t,f)

    for i in np.arange(len(t)-1):
        k0 = dt *  f1( t[i], x1[i], x2[i])
        l0 = dt * (f2( t[i], x1[i], x2[i]) + f[2*i])

        k1 = dt *  f1( t[i]+0.5*dt, x1[i]+0.5*k0, x2[i]+0.5*l0 )
        l1 = dt * (f2( t[i]+0.5*dt, x1[i]+0.5*k0, x2[i]+0.5*l0) + f[2*i+1])

        k2 = dt *  f1( t[i]+0.5*dt, x1[i]+0.5*k1, x2[i]+0.5*l1 )
        l2 = dt * (f2( t[i]+0.5*dt, x1[i]+0.5*k1, x2[i]+0.5*l1) + f[2*i+1])

        k3 = dt *  f1( t[i]+0.5*dt, x1[i]+0.5*k2, x2[i]+0.5*l2 )
        l3 = dt * (f2( t[i]+0.5*dt, x1[i]+0.5*k2, x2[i]+0.5*l2) + f[2*i+1])

        x1.append(x1[i] + 1/6 * (k0 + 2*k1 + 2*k2 + k3))
        x2.append(x2[i] + 1/6 * (l0 + 2*l1 + 2*l2 + l3))

    # axes[1].plot(t,x1)
    # plt.grid()
    # plt.show()
    return t, x1


