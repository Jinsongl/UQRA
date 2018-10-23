#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""

import numpy as np
import matplotlib.pyplot as plt

def duffing_oscillator(tmax, dt, t_trans, x0, v0, zeta, omega0, mu, spectrum_hz=None,correfunc=None,init_conds=[0,0], norm=False):
    """
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
    # Evalute f
    f = np.sin(t)
    # axes[0].plot(t,f)

    for i in np.arange(len(t)-1):
        k0 = dt *  f1( t[i], x1[i], x2[i])
        l0 = dt * (f2( t[i], x1[i], x2[i]) + f[i])

        k1 = dt *  f1( t[i]+0.5*dt, x1[i]+0.5*k0, x2[i]+0.5*l0 )
        l1 = dt * (f2( t[i]+0.5*dt, x1[i]+0.5*k0, x2[i]+0.5*l0) + 0.5*(f[i] + f[i+1]))

        k2 = dt *  f1( t[i]+0.5*dt, x1[i]+0.5*k1, x2[i]+0.5*l1 )
        l2 = dt * (f2( t[i]+0.5*dt, x1[i]+0.5*k1, x2[i]+0.5*l1) + 0.5*(f[i] + f[i+1]))

        k3 = dt *  f1( t[i]+0.5*dt, x1[i]+0.5*k2, x2[i]+0.5*l2 )
        l3 = dt * (f2( t[i]+0.5*dt, x1[i]+0.5*k2, x2[i]+0.5*l2) + 0.5*(f[i] + f[i+1]))

        x1.append(x1[i] + 1/6 * (k0 + 2*k1 + 2*k2 + k3))
        x2.append(x2[i] + 1/6 * (l0 + 2*l1 + 2*l2 + l3))

    # axes[1].plot(t,x1)
    # plt.grid()
    # plt.show()
    return t, x1

def main():
    fig, axes = plt.subplots(2,2)
    # Test 1: y'' -y = 0, y0= 1, y'0 = 0 =>  y = 0.5*e^-t + 0.5*e^t
    t, x= duffing_oscillator(10,0.1,2,1,0,0,-1j,0)
    y = 0.5 * np.exp(-t) + 0.5 * np.exp(t)
    axes[0,0].plot(t, x, t, y )
    # axes[0,0].plot(t, abs((x-y)/y ))

    # Test 2: y'' +5y' +4y = 0, y(0) = 1, y'(0) = -7 => y = -e^-t + 2*e^-4t
    t, x= duffing_oscillator(10,0.1,2,1,-7,1.25,2,0)
    y =  -np.exp(-t) + 2 * np.exp(-4*t)
    axes[0,1].plot(t, x, t, y )
 
    # Test 3: y"+ 2y' +5y = 0 , y(0)= 4, y'(0) = 6 => y = 4e−t* cos 2t + 5e−t*sin 2t. 
    t, x= duffing_oscillator(10,0.1,2,4,6,1/np.sqrt(5),np.sqrt(5),0)
    y =  4 * np.exp(-t)*np.cos(2*t) + 5*np.exp(-t) * np.sin(2*t)
    axes[1,0].plot(t, x, t, y )

    # Test 4: y"+ y' -2y = sinx , y(0)= 1, y'(0) = 0 => y =c1*e^x+c2*e^-2x-0.1*(cos x +3 sin x) 
    t, x= duffing_oscillator(10,0.1,2,1,0,1/(2j*np.sqrt(2)),np.sqrt(2)*1j,0)
    y =  5/6 * np.exp(t) + 4/15 * np.exp(-2*t) - 0.1* (np.cos(t) + 3 * np.sin(t))
    axes[1,1].plot(t, x, t, y )

    plt.show()

if __name__=='__main__':
    main()
    

