#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import context
import numpy as np
import matplotlib.pyplot as plt
import time
from utilities.gen_gauss_time_series import *
# from src/utilities import gen_gauss_time_series
# from gen_gauss_time_series.py import * 
# from dynamic_models import *



def spec_test1(f, c=2):
    """
    Test FFT and iFFT for spectrum and acf 
    F(w) = Fourier(f(t))
    where
    F(w) = 2c / (c**2 + w**2)
    f(t) = e^(-c|t|)

    Arguments:
        f: frequencies to be evaluated at (Hz)
        c: arbitrary real constant larger than 0
    Returns:
        sf: psd value at specified f
        sa: approximated area under psd curve with specified f
        
    """
    f = 2 * np.pi * f
    sf = 2*c/(c**2 + f**2)
    df = f[1] - f[0]
    sa = np.sum(sf*df) 
    return sf, sa
   
def spec_test2(f):
    f = 2*np.pi*f
    # df = f[1] - f[0]
    # sf = 2 * np.sinc(f)
    # sa = np.sum(sf * df)

    sf = 2 * np.sin(f) /f
    sf[np.isnan(sf)] = 2
    return sf, 2

def spec_test3(w):
    w = 2*np.pi*w
    dw = w[1] - w[0]
    sw = abs(w)<1 
    sa = np.sum(sw * dw)
    return sw, sa

def acf_test1(t,c=2):
    """
    f(t) corresponding to spec_test1
    """
    return np.exp(-c*abs(t)) 
def acf_test2(t):
    return abs(t)<=1
def acf_test3(t):
    return np.sinc(t)/np.pi


def main():
################################################################################ 
    # T=100
    # dt = 0.01
    # seed = [1,100]
    # y = deterministic_lin_sdof(Hs,Tp,T,dt,seed=seed)

################################################################################
# >>>  PSD to ACF and ACF to PSD test
# >>>  Given one equation get the other
# >>>  Psd = Fourier (ACF)
################################################################################

    # psd_tests = [spec_test1, spec_test2, spec_test3] 
    # acf_tests = [acf_test1, acf_test2, acf_test3] 

    # f, axes = plt.subplots(len(psd_tests),2)
    # t_max = 10
    # dt = .01
    # N = int(t_max/dt)
    # t = np.arange(-N, N) * dt


    # for i, (psd_func, acf_func) in enumerate(zip(psd_tests, acf_tests)):
        # acf_true = acf_func(t) 
        # ## Given psd function and the time domain want to achieve,compare 
        # ## calculated acf and true acf
        # t_sim, ft_sim = psd2acf(0.5/dt, 0.5/t_max, psd_func)
        # # print(t_sim, ft_sim)

        # axes[i,0].plot(t,acf_true,label='real')
        # axes[i,0].plot(t_sim,ft_sim,label='sim')
        # # axes[0].legend()

        # freq, amp = acf2psd(t_max,dt, acf_func) 
        # axes[i,1].plot(freq, psd_func(freq)[0],label='real')
        # axes[i,1].plot(freq, amp,label='sim')
        # # axes[1].plot(freq, amp - psd_func(freq)[0])
        # axes[i,1].legend()
    # plt.show()


################################################################################
# >>>  Linear Oscillator testing 
# >>>  Tested free decay cases for linear system (u = 0 in duffing oscillator) 
# >>>  
################################################################################
    # fig, axes = plt.subplots(2,2)
    # f = np.zeros(int(10/0.1)+1) ##
    # # Test 1: y'' -y = 0, y0= 1, y'0 = 0 =>  y = 0.5*e^-t + 0.5*e^t
    # t, x= duffing_oscillator(10,0.1,2,1,0,0,-1j,0, f)
    # y = 0.5 * np.exp(-t) + 0.5 * np.exp(t)
    # axes[0,0].plot(t, x, t, y )
    # # axes[0,0].plot(t, abs((x-y)/y ))

    # # Test 2: y'' +5y' +4y = 0, y(0) = 1, y'(0) = -7 => y = -e^-t + 2*e^-4t
    # t, x= duffing_oscillator(10,0.1,2,1,-7,1.25,2,0, f)
    # y =  -np.exp(-t) + 2 * np.exp(-4*t)
    # axes[0,1].plot(t, x, t, y )
 
    # # Test 3: y"+ 2y' +5y = 0 , y(0)= 4, y'(0) = 6 => y = 4e−t* cos 2t + 5e−t*sin 2t. 
    # t, x= duffing_oscillator(10,0.1,2,4,6,1/np.sqrt(5),np.sqrt(5),0, f)
    # y =  4 * np.exp(-t)*np.cos(2*t) + 5*np.exp(-t) * np.sin(2*t)
    # axes[1,0].plot(t, x, t, y )

    # # Test 4: y"+ y' -2y = sinx , y(0)= 1, y'(0) = 0 => y =c1*e^x+c2*e^-2x-0.1*(cos x +3 sin x) 

    # f = np.sin(np.arange(int(10/0.1)+1) * 0.1)
    # t, x= duffing_oscillator(10,0.1,2,1,0,1/(2j*np.sqrt(2)),np.sqrt(2)*1j,0,f)
    # y =  5/6 * np.exp(t) + 4/15 * np.exp(-2*t) - 0.1* (np.cos(t) + 3 * np.sin(t))
    # axes[1,1].plot(t, x, t, y )

    # plt.show()

################################################################################
# >>>  Gaussian time series generation 
# >>>  Tested Gaussian time series generator with ifft and direct summation method
# >>>  
################################################################################

    tmax = 100
    dt = 0.1
    Hs = 8
    Tp = 6
    fig, axes = plt.subplots(1,2)

    N = int(tmax/dt)
    t = np.arange(-N,N+1) * dt
    df = 0.5/tmax
    f_fft = np.arange(-N,N+1) * df
    f_sum= np.arange(N+1) * df

    f, sf = spec_jonswap(f_sum,Hs, Tp)
    f2,sf2 = single_psd2double_psd(f, sf)
    start_time = time.time()
    np.random.seed( 10 )
    t1, eta1 = gen_gauss_time_series(t, 'JS',  Hs, Tp, method='sum')
    elapsed_time = time.time() - start_time
    print('SUM elapsed time(sec) :{:.2f}'.format(elapsed_time))

    start_time = time.time()
    np.random.seed( 10 )
    t2, eta2 = gen_gauss_time_series(t, 'JS',  Hs, Tp, method='ifft')
    elapsed_time = time.time() - start_time
    print('IFFT elapsed time(sec) :{:.2f}'.format(elapsed_time))
    axes[0].set_xlim(0,1)
    axes[1].plot(t1, eta1,label='sum')
    axes[1].plot(t2, eta2,label='ifft')
    axes[1].legend()
    

    # axes[1].set_xlim(0,1)

    # axes[1].plot(t, eta1)
    # axes[1].plot(t, eta2)
    # print('std(eta1) = {:f}'.format(np.std(eta1)))
    print('std(eta2) = {:f}'.format(np.std(eta2)))

    plt.show()



################################################################################
################################################################################
################################################################################

if __name__ == '__main__':
    main()
