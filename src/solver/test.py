#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
from dynamic_models import *
import numpy as np
import matplotlib.pyplot as plt


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
###  PSD to ACF and ACF to PSD test
###  Given one equation get the other
###  Psd = Fourier (ACF)
################################################################################

    psd_tests = [spec_test1, spec_test2, spec_test3] 
    acf_tests = [acf_test1, acf_test2, acf_test3] 

    f, axes = plt.subplots(len(psd_tests),2)
    t_max = 10
    dt = .01
    N = int(t_max/dt)
    t = np.arange(-N, N) * dt


    for i, (psd_func, acf_func) in enumerate(zip(psd_tests, acf_tests)):

        acf_true = acf_func(t) 
        ## Given psd function and the time domain want to achieve,compare 
        ## calculated acf and true acf
        t_sim, ft_sim = psd2acf(0.5/dt, 0.5/t_max, psd_func)
        print(t_sim, ft_sim)

        axes[i,0].plot(t,acf_true,label='real')
        axes[i,0].plot(t_sim,ft_sim,label='sim')
        # axes[0].legend()

        freq, amp = acf2psd(t_max,dt, acf_func) 
        axes[i,1].plot(freq, psd_func(freq)[0],label='real')
        axes[i,1].plot(freq, amp,label='sim')
        # axes[1].plot(freq, amp - psd_func(freq)[0])
        axes[i,1].legend()
    plt.show()
################################################################################
    # f, axes = plt.subplots(1,2)
    # t_max = 0.5
    # dt = 0.001
    # N = int(t_max/dt)
    # t = np.arange(-N, N) * dt
    # acf_true = acf_test2(t) 

    # freq = np.fft.fftfreq(int(t_max/dt)*2, d=dt)
    # t_sim, ft_sim = psd2acf(0.5/dt,0.5/t_max, spec_test2)

    # axes[0].plot(t,acf_true,label='real')
    # axes[0].plot(t_sim,ft_sim, label='sim')
    # axes[0].legend()

    # freq, amp = acf2psd(t_max, dt, acf_test2) 
    # axes[1].plot(freq, spec_test2(freq)[0],label='real')
    # axes[1].plot(freq, amp,label='sim')
    # axes[1].legend()
    # plt.show()

    # f, axes = plt.subplots(1,2)
    # t_max = 0.5
    # dt = 0.001
    # t = np.arange(0,t_max,dt)
    # acf_true = acf_test3(t) 
    # freq = np.fft.fftfreq(int(t_max/dt)*2, d=dt)
    # t_sim, ft_sim = twoside_psd2acf(freq, spec_test3)

    # axes[0].plot(t,acf_true,label='real')
    # axes[0].plot(t_sim,ft_sim, label='sim')
    # axes[0].legend()

    # freq, amp = acf2psd(t,acf_test3) 
    # axes[1].plot(freq, spec_test2(freq)[0],label='real')
    # axes[1].plot(freq, amp,label='sim')
    # axes[1].legend()
    # plt.show()

if __name__ == '__main__':
    main()
