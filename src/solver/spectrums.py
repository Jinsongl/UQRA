#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import numpy as np

def jonswap(w, Hs, Tp):
    """ JONSWAP wave spectrum, IEC 61400-3
    w: frequencies to be sampled at, hz 
    Hs: significant wave height, m
    Tp: wave peak period, sec
    """

    with np.errstate(divide='ignore'):
        # print "sample frequency: \n", w
        wp    = 2*np.pi/Tp
        gamma = 3.3 
        sigma = 0.07 * np.ones(w.shape)
        sigma[w > wp] = 0.09
        # print "wp:", wp
        # print "sigma: ", sigma
        
        assert w[0] >= 0 ,'Single side power spectrum start with frequency greater or eqaul to 0, w[0]={:4.2f}'.format(w[0])

        JS1 = 5/16 * Hs**2 * wp**4 * w**-5
        JS2 = np.exp(-1.25*(w/wp)**-4) * (1-0.287*np.log(gamma))
        JS3 = gamma**(np.exp(-0.5*((w-wp)/sigma/wp)**2))

        JS1[np.isinf(JS1)] = 0
        JS2[np.isinf(JS2)] = 0
        JS3[np.isinf(JS3)] = 0
        # print(np.isnan(JS1).any())
        JS = JS1 * JS2 * JS3

    return w, JS

def spec_test1(w, c=2):
    """
    Test FFT and iFFT for spectrum and acf 
    F(w) = Fourier(f(t))
    where
    F(w) = 2c / (c**2 + w**2)
    f(t) = e^(-c|t|)

    Arguments:
        w: frequencies to be evaluated at (Hz)
        c: arbitrary real constant larger than 0
    Returns:
        Sw: psd value at specified w
        sa: approximated area under psd curve with specified w
        
    """
    # print('\t{:s} : c= {:.2f}'.format(spec_test1.__name__, c))
    Sw = 2*c/(c**2 + w**2) 
    dw = w[1] - w[0]
    sa = np.sum(Sw*dw) 
    return w, Sw

def white_noise(w, F0=1,a=0,b=5):
    Sw = F0 
    sa = abs(b-a) * F0
    return w, Sw

