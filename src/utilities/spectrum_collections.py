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


def spec_jonswap(f, Hs, Tp):
    """ JONSWAP wave spectrum, IEC 61400-3
    f: frequencies to be sampled at, hz 
    Hs: significant wave height, m
    Tp: wave peak period, sec
    """

    with np.errstate(divide='ignore'):
        # print "sample frequency: \n", f
        fp = 1.0/Tp
        fr = f/fp
        gamma = 3.3 
        sigma = 0.07 * np.ones(f.shape)
        sigma[f > fp] = 0.09
        # print "fp:", fp
        # print "sigma: ", sigma
        
        assert f[0] >= 0 ,'Single side power spectrum start with frequency greater or eqaul to 0, f[0]={:4.2f}'.format(f[0])

        # if fr[0] == 0:
            # fr[0] = 1/np.inf
        JS1 = 0.3125 * Hs**2 * Tp * fr**-5
        JS2 = np.exp(-1.25*fr**-4) * (1-0.287*np.log(gamma))
        JS3 = gamma**(np.exp(-0.5*(fr-1)**2/sigma**2))

        JS1[np.isinf(JS1)] = 0
        JS2[np.isinf(JS2)] = 0
        JS3[np.isinf(JS3)] = 0
        # print(np.isnan(JS1).any())
        # print(np.isnan(JS2).any())
        # print(np.isnan(JS3).any())
        # print(np.isinf(JS1).any())
        # print(np.isinf(JS2).any())
        # print(np.isinf(JS3).any())
        JS = JS1 * JS2 * JS3

    return f, JS

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
    # print('\t{:s} : c= {:.2f}'.format(spec_test1.__name__, c))
    f = 2 * np.pi * f
    sf = 2*c/(c**2 + f**2)
    df = f[1] - f[0]
    sa = np.sum(sf*df) 
    return f, sf

def get_spec_dict():
    spectrum_collection = {
            'JONSWAP': spec_jonswap,
            'JS':spec_jonswap,
            'T1':spec_test1
            }
    return spectrum_collection
       
