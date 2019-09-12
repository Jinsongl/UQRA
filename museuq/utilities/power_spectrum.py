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
# l = []
# for key, value in locals().items():
    # if callable(value) and value.__module__ == __name__:
        # l.append(key)
# print l

def acf2psd(tau_max, dtau, acf):
    """
    Given auto correlation function acf_tau in [0,t] with dtau, return corresponding power spectral density function
    Process is assumed to be stationary such that acf is just a function of tau and acf is symmetric. e.g. acf(-tau) = acf(tau)
    Arguments:
        tau: time interval, if only [0,tmax] is specified, [-tmax,0) will be pre-appended automatically
            Values of acf at both positive and negative lags (tau) is necessary. When applying fft(data), algorithm assumes data repeat after time interval. If only positive part provided, symmetric information of acf will not be passed ([-tmax,0] will be exactly same as [0, tmax] instead of symmetric acf(tau) = acf(-tau))
        acf: autocorrelation function or values at specified tau 
    Returns:
        Power spectral density function
        psd_f, psd_pxx
    """
    N = int(tau_max/dtau)
    tau = np.arange(-N,N)*dtau

    # Create acf values at tau if not given
    try:
        acf_tau = acf(tau)
    except TypeError:
        print(f'{acf} is not callable')

    acf_fft = np.fft.fft(np.roll(acf_tau,N))

    psd_f = np.fft.fftfreq(acf_tau.shape[-1],d=dtau)
    # Since we know acf function is even, fft of acf_tau should only contain real parts
    # psd_pxx = np.sqrt(acf_fft.real**2 + acf_fft.imag**2) * dtau
    psd_pxx = acf_fft.real * dtau
    
    # reorder frequency from negative to positive ascending order
    psd_pxx= np.array([x for _,x in sorted(zip(psd_f,psd_pxx))])
    psd_f= np.array([x for _,x in sorted(zip(psd_f,psd_f))])
    return psd_f, psd_pxx

def psd2acf(fmax, df, psd_func, spec_type='2SIDE', fmin=0):
    """
    will force one side psd to be two side work?
    """
    N = int(fmax / df)
    dt = 1/ (2*fmax)

    if spec_type.upper() == '2SIDE':
        """
        The input psd_pxx should be ordered in the same way as is returned by fft, i.e.,
        psd_pxx[0] should contain the zero frequency term,
        psd_pxx[1:n//2] should contain the positive-frequency terms,
        psd_pxx[n//2 + 1:] should contain the negative-frequency terms, in increasing order starting from the most negative frequency.
        """
        psd_f = np.fft.fftfreq(2*N, dt) 
        fmax = max(abs(psd_f))
        dt = 1/ (2*fmax)
        t = np.arange(-N,N) * dt

        psd_pxx, var_approx = psd_func(psd_f)

        ft_ifft = np.fft.ifft(psd_pxx) / dt
        ft = np.sqrt(ft_ifft.real**2 + ft_ifft.imag**2)
        ft = np.roll(ft,N)
    elif spec_type.upper() == '1SIDE':
        pass
        # implement here
    else:
        raise ValueError('spec_type {:s} is not defined'.format(spec_type))

    return t, ft

def psd_single2double(f,pxx):
    """
    Convert single side psd specified by (f, pxx) to double side
    Arguments:
        f   : single side frequency vector, could start from arbitrary value 
        pxx : single side power spectral density (PSD) estimate, pxx, 
    Returns:
        ff  : double side frequency vector
        pxx2: double side power spectral density (PSD) estimate, pxx2
    """
    assert f[0] >= 0 and f[-1] >0 # make sure frequency vector in positive range
    df  = f[1] - f[0]
    N   = len(np.arange(f[-1], 0, -df)) # does not include 0
    pxx2= np.zeros(2*N+1)
    ff  = np.zeros(2*N+1)
    # print(ff.shape, pxx2.shape)
   
    # Positive frequencies part
    ## padding psd from 0 to f[0] with 0
    N0 = len(np.arange(f[0]-df, 0, -df))
    pxx2[0:N0] = 0
    ## assign half power to corresponding frequencies
    pxx2[N0:N+1] = pxx/2
    ff[1:N+1] = np.flip(np.arange(f[-1], 0, -df))

    # Negative frequencies part
    pxx2[N+1:] = np.flip(pxx[1:]/2) 
    ff[N+1:] = -np.arange(f[-1], 0, -df)
    
    pxx2 = np.roll(pxx2, N)
    ff = np.roll(ff, N)

    return ff, pxx2

def get_spec_dict():
    spectrum_collection = {
            'JONSWAP': jonswap,
            'JS':jonswap,
            'T1':spec_test1
            }
    return spectrum_collection

def jonswap(f, Hs, Tp):
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
    sf = 2*c/(c**2 + f**2) * 2 * np.pi
    df = f[1] - f[0]
    sa = np.sum(sf*df) 
    return f, sf

       
