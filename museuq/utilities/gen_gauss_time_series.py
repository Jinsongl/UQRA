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
import spectrum_collections as spectrum

def nextpow2(x):
    return 2**(int(x)-1).bit_length()

def single_psd2double_psd(f,pxx):
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

def gen_gauss_time_series(t, *args, **kwargs):
    """
    Generate Gaussian time series, e.g. Gaussian wave, with given spectrum at specified args parameters
    
    Arguments:
        t: ndarry, time index 
        specturm: string, spectral function name 
        method:
            sum: sum(A_i*cos(w_i*t + theta_i))
                for 1side psd, i =  0 to N, A_i = sqrt(2*S(f)*df)
                for 2side psd, i = -N to N, A_i = sqrt(S(f)*df), note: theta(f) = -theta(-f)
            ifft: ifft(A_i * exp(j*theta_i)), A_i = sqrt(S(f)*df), theta(f) = -theta(-f)
        sides:
            1side or 2side psd 
        args: arguments needed to return spectrum density

    Return:
        t: time index, start 0 to tmax 
        etat: surface wave time series
        psd_f: frequencies of spectral density 
        eta_fft_coeffs: surface wave power spectral denstiy
    """
    # ---------------------------------------------------------
    #                   |   Range       |   delta step
    # Time domain       | [-tmax, tmax] | dt = given
    # Frequency domain  | [-fmax, fmax] | df = 1/(2(tmax))  
    #                   | fmax = 1/(2*dt)
    # ---------------------------------------------------------
    spec_dict       = spectrum.get_spec_dict()
    methods         = {'SUM':'Direct summation', 'IFFT':'Inverse Fourier Transform'}
    spectrum_name   = kwargs.get('name',    'JONSWAP')
    method          = kwargs.get('method',  'IFFT')
    sides           = kwargs.get('sides',   '1side')
    tmax,dt         = t[-1], t[1]-t[0]
    N               = int(tmax/dt)
    t               = np.arange(-N,N+1) * dt
    df              = 0.5/tmax
    spectrum_func   = spec_dict[spectrum_name.upper()]

    if sides.upper() in ['1','1SIDE','SINGLE','1SIDES']:
        f       = np.arange(N+1) * df
        theta   = np.random.uniform(-np.pi, np.pi, len(f))
        psd_f, psd_pxx = spectrum_func(f, *args)
        if method.upper() == 'SUM':
            # Direct sum with single side psd
            ampf  = np.sqrt(2*psd_pxx*(psd_f[1] - psd_f[0])) # amplitude
            # Reshape to matrix operation format
            ampf  = np.reshape(ampf,  (N+1,1))
            psd_f = np.reshape(psd_f, (N+1,1))
            theta = np.reshape(theta, (N+1,1))
            t     = np.reshape(t,     (1, 2*N+1))
            ## psd_f * t -> (N+1,1) * (1, 2*N+1) = (N+1, 2*N+1)
            eta = np.sum(ampf * np.cos(2*np.pi*psd_f*t + theta),axis=0)

        elif method.upper() == 'IFFT':
            # To use IFFT method, need to create IFFT coefficients for negative frequencies
            # Single side psd need to be divide by 2 to create double side psd, S1(f) = S1(-f) = S2(f)/2
            # Phase: theta(f) = -theta(-f) 
            theta   = np.hstack((-np.flip(theta[1:]),theta)) # concatenation along the second axis
            psd_f, psd_pxx = single_psd2double_psd(psd_f, psd_pxx)
            ampf    = np.sqrt(psd_pxx*(psd_f[1] - psd_f[0])) # amplitude

            eta_fft_coeffs = ampf * np.exp(1j*theta)
            # numpy.fft.ifft(a, n=None, axis=-1, norm=None)
            #   The input should be ordered in the same way as is returned by fft, i.e.,
            #       a[0] should contain the zero frequency term,
            #       a[1:n//2] should contain the positive-frequency terms,
            #       a[n//2 + 1:] should contain the negative-frequency terms, in increasing order starting from the most negative frequency.
            eta = np.fft.ifft(np.roll(eta_fft_coeffs,N+1)) *(2*N+1)
            eta = np.roll(eta,N).real
        else:
            raise ValueError('Mehtod {} not defined for one-side power spectral density function'.format(method))


    elif sides.upper() in ['2','2SIDE','DOUBLE','2SIDES']:

        f       = np.arange(-N,N+1) * df
        psd_f, psd_pxx = spectrum_func(f, *args)
        theta   = np.random.uniform(-np.pi, np.pi, N+1)
        theta   = np.hstack((-np.flip(theta[1:]),theta))
        ampf    = np.sqrt(psd_pxx*(psd_f[1] - psd_f[0])) # amplitude
        if method.upper() == 'SUM':
            # Direct sum with double side psd
            # Reshape to matrix operation format
            ampf    = ampf.reshape(2*N+1, 1)
            psd_f   = psd_f.reshape(2*N+1, 1)
            t       = t.reshape(1, 2*N+1)
            theta   = theta.reshape(2*N+1, 1)
            eta = np.sum(ampf * np.cos(2*np.pi*psd_f*t + theta),axis=0)
            eta = eta.reshape((eta.shape[1],))

        elif method.upper() == 'IFFT':
            eta_fft_coeffs = ampf * np.exp(1j*theta)
            eta = np.fft.ifft(np.roll(eta_fft_coeffs,N+1)) *(2*N+1)
            eta = np.roll(eta,N).real
        else:
            raise ValueError('Mehtod {} not defined for two-side power spectral density function'.format(method))
    else:
        raise ValueError('Power spectral density type {} not found'.format(sides))

    if t.ndim == 2:
        t = t.reshape((t.shape[1],))
    return t[N:2*N+1], eta[N:2*N+1]
    # return t, eta

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

