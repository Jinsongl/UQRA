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
from . import power_spectrum as psd


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
        f: frequencies of spectral density 
        eta_fft_coeffs: surface wave power spectral denstiy
    """
    # ---------------------------------------------------------
    #                   |   Range       |   delta step
    # Time domain       | [-tmax, tmax] | dt = given
    # Frequency domain  | [-fmax, fmax] | df = 1/(2(tmax))  
    #                   | fmax = 1/(2*dt)
    # ---------------------------------------------------------
    spec_dict       = psd.get_spec_dict()
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
        f, pxx = spectrum_func(f, *args)
        if method.upper() == 'SUM':
            # Direct sum with single side psd
            ampf  = np.sqrt(2*pxx*(f[1] - f[0])) # amplitude
            # Reshape to matrix operation format
            ampf  = np.reshape(ampf,  (N+1,1))
            f = np.reshape(f, (N+1,1))
            theta = np.reshape(theta, (N+1,1))
            t     = np.reshape(t,     (1, 2*N+1))
            ## f * t -> (N+1,1) * (1, 2*N+1) = (N+1, 2*N+1)
            eta = np.sum(ampf * np.cos(2*np.pi*f*t + theta),axis=0)

        elif method.upper() == 'IFFT':
            # To use IFFT method, need to create IFFT coefficients for negative frequencies
            # Single side psd need to be divide by 2 to create double side psd, S1(f) = S1(-f) = S2(f)/2
            # Phase: theta(f) = -theta(-f) 
            theta   = np.hstack((-np.flip(theta[1:]),theta)) # concatenation along the second axis
            f, pxx = psd.psd_single2double(f, pxx)
            ampf    = np.sqrt(pxx*(f[1] - f[0])) # amplitude

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
        f, pxx = spectrum_func(f, *args)
        theta   = np.random.uniform(-np.pi, np.pi, N+1)
        theta   = np.hstack((-np.flip(theta[1:]),theta))
        ampf    = np.sqrt(pxx*(f[1] - f[0])) # amplitude
        if method.upper() == 'SUM':
            # Direct sum with double side psd
            # Reshape to matrix operation format
            ampf    = ampf.reshape(2*N+1, 1)
            f   = f.reshape(2*N+1, 1)
            t       = t.reshape(1, 2*N+1)
            theta   = theta.reshape(2*N+1, 1)
            eta = np.sum(ampf * np.cos(2*np.pi*f*t + theta),axis=0)
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


