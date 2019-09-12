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


def psd2process(f, pxx):
    """
    Generate Gaussian time series for given spectrum with IFFT method
    Note: For one side psd, one need to create IFFT coefficients for negative frequencies to use IFFT method. 
        Single side psd need to be divide by 2 to create double side psd, S1(f) = S1(-f) = S2(f)/2
        Phase: theta(f) = -theta(-f) 

    Arguments:
        f: ndarry, frequency in Hz
        pxx: pwd values corresponding to f array. 
    Return:
        t: time index, start 0 to tmax 
        etat: surface wave time series
        psd_f: frequencies of spectral density 
        eta_fft_coeffs: surface wave power spectral denstiy

    Features need to add:
        1. douebl side psd
        2. padding zero values for pxx when f[0] < 0 and f is not symmetric
        3. psd2process arguments should be time, not frequency , sounds more reasonable.
            if this feature need to be added, interpolation of f may needed.

    """
    # ---------------------------------------------------------
    #                   |   Range       |   delta step
    # Time domain       | [-tmax, tmax] | dt = given
    # Frequency domain  | [-fmax, fmax] | df = 1/(2(tmax))  
    #                   | fmax = 1/(2*dt)
    # ---------------------------------------------------------
    assert f[0] == 0 , 'Frequency must start from 0 for now'
    df, fmax = f[1]-f[0], f[-1]
    tmax, dt = 0.5/df, 0.5/fmax

    theta   = np.random.uniform(-np.pi, np.pi, len(f))
    ntimes_steps = int(tmax/dt)
    t = np.arange(-ntimes_steps,ntimes_steps+1) * dt

    theta   = np.hstack((-np.flip(theta[1:]),theta)) # concatenation along the second axis
    f2, pxx2= psd.psd_single2double(f, pxx)
    ampf    = np.sqrt(pxx2*df) # amplitude

    eta_fft_coeffs = ampf * np.exp(1j*theta)
    # numpy.fft.ifft(a, n=None, axis=-1, norm=None)
    #   The input should be ordered in the same way as is returned by fft, i.e.,
    #     - a[0] should contain the zero frequency term,
    #     - a[1:n//2] should contain the positive-frequency terms,
    #     - a[n//2 + 1:] should contain the negative-frequency terms, in increasing order starting from the most negative frequency.
    eta = np.fft.ifft(np.roll(eta_fft_coeffs,ntimes_steps+1)) *(2*ntimes_steps+1)
    eta = np.roll(eta,ntimes_steps).real

    return t[ntimes_steps:2*ntimes_steps+1], eta[ntimes_steps:2*ntimes_steps+1]
    # return t, eta

