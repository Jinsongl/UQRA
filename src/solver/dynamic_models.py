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

import matplotlib.pyplot as plt
def nextpow2(x):
    return 2**(int(x)-1).bit_length()


def spec_jonswap(f, Hs, Tp):
    """ JONSWAP wave spectrum, IEC 61400-3
    f: frequencies to be sampled at, hz 
    Hs: significant wave height, m
    Tp: wave peak period, sec
    """

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
    JS = JS1 * JS2 * JS3
    JS[np.isnan(JS)] = 0

    return f, JS


def single_psd2double_psd(f,sf):
    """
    Convert single side psd to double side
    """
    assert f[0] >= 0 and f[-1] >0
    df  = f[1] - f[0]
    N   = len(np.arange(f[-1], 0, -df))
    sff = np.zeros(2*N+1)
    ff  = np.zeros(2*N+1)
    # print(ff.shape, sff.shape)
   
    # Positive frequencies part
    N0 = len(np.arange(f[0]-df, 0, -df))
    # print(N0)
    sff[0:N0] = 0
    sff[N0:N+1] = sf/2
    ff[1:N+1] = np.flip(np.arange(f[-1], 0, -df))

    # Negative frequencies part
    sff[N+1:] = np.flip(sf[1:]/2) 
    ff[N+1:] = -np.arange(f[-1], 0, -df)
    
    sff = np.roll(sff, N)
    ff = np.roll(ff, N)

    return ff, sff

spectrum_collection = {
        'JONSWAP': spec_jonswap,
        'JS':spec_jonswap
        }

def gen_gauss_time_series(tmax, dt, spectrum_name, *args, method='sum', sides='1side'):
    """
    Generate Gaussian time series, e.g. Gaussian wave, with given spectrum at specified args parameters
    
    Arguments:
        tmax: maximum time duration
        dt: time step
        specturm: string, spectral function name 
        method:
            sum: sum(A_i*cos(w_i*t + theta_i))
                for 1side psd, i =  0 to N, A_i = sqrt(2 * S(f) * df)
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
    methods = {'SUM':'Direct summation',
            'IFFT':'Inverse Fourier Transform'}
    N = int(tmax/dt)
    t = np.arange(-N,N+1) * dt
    tmax = t[-1]
    df = 0.5/tmax
    spectrum_func = spectrum_collection[spectrum_name.upper()]
    print('\tGenerating Gaussian time series in [0, {:4.2}] with dt={:4.2}'.format(tmax, dt))
    print('\t>>> Power spectrum: {}'.format(spectrum_func.__name__))
    print('\t>>> Method: {}'.format(methods[method.upper()]))


    if sides.upper() in['1','1SIDE','SINGLE','1SIDES']:
        f= np.arange(N+1) * df
        theta = np.random.uniform(-np.pi, np.pi, len(f))
        psd_f, psd_pxx = spectrum_func(f, *args)
        if method.upper() == 'SUM':
            # Direct sum with single side psd
            ampf = np.sqrt(2*psd_pxx*(psd_f[1] - psd_f[0])) # amplitude

            # Reshape to matrix operation format
            ampf  = np.reshape(ampf,  (N+1,1))
            psd_f = np.reshape(psd_f, (N+1,1))
            theta = np.reshape(theta, (N+1,1))
            t     = np.reshape(t,     (1, 2*N+1))

            eta = np.sum(ampf * np.cos(2*np.pi*psd_f*t + theta),axis=0)

        elif method.upper() == 'IFFT':
            # To use IFFT method, need to create IFFT coefficients for negative frequencies
            # Single side psd need to be divide by 2 to create double side psd, S1(f) = S1(-f) = S2(f)/2
            # Phase: theta(f) = -theta(-f) 
            theta = np.hstack((-np.flip(theta[1:]),theta))
            psd_f, psd_pxx = single_psd2double_psd(psd_f, psd_pxx)
            ampf    = np.sqrt(psd_pxx*(psd_f[1] - psd_f[0])) # amplitude

            eta_fft_coeffs = ampf * np.exp(1j*theta)
            eta = np.fft.ifft(np.roll(eta_fft_coeffs,N+1)) *(2*N+1)
            eta = np.roll(eta,N).real
        else:
            raise ValueError('Mehtod {} not defined for one-side power spectral density function'.format(method))


    elif sides.upper() in ['2','2SIDE','DOUBLE','2SIDES']:

        f = np.arange(-N,N+1) * df
        psd_f, psd_pxx = spectrum_func(f, *args)
        theta = np.random.uniform(-np.pi, np.pi, N+1)
        theta = np.hstack((-np.flip(theta[1:]),theta))
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
            # IFFT

            eta_fft_coeffs = ampf * np.exp(1j*theta)
            eta = np.fft.ifft(np.roll(eta_fft_coeffs,N+1)) *(2*N+1)
            eta = np.roll(eta,N).real

        else:
            raise ValueError('Mehtod {} not defined for two-side power spectral density function'.format(method))
    else:
        raise ValueError('Power spectral density type {} not found'.format(sides))

    if t.ndim == 2:
        t = t.reshape((t.shape[1],))
    return t, eta

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
    assert callable(acf)
    acf_tau = acf(tau)
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


def transfer_func(f,f_n=0.15, zeta=0.1):
    """
    Transfer function of single degree freedom system.
    f_n: natural frequency, Hz
    zeta: dampling coefficient
    """
    fr = f/f_n
    y1 = (1-fr**2)**2
    y2 = (2*zeta*fr) **2

    y = np.sqrt(y1 + y2)
    y = 1./y 
    return y
def deterministic_lin_sdof(Hs, Tp, T=int(1e2), dt=0.1, seed=[0,100]):
    """
    Dynamics of deterministic linear sdof system with given inputs: Hs, Tp
    (Hs,Tp): Environment variables
    T: Simulation duration in seconds
    dt: Simulation time step, default=0.1
    seed=[bool, int], if seed[0]==True, seed is fixed with seed[1]
    """
    if seed[0]:
        np.random.seed(int(seed[1]))
    else:
        np.random.seed() 

    numPts_T = int(nextpow2(T/dt)) ## Number of points in Time domain
    numPts_F = numPts_T
    df      = 1.0/(numPts_T * dt)
    f       = np.arange(1,numPts_F) * df
    # JS_area = np.sum(JS*df)

    # H = np.ones(f.shape)
    H = transfer_func(f)
    t, etat, psd_f, eta_fft_coeffs = gen_surfwave('jonswap',f ,Hs, Tp)
    t = t * dt
    assert np.array_equal(f, psd_f)
    psd_y  = eta_fft_coeffs * H 
    y   = np.fft.ifft(psd_y).real * numPts_F 
    # print("\tSystem response Done!")
    # print "  > Significant wave height check:"
    # print "     Area(S(f))/Hs: ",'{:04.2f}'.format(4 * np.sqrt(JS_area) /Hs)    
    # print "     4*std/Hs:      ",'{:04.2f}'.format(4 * np.std(etat[int(100.0/dt):])/Hs)
    t = t[:int(T/dt)]
    etat = etat[:int(T/dt)]
    y = y[:int(T/dt)]
    res = np.array([t,etat, y]).T
    np.random.seed() 
    # return t, etat, y 
    return res


def main(Hs=12.5,Tp=15.3,T=int(1e2)):
    # Test Gaussian Wave

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

    np.random.seed( 10 )
    t1, eta1 = gen_gauss_time_series(tmax, dt, 'JS',  Hs, Tp, method='sum')
    np.random.seed( 10 )
    t2, eta2 = gen_gauss_time_series(tmax, dt, 'JS',  Hs, Tp, method='ifft')
    axes[0].set_xlim(0,1)
    axes[1].plot(t1, eta1)
    axes[1].plot(t2, eta2)

    # axes[1].set_xlim(0,1)

    # axes[1].plot(t, eta1)
    # axes[1].plot(t, eta2)
    print('std(eta1) = {:f}'.format(np.std(eta1)))
    print('std(eta2) = {:f}'.format(np.std(eta2)))

    plt.show()




if __name__ == "__main__":
    main()

