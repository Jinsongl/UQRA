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
   
def spec_test2(w):
    w = 2*np.pi*w
    dw = w[1] - w[0]
    sw = 2 * np.sinc(w)
    sa = np.sum(sw * dw)
    return sw, sa

def spec_test3(w):
    w = 2*np.pi*w
    dw = w[1] - w[0]
    sw = abs(w)<1 
    sa = np.sum(sw * dw)
    return sw, sa
def spec_jonswap(f,Hs,Tp):
    """ JONSWAP wave spectrum, IEC 61400-3
    f: frequencies to be sampled at, hz 
    Hs: significant wave height, m
    Tp: wave peak period, sec
    """

    f = np.asarray(f)
    # print "sample frequency: \n", f
    fp = 1.0/Tp
    fr = f/fp
    gamma = 3.3 
    sigma = 0.07 * np.ones(f.shape)
    sigma[f > fp] = 0.09
    # print "fp:", fp
    # print "sigma: ", sigma
    
    JS1 = 0.3125 * Hs**2 * Tp * fr**-5
    JS2 = np.exp(-1.25*fr**-4) * (1-0.287*np.log(gamma))
    JS3 = gamma**(np.exp(-0.5*(fr-1)**2/sigma**2))
    JS = JS1 * JS2 * JS3
    return JS

spectrum_collection = {
        'JONSWAP': spec_jonswap,
        'JW':spec_jonswap
        }

def acf_test1(t,c=2):
    """
    f(t) corresponding to spec_test1
    """
    return np.exp(-c*abs(t)) 
def acf_test2(t):
    return abs(t)<=1
def acf_test3(t):
    return np.sinc(t)/np.pi

def gen_surfwave(spectrum_name, *args):
    """
    Generate surface wave time series with given spectrum at specified args parameters
    
    Arguments:
        psd_f: frequency in Hz to be sampled at
        specturm: string, spectral function name 
        args: arguments needed to return spectrum density

    Return:
        t: time start with 1, need to multiply by delta_t 
        etat: surface wave time series
        psd_f: frequencies of spectral density 
        psd_eta: surface wave power spectral denstiy
    """
    spectrum_func = spectrum_collection[spectrum_name.upper()]
    psd_f, psd_pxx = spectrum_func(*args)
    df      = psd_f[1] - psd_f[0]
    ampf    = np.sqrt(2*psd_pxx*df) # amplitude
    theta   = np.random.uniform(-np.pi, np.pi, len(psd_f))
    psd_eta = ampf * np.exp(1j*theta)
    etat    = np.fft.ifft(psd_eta).real * len(psd_f)
    t       = np.arange(1, len(psd_f))

    return (t, etat, psd_f, psd_eta)

def acf2psd(tau, acf):
    """
    Given auto correlation function acf_tau in [0,t] with dt, return corresponding power spectral density function
    Process is assumed to be stationary such that acf is just a function of tau and acf is symmetric. e.g. acf(-tau) = acf(tau)
    Arguments:
        tau: time intervals
        acf: autocorrelation values at specified tau or
            callable function 
    Returns:
        Power spectral density function and evaluated frequencies
        psd_f, psd_s
    """
    tau = np.array(tau)
    dt = tau[1]-tau[0]
    if tau[0] < 0:
        tau_max = max(abs(tau))
    else:
        assert tau[0] == 0
        assert tau[-1] == max(tau)
        tau_max = tau[-1]
    N = int(tau_max/dt)
    tau = np.arange(-N,N)*dt

    # Create acf values at tau if not given
    if callable(acf):
        acf_tau = acf(tau)
    else:
        acf_tau = acf
    # acf_tau must be symmetric and include negative parts
    # acf_tau = np.hstack((np.flip(acf_tau,axis=0)[0:-1], acf_tau))
    acf_fft = np.fft.fft(acf_tau)
    psd_f = np.fft.fftfreq(acf_tau.shape[-1],d=dt)
    psd_s = np.sqrt(acf_fft.real**2 + acf_fft.imag**2) * dt  
    psd_s= np.array([x for _,x in sorted(zip(psd_f,psd_s))])
    psd_f= np.array([x for _,x in sorted(zip(psd_f,psd_f))])
    return psd_f, psd_s


def twoside_psd2acf(psd_f,spectrum):
    """
    will force one side psd to be two side work?
    The input psd_s should be ordered in the same way as is returned by fft, i.e.,
        psd_s[0] should contain the zero frequency term,
        psd_s[1:n//2] should contain the positive-frequency terms,
        psd_s[n//2 + 1:] should contain the negative-frequency terms, in increasing order starting from the most negative frequency.
    """
    psd_f = np.array(psd_f)
    assert psd_f[0] == 0
    df = psd_f[1] - psd_f[0]
    if psd_f[-1] < 0:
        fmax = max(abs(psd_f))
    else:
        assert psd_f[-1] == max(psd_f)
        fmax = psd_f[-1]
    N = int(fmax / df)
    psd_f = np.arange(-N, N) * df
    psd_f = np.roll(psd_f, N)

    dt = 1/ (2*fmax)

    if callable(spectrum):
        psd_s, var_approx = spectrum(psd_f)
    else:
        psd_s = spectrum

    t = np.arange(0,N) * dt
    ft_ifft = np.fft.ifft(psd_s) / dt
    ft = np.sqrt(ft_ifft.real**2 + ft_ifft.imag**2)[0:N]

    return t, ft

    



def gen_process():
    pass

def gen_process_from_spectrum(spectrum_name, psd_order, method='SR', *args):
    """
    Generate one stochastic process realization (time series) with given spectrum and specified args parameters
    
    Arguments:
        specturm: string, spectral function name 
        method:
            SR: spectral representation
            KL: Karhunen Loeve Expansion
        psd_order: accuracy order for selected method
            SR: Frequency in Hz to be sampled at of shape(n,), larger n will lead to more precise estimation of psd
            KL: Number of terms used in KL expansion 
        args: arguments needed to return spectrum density

    Return:
        process: stochastic process indexed with time
    """
    spectrum_func = spectrum_collection[spectrum_name.upper()]
    if method.upper() == 'SR':
        psd_f = np.asarray(psd_order)
        print('\tCreating process with spectral representation, N = {:d}'.format(len(psd_f)))
        psd_pxx = spectrum_func(psd_f, *args)
        df      = psd_f[1] - psd_f[0]
        ampf    = np.sqrt(2*psd_pxx*df) # amplitude
        theta   = np.random.uniform(-np.pi, np.pi, len(psd_f))
        psd_eta = ampf * np.exp(1j*theta)
        process = np.fft.ifft(psd_eta).real * len(psd_f)
    elif method.upper() == 'KL':
        psd_order = int(psd_order)
        # Transform spectrum to correlation function 

        # Generate process with correlation function
        process = gen_process_from_correfunc()

    return process 


def gen_process_from_correfunc():
    pass

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
    t, etat, psd_f, psd_eta = gen_surfwave('jonswap',f ,Hs, Tp)
    t = t * dt
    assert np.array_equal(f, psd_f)
    psd_y  = psd_eta * H 
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
    # T=100
    # dt = 0.01
    # seed = [1,100]
    # y = deterministic_lin_sdof(Hs,Tp,T,dt,seed=seed)

    # f, axes = plt.subplots(1,2)
    # t_max = 20
    # dt = 0.01
    # t = np.arange(0,t_max,dt)
    # ft_exp = acf_test1(t) 
    # freq = np.fft.fftfreq(int(t_max/dt)*2, d=dt)
    # t_sim, ft_sim = twoside_psd2acf(freq, spec_test1)

    # axes[0].plot(t,ft_exp,label='real')
    # axes[0].plot(t_sim,ft_sim,label='sim')
    # axes[0].legend()

    # freq, amp = acf2psd(t,acf_test1) 
    # axes[1].plot(freq, spec_test1(freq)[0],label='real')
    # axes[1].plot(freq, amp,label='sim')
    # axes[1].legend()
    # plt.show()

    # f, axes = plt.subplots(1,2)
    # t_max = 0.5
    # dt = 0.001
    # t = np.arange(0,t_max,dt)
    # ft_exp = acf_test2(t) 
    # freq = np.fft.fftfreq(int(t_max/dt)*2, d=dt)
    # t_sim, ft_sim = twoside_psd2acf(freq, spec_test2)

    # axes[0].plot(t,ft_exp,'-o',label='real')
    # axes[0].plot(t_sim,ft_sim,'-*', label='sim')
    # axes[0].legend()

    # freq, amp = acf2psd(t,acf_test2) 
    # axes[1].plot(freq, spec_test2(freq)[0],'-o',label='real')
    # axes[1].plot(freq, amp,'-*',label='sim')
    # axes[1].legend()
    # plt.show()

    f, axes = plt.subplots(1,2)
    t_max = 0.5
    dt = 0.001
    t = np.arange(0,t_max,dt)
    ft_exp = acf_test3(t) 
    freq = np.fft.fftfreq(int(t_max/dt)*2, d=dt)
    t_sim, ft_sim = twoside_psd2acf(freq, spec_test3)

    axes[0].plot(t,ft_exp,label='real')
    axes[0].plot(t_sim,ft_sim, label='sim')
    axes[0].legend()

    freq, amp = acf2psd(t,acf_test3) 
    axes[1].plot(freq, spec_test2(freq)[0],label='real')
    axes[1].plot(freq, amp,label='sim')
    axes[1].legend()
    plt.show()


    # print(y.shape)
    # # t, eta, y= deterministic_lin_sdof(Hs,Tp,T,dt,seed=seed)
    # # print t.shape, eta.shape
    # numPts_T = int(nextpow2(T/dt)) ## Number of points in Time domain
    # numPts_F = int(numPts_T/2+1)
    # df = 1.0/(numPts_T * dt)
    # f = np.arange(1,numPts_F) * df
    # JS = JONSWAP(Hs,Tp,f)
    # JS_area = np.sum(JS*df)

    # # H = np.ones(f.shape)
    # H = transfer_func(f)

    # # f, JS = JONSWAP(Hs,Tp,1.0/dt, 1.0/T)
    # # H = transfer_func(f)
    
    # ## Truncate time series
    # tStart = int(100.0/dt)
    # t = t[tStart:]
    # eta = eta[tStart:]
    # y = y[tStart:]

    # nperseg = int(len(eta)/5.0) # Determine length of each segment, ref to matlab pwelch
    # nfft = nextpow2(len(eta))
    # f_eta, psd_eta = sig.welch(eta, 1.0/dt,nperseg = nperseg, nfft=nfft)
    # f_y, psd_y = sig.welch(y, 1.0/dt,nperseg = nperseg,nfft=nfft)

    # numPts_T = int(nextpow2(T/dt)) ## Number of points in Time domain
    # numPts_F = numPts_T
    # df      = 1.0/(numPts_T * dt)
    # f       = np.arange(1,numPts_F) * df

    # H = transfer_func(f)
    # f, JS = spec_jonswap(Hs,Tp,f)
    # plt.clf()
    # fig = plt.figure()
    # ax1 = fig.add_subplot(221)
    # ln1 = ax1.plot(f, JS,'-b', label='JONSWAP $S(f)$')
    # ax2 = ax1.twinx()
    # ln2 = ax2.plot(f, H,'-g', label='Transfer function $H(f)$')
    # lns = ln1+ln2
    # labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, loc=0, fontsize="xx-small")
    # # ax1.set_xlabel("Frequencey f(Hz)")
    # plt.xlim((0,0.3))
    # # ax1.set_ylabel(r"$S(f), m^2/Hz$")
    # # ax2.set_ylabel(r"$H(f), m^2/Hz$")
    # plt.title("Spectrum")


    
    # ax = fig.add_subplot(222)
    # ax.plot(y[:,0],y[:,1])
    # plt.title('Wave Elevation $\eta(t)$')
    # plt.xlim((100,400))


    # ax1 = fig.add_subplot(223)
    # ln1 = ax1.plot(f_eta, psd_eta,'-b', label='Wave elevation')
    # ax2 = ax1.twinx()
    # ln2 = ax2.plot(f_y, psd_y,'-g', label='System response')
    # lns = ln1 + ln2
    # labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, loc=0, fontsize="small")
    # ax1.set_xlabel("Frequencey f(Hz)")
    # plt.xlim((0,0.3))
    # # plt.title("Power spectrum density")

    # ax = fig.add_subplot(224)
    # ax.plot(y[:,0],y)
    # ax.set_xlabel(r"time (sec)")
    # plt.title('Response $y(t)$')
    # plt.xlim((100,400))
    # #plt.savefig('../Figures/deterministic_lin_sdof.eps')
    # plt.show()


if __name__ == "__main__":
    main()

