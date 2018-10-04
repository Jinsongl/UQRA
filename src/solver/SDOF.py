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

def nextpow2(x):
    return 2**(int(x)-1).bit_length()

def JONSWAP(Hs,Tp,f):
    """ JONSWAP wave spectrum, IEC 61400-3
    Hs: significant wave height, m
    Tp: wave peak period, sec
    f: frequencies to be sampled, hz 
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

def TransferFun(f,f_n=0.15, zeta=0.1):
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

def SDOF(Hs, Tp, T=int(1e3), dt=0.1, seed=[0,100]):
    """
    Dynamics of deterministic SDOF system with given inputs: Hs, Tp
    (Hs,Tp): Environment variables
    T: Simulation duration in seconds
    dt: Simulation time step, default=0.1
    seed=[bool, int], if seed[0]==True, seed is fixed with seed[1]
    """
    if seed[0]:
        np.random.seed(int(seed[1]))
    else:
        pass

    # print ">>> Single DOF system with linear wave: Hs=", Hs, ", Tp=", Tp
    numPts_T = int(nextpow2(T/dt)) ## Number of points in Time domain
    # numPts_F = int(numPts_T/2+1)
    numPts_F= numPts_T
    df      = 1.0/(numPts_T * dt)
    f       = np.arange(1,numPts_F) * df
    t       = np.arange(1,numPts_F) * dt
    JS      = JONSWAP(Hs,Tp,f)
    JS_area = np.sum(JS*df)

    # H = np.ones(f.shape)
    H = TransferFun(f)

    ampf= np.sqrt(2*JS*df) # amplitude
    theta = np.random.uniform(-np.pi, np.pi, len(f))
    # print "  > Random phases:\n  ", theta[0:5],'...'
    etaf= ampf * np.exp(1j*theta)
    yf  = etaf * H 
    etat= np.fft.ifft(etaf).real * numPts_F
    y   = np.fft.ifft(yf).real * numPts_F 

    # print ">>> System response Done!" 
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

def main(Hs=12.5,Tp=15.3,T=int(1e3)):
    T=1000
    dt = 0.01
    seed = [1,100]
    t, eta, y= SDOF(Hs,Tp,T,dt,seed=seed)
    # print t.shape, eta.shape
    numPts_T = int(nextpow2(T/dt)) ## Number of points in Time domain
    numPts_F = int(numPts_T/2+1)
    df = 1.0/(numPts_T * dt)
    f = np.arange(1,numPts_F) * df
    JS = JONSWAP(Hs,Tp,f)
    JS_area = np.sum(JS*df)

    # H = np.ones(f.shape)
    H = TransferFun(f)

    # f, JS = JONSWAP(Hs,Tp,1.0/dt, 1.0/T)
    # H = TransferFun(f)
    
    ## Truncate time series
    tStart = int(100.0/dt)
    t = t[tStart:]
    eta = eta[tStart:]
    y = y[tStart:]

    nperseg = int(len(eta)/5.0) # Determine length of each segment, ref to matlab pwelch
    nfft = nextpow2(len(eta))
    f_eta, psd_eta = sig.welch(eta, 1.0/dt,nperseg = nperseg, nfft=nfft)
    f_y, psd_y = sig.welch(y, 1.0/dt,nperseg = nperseg,nfft=nfft)

    plt.clf()
    fig = plt.figure()

    ax1 = fig.add_subplot(221)
    ln1 = ax1.plot(f, JS,'-b', label='JONSWAP $S(f)$')
    ax2 = ax1.twinx()
    ln2 = ax2.plot(f, H,'-g', label='Transfer function $H(f)$')
    lns = ln1+ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0, fontsize="xx-small")
    # ax1.set_xlabel("Frequencey f(Hz)")
    plt.xlim((0,0.3))
    # ax1.set_ylabel(r"$S(f), m^2/Hz$")
    # ax2.set_ylabel(r"$H(f), m^2/Hz$")
    plt.title("Spectrum")


    ax = fig.add_subplot(222)
    ax.plot(t,eta)
    plt.title('Wave Elevation $\eta(t)$')
    plt.xlim((100,400))


    ax1 = fig.add_subplot(223)
    ln1 = ax1.plot(f_eta, psd_eta,'-b', label='Wave elevation')
    ax2 = ax1.twinx()
    ln2 = ax2.plot(f_y, psd_y,'-g', label='System response')
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0, fontsize="small")
    ax1.set_xlabel("Frequencey f(Hz)")
    plt.xlim((0,0.3))
    # plt.title("Power spectrum density")

    ax = fig.add_subplot(224)
    ax.plot(t,y)
    ax.set_xlabel(r"time (sec)")
    plt.title('Response $y(t)$')
    plt.xlim((100,400))
    #plt.savefig('../Figures/SDOF.eps')
    plt.show()


if __name__ == "__main__":
    main()

