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
from . import spectrums
import inspect

class PowerSpectrum(object):
    """
    Reference: https://www.imft.fr/IMG/pdf/psdtheory.pdf  "The Power Spectral Density and the Autocorrelation"
    Definitions:
    - ACF: for a real, stationary signal x(t), its ACF R is defined as: R(tau) = E[x(t)x(t+tau)]
    - PSD: Fourier transform of R(tau) is called the Power spectral denstity (PSD), Sx/pxx
    => Sx(w_rad) = \int R(tau) exp(-2*pi*j*w_rad*tau) d tau

    Properties:
    1. since Sx is an average of the magnitude squared of the Fourier transform
        Sx(w_rad) = lim T->inf 1/T E[|X_T(w_rad)|^2],
        where X_T(w_rad) is the Fourier transform of x(t), X_T(w_rad) = int x(t) exp(-2*pi*j*w_rad*t) dt

    2. Sx(-w_rad) = Sx(w_rad)
    3. Dual relationship: R(tau) = int Sx(w_rad) exp(2*pi*j*w_rad*tau) d tau
    4. At tau=0, R(0) = E[x(t)^2] = int Sx(w_rad) dw, "variance" = Area under curve
    5. Parseval's Identity:
        Assume x(t) ergodic in the autocorrelation, i.e. R(tau) = E[x(t) x(t+tau)] = \lim_{T->inf} 1/T \int_{-T/2} ^{T/2} x(t) x(t+tau) dt
        for any signal x(t):
        lim_{T->inf} 1/T \int _{-T/2} ^{T/2} x(t)^2 dt = \int_{-inf} ^{inf} Sx(w_rad) dw

    """

    def __init__(self, name=None, *args):
        self.name   = name
        self.sides  = 'single' 
        self.args   = args
        if self.name is None:
            self.ndim = None
        elif isinstance(self.name, str):
            self.name = self.name.lower()
            try:
                self.density_func = getattr(spectrums, self.name.lower())
                sig = inspect.signature(self.density_func) 
                self.ndim = len(sig.parameters) - 1
            except AttributeError as e:
                self.ndim = None
                raise ValueError(e)
        else:
            raise ValueError("'module 'uqra.solver.spectrums' has no attribute '{}'".format(self.name))

    def set_density(self, w_rad, pxx, sides='single'):
        assert w_rad.shape == pxx.shape, 'w_rad and pxx should have same shape. w_rad.shape={}, pxx.shape={}'.format(w_rad.shape, pxx.shape)
        self.w_rad  = w_rad
        self.pxx    = pxx
        self.sides  = sides
        self.dw     = w_rad[1] -w_rad[0]

    def cal_density(self, w_rad):
        w_rad, pxx = self.density_func(w_rad, *self.args)
        assert w_rad.shape == pxx.shape, 'w_rad and pxx should have same shape. w_rad.shape={}, pxx.shape={}'.format(w_rad.shape, pxx.shape)
        self.w_rad= w_rad
        self.pxx  = pxx
        self.dw   = w_rad[1] - w_rad[0]
        return pxx

    def get_acf(self):
        """
        Return Autocorrelation function corresponding to the specified power spectrum assuming stationary process by Inverse Fourier Transform (R=IFFT(PSD))
        PowerSpectrum.sides = 'double'
        """
        # ---------------------------------------------------------
        #                   |   Range       |   delta step
        # Time domain       | [-tmax, tmax] | dt = given
        # Frequency domain  | [-fmax, fmax] | dw = 1/(2(tmax))  
        #                   | fmax = 1/(2*dt)
        # ---------------------------------------------------------
        # for single side psd, transform to double side and return corresponding w_rad, pxx
        # Transformation shouldn't change obj.w_rad, obj.pxx
        if not self._is_symmetric(self.w_rad, self.pxx):
            psd_w, psd_pxx = self.psd_single2double()
        else:
            psd_w   = self.w_rad
            psd_pxx = self.pxx

        nfrequencies    = psd_w.size 
        w_min, w_max, dw= self.w_rad[0], self.w_rad[-1], self.w_rad[1]-self.w_rad[0]
        dt              = 1.0*2*np.pi/(2*w_max)
        acf_ifft        = np.fft.ifft(psd_pxx) / dt
        acf             = np.sqrt(acf_ifft.real**2 + acf_ifft.imag**2)
        t               = np.arange(-nfrequencies//2, nfrequencies//2) * dt
        acf             = np.roll(acf,nfrequencies//2)
        assert t.size == acf.size
        return t, acf

    def from_acf(self, t, acf_t):
        """
        Given auto correlation function acf_tau in [0,t] with dtau, return corresponding power spectral density function
        Process is assumed to be stationary such that acf is just a function of tau and acf is symmetric. e.g. acf(-tau) = acf(tau)
        Arguments:
            tau: time interval, if only [0,tmax] is specified, [-tmax,0) will be pre-appended automatically
                Values of acf at both positive and negative lags (tau) is necessary. When applying fft(data), algorithm assumes data repeat after time interval. If only positive part provided, symmetric information of acf will not be passed ([-tmax,0] will be exactly same as [0, tmax] instead of symmetric acf(tau) = acf(-tau))
            acf: autocorrelation function or values at specified tau 
        Returns:
            Power spectral density function
            psd_w, psd_pxx
        """
        
        t, acf_t= self._correct_acf_format(t, acf_t)
        dt      = t[1]-t[0]
        psd_pxx = np.fft.fft(acf_t).real * dt
        psd_w   = 2*np.pi*np.fft.fftfreq(t.size,d=dt)
        # Since we know acf function is even, fft of acf_tau should only contain real parts
        # psd_pxx = np.sqrt(acf_fft.real**2 + acf_fft.imag**2) * dtau
        
        # reorder frequency from negative to positive ascending order
        psd_pxx = np.array([x for _,x in sorted(zip(psd_w,psd_pxx))])
        psd_w   = np.array([x for _,x in sorted(zip(psd_w,psd_w))])
        self.w_rad  = psd_w
        self.pxx= psd_pxx
        self.sides='double'
        return psd_w, psd_pxx

    def gen_process(self,t=None, phase=None, random_seed=None):
        """
        Generate Gaussian time series for given spectrum with IFFT method
        Note: For one side psd, one need to create IFFT coefficients for negative frequencies to use IFFT method. 
            Single side psd need to be divide by 2 to create double side psd, S1(self.w_rad) = S1(-self.w_rad) = S2(self.w_rad)/2
            Phase: theta(self.w_rad) = -theta(-self.w_rad) 

        Arguments: optional
            t: time index, start 0 to tmax 
            phase: phase uniformly between[-pi, pi]
            random_seed: seed to generate phase if phase is not given
        Return:
            psd_w: frequencies of spectral density 
            eta_fft_coeffs: surface wave power spectral denstiy


        """
        # ---------------------------------------------------------
        #                   |   Range       |   delta step
        # Time domain       | [0, tmax]     | dt = given
        # Frequency domain  | [-fmax, fmax] | dw = 1/tmax  
        #                   | fmax = 1/(2*dt)
        # ---------------------------------------------------------
        if self.sides.lower() == 'single':
            psd_w, pxx= self.psd_single2double()
        else:
            assert self._is_symmetric(self.w_rad, self.pxx)
            psd_w, pxx = self.w_rad, self.pxx

        psd_w_min, psd_w_max, psd_dw = np.amin(abs(psd_w)), np.amax(abs(psd_w)) , psd_w[1]-psd_w[0]
        if t is None:
            ### if t is not given, returned time series will be generated based on specified frequency values in psd_w
            tmin, tmax, dt = 0, 2*np.pi/psd_dw, np.pi/psd_w_max
            fft_freq_min, fft_freq_max, fft_freq_dw = 0, psd_w_max, psd_dw
            fft_freq_pos = np.arange(int(round(fft_freq_max/fft_freq_dw) +1)) * fft_freq_dw
            time = np.arnage(int(round(tmax/dt))) *dt  
        else:
            ### when a time index is given, sampling frequency need to be small enough to cover the period (tmax) and frequency window need to be large enough to have time resolution dt
            tmax, tmin, dt = t[-1], t[0], t[1] - t[0]
            fft_freq_min, fft_freq_max, fft_freq_dw = 0, max(np.pi/dt, psd_w_max), min(psd_dw, 2*np.pi/tmax) 
            fft_freq_pos = np.arange(int(round(fft_freq_max/fft_freq_dw) +1)) * fft_freq_dw
            tmin, tmax, dt = 0, 2*np.pi/fft_freq_dw, np.pi/fft_freq_max 
            time = np.arnage(int(round(tmax/dt))) *dt  

        # ntime_steps = psd_w.size//2 #int(w_max/dw) same as number of frequencies
        # time = np.arange(-ntime_steps,ntime_steps+1) * dt

        psd_w_idx = [int(round(psd_w_min/fft_freq_dw)),int(round(psd_w_max/fft_freq_dw))]
        psd_w_= np.arange(psd_w_idx[0],psd_w_idx[1])*fft_freq_dw
        pxx_  = self.cal_density(psd_w_) 
        if phase is None:
            np.random.seed(random_seed)
            theta = np.random.uniform(-np.pi, np.pi, psd_w_.size)
        else:
            theta = phase[:psd_w_.size]
        psd_A = np.sqrt(2.0*pxx_*fft_freq_dw)/2.0 * np.exp(-1j*theta) * (fft_freq_pos*2-1) # amplitude
        fft_A = np.zeros(fft_freq_pos.size, dtype=complex)
        fft_A[psd_w_idx[0]:psd_w_idx[1]] = psd_A
        fft_A = np.append(fft_A, np.flip(np.conj(fft_A[1:])))
        # eta = np.roll(eta,ntime_steps).real # roll back to [-T, T], IFFT result should be real, imaginary part is very small ~10^-16
        eta = np.fft.ifft(fft_A).real
        assert eta.size == time.size

        if t is not None:
            eta = np.interp(t, time, eta)
            time = t
        return time, eta , psd_w_, theta

    def _is_symmetric(self, x, y):
        """
        Check if the following two symmetric properties are met:
        1. x[i] = -x[i]
        2. y[x] = -y[x]
        """
        x = np.array(x)
        y = np.array(y)
        x = x[x.argsort()]
        y = y[x.argsort()]
        assert x.size == y.size

        if x[0] * x[-1] >= 0: # return false if xmin, xmax same sign
            return False
        n     = x.size+1 if (x.size) %2 else x.size
        bool1 = np.array_equal(x[1:n/2], -np.flip(x[n/2:-1]))
        bool2 = np.array_equal(y[1:n/2], -np.flip(y[n/2:-1]))
        return bool1 and bool2

    def _correct_acf_format(self, t, acf_t):
        """
        Modify (t, acf_t) to standard format corresponding to frequenceis
        t = [0,1,...,n, -n,...,-1] * dt
        2n+1 time steps
        """
        t     = np.array(t)
        t     = t[t.argsort()]
        acf_t = acf_t[t.argsort()]

        ntimesteps = t.size
        ## if not symmetric, following steps are taken:
        # 1. time step dt is taken as the mean value of all time steps
        # 2. corresponding acf values are interpolated 
        if not self._is_symmetric(t, acf_t):
            dt = np.mean(t[1:-1] - t[:-2], dtype=np.float64)
            ntimesteps = np.ceil(max(abs(t))/dt)
            t1 = np.arange(-ntimesteps, ntimesteps) * dt
            acf_t_interp = np.interp(t1, t, acf_t)
            return np.roll(t1, ntimesteps+1), np.roll(acf_t_interp, ntimesteps+1)
        else:
            return t, acf_t

    def psd_single2double(self):
        """
        Convert single side psd specified by (w_rad, pxx) to double side
        Arguments:
            w_rad   : single side frequency vector, could start from arbitrary value 
            pxx : single side power spectral density (PSD) estimate, pxx, 
        Returns:
            ww_rad  : double side frequency vector
            pxx2: double side power spectral density (PSD) estimate, pxx2
        """
        assert self.w_rad is not None and self.pxx is not None
        w_rad, pxx = self.w_rad, self.pxx
        ## sorting w_rad in ascending order
        w_rad= w_rad[w_rad.argsort()]
        pxx = pxx [w_rad.argsort()]
        assert w_rad[0] >= 0 and w_rad[-1] >0 # make sure frequency vector in positive range
        dw      = np.mean(w_rad[1:]-w_rad[:-1], dtype=np.float64) # dw is the average frequency difference
        N       = len(np.arange(w_rad[-1], 0, -dw)) # does not include 0
        pxx2    = np.zeros(2*N+1)
        ww_rad  = np.zeros(2*N+1)
        # print(ww_rad.shape, pxx2.shape)
        # 
        # pxx2: [0,|1,2,...,N | N+1,...,2N+1] 
        #       [0,| positive |   negative  ]

        
        ## padding psd from 0 to w_rad[0] with 0
        # first element of pxx2[0] is power at frequency 0. If pxx(0) is not given, 0 is padded
        pxx2[0] = pxx[0] if w_rad[0]==0 else 0
        # Positive frequencies part
        m = pxx.size
        pxx2[N+1-m:N+1] = pxx/2
        # Negative frequencies part
        pxx2[N+1:]  = np.flip(pxx2[1:N+1]) 
        ww_rad[1:N+1]   = np.flip(np.arange(w_rad[-1], 0, -dw))
        ww_rad[N+1:]    = -np.arange(w_rad[-1], 0, -dw)
        
        ## Reorder psd to [-inf, negative, 0, positive, +inf] 
        pxx2= np.roll(pxx2, N)
        ww_rad  = np.roll(ww_rad, N)

        return ww_rad, pxx2

    def _gen_process_sum(self):
        if self.sides.lower() == 'double':
            raise NotImplementedError('Stay tuned')
        else:
            psd_w, psd_pxx = self.w_rad, self.pxx
            w_max, dw = psd_w[-1]  , psd_w[1]-psd_w[0]
            tmax, dt = 0.5*2*np.pi/dw , 0.5/w_max
            N = int(tmax/dt)
            t = np.arange(-N,N+1) * dt
            # tmax = t[-1]
            theta = np.random.uniform(-np.pi, np.pi, len(psd_w))
            # Direct sum with single side psd
            ampf = np.sqrt(2*psd_pxx*(dw)) # amplitude

            # Reshape to matrix operation format
            ampf  = np.reshape(ampf,  (N+1,1))
            psd_w = np.reshape(psd_w, (N+1,1))
            theta = np.reshape(theta, (N+1,1))
            t     = np.reshape(t,     (1, 2*N+1))
            eta = np.sum(ampf * np.cos(2*np.pi*psd_w*t + theta),axis=0)

        return t, eta



