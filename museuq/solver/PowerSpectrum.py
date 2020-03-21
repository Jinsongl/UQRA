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
    => Sx(f) = \int R(tau) exp(-2*pi*j*f*tau) d tau

    Properties:
    1. since Sx is an average of the magnitude squared of the Fourier transform
        Sx(f) = lim T->inf 1/T E[|X_T(f)|^2],
        where X_T(f) is the Fourier transform of x(t), X_T(f) = int x(t) exp(-2*pi*j*f*t) dt

    2. Sx(-f) = Sx(f)
    3. Dual relationship: R(tau) = int Sx(f) exp(2*pi*j*f*tau) d tau
    4. At tau=0, R(0) = E[x(t)^2] = int Sx(f) df, "variance" = Area under curve
    5. Parseval's Identity:
        Assume x(t) ergodic in the autocorrelation, i.e. R(tau) = E[x(t) x(t+tau)] = \lim_{T->inf} 1/T \int_{-T/2} ^{T/2} x(t) x(t+tau) dt
        for any signal x(t):
        lim_{T->inf} 1/T \int _{-T/2} ^{T/2} x(t)^2 dt = \int_{-inf} ^{inf} Sx(f) df

    """

    def __init__(self, name, *args):
        self.name   = name
        self.f      = None 
        self.pxx    = None 
        self.sides  = 'single' 
        self.args   = args
        try:
            spectrum_func = getattr(spectrums, self.name.lower())
            sig =inspect.signature(spectrum_func) 
            self.ndim = len(sig.parameters) - 1
        except AttributeError:
            self.ndim = None

    def set_psd(self, f, pxx, sides='single'):
        assert f.shape == pxx.shape, 'f and pxx should have same shape. f.shape={}, pxx.shape={}'.format(f.shape, pxx.shape)
        self.f      = f
        self.pxx    = pxx
        self.sides  = sides

    def get_pxx(self, f):
        spectrum_func = getattr(spectrums, self.name.lower())
        f, pxx = spectrum_func(f, *self.args)
        assert f.shape == pxx.shape, 'f and pxx should have same shape. f.shape={}, pxx.shape={}'.format(f.shape, pxx.shape)
        self.f = f if self.f is None else self.f
        self.pxx = pxx if self.pxx is None else self.pxx
        return pxx

    def get_acf(self):
        """
        Return Autocorrelation function corresponding to the specified power spectrum assuming stationary process by Inverse Fourier Transform (R=IFFT(PSD))
        PowerSpectrum.sides = 'double'
        """
        # ---------------------------------------------------------
        #                   |   Range       |   delta step
        # Time domain       | [-tmax, tmax] | dt = given
        # Frequency domain  | [-fmax, fmax] | df = 1/(2(tmax))  
        #                   | fmax = 1/(2*dt)
        # ---------------------------------------------------------
        # for single side psd, transform to double side and return corresponding f, pxx
        # Transformation shouldn't change obj.f, obj.pxx
        if not self._is_symmetric(self.f, self.pxx):
            psd_f, psd_pxx = self.psd_single2double()
        else:
            psd_f   = self.f
            psd_pxx = self.pxx

        nfrequencies    = psd_f.size 
        fmin, fmax, df  = self.f[0], self.f[-1], self.f[1]-self.f[0]
        dt              = 1.0/(2*fmax)
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
            psd_f, psd_pxx
        """
        
        t, acf_t= self._correct_acf_format(t, acf_t)
        dt      = t[1]-t[0]
        psd_pxx = np.fft.fft(acf_t).real * dt
        psd_f   = np.fft.fftfreq(t.size,d=dt)
        # Since we know acf function is even, fft of acf_tau should only contain real parts
        # psd_pxx = np.sqrt(acf_fft.real**2 + acf_fft.imag**2) * dtau
        
        # reorder frequency from negative to positive ascending order
        psd_pxx = np.array([x for _,x in sorted(zip(psd_f,psd_pxx))])
        psd_f   = np.array([x for _,x in sorted(zip(psd_f,psd_f))])
        self.f  = psd_f
        self.pxx= psd_pxx
        self.sides='double'
        return psd_f, psd_pxx

    def gen_process(self,t=None):
        """
        Generate Gaussian time series for given spectrum with IFFT method
        Note: For one side psd, one need to create IFFT coefficients for negative frequencies to use IFFT method. 
            Single side psd need to be divide by 2 to create double side psd, S1(self.f) = S1(-self.f) = S2(self.f)/2
            Phase: theta(self.f) = -theta(-self.f) 

        Arguments:
            self.f: ndarry, frequency in Hz
            pxx: pwd values corresponding to self.f array. 
        Return:
            t: time index, start 0 to tmax 
            etat: surface wave time series
            psd_f: frequencies of spectral density 
            eta_fft_coeffs: surface wave power spectral denstiy

        Features need to add:
            1. douebl side psd
            2. padding zero values for self.pxx when self.f[0] < 0 and self.f is not symmetric
            3. gen_process arguments should be time, not frequency , sounds more reasonable.
                if this feature need to be added, interpolation of self.f may needed.

        # numpy.fft.ifft(a, n=None, axis=-1, norm=None)
        #   The input should be ordered in the same way as is returned by fft, i.e.,
        #     - a[0] should contain the zero frequency term,
        #     - a[1:n//2] should contain the positive-frequency terms,
        #     - a[n//2 + 1:] should contain the negative-frequency terms, in increasing order starting from the most negative frequency.
        """
        # ---------------------------------------------------------
        #                   |   Range       |   delta step
        # Time domain       | [-tmax, tmax] | dt = given
        # Frequency domain  | [-fmax, fmax] | df = 1/(2(tmax))  
        #                   | fmax = 1/(2*dt)
        # ---------------------------------------------------------
        if self.sides.lower() == 'single':
            f, pxx= self.psd_single2double()
        else:
            assert self._is_symmetric(self.f, self.pxx)
            f, pxx = self.f, self.pxx

        fmax, df = f[-1]  , f[1]-f[0]
        tmax, dt = 0.5/df , 0.5/fmax
        ntime_steps = f.size//2 #int(fmax/df) same as number of frequencies
        time = np.arange(-ntime_steps,ntime_steps+1) * dt

        theta   = np.random.uniform(-np.pi, np.pi, ntime_steps + 1)
        theta   = np.hstack((-np.flip(theta[1:]),theta)) # concatenation along the second axis
        ampf    = np.sqrt(pxx*df) # amplitude

        eta_fft_coeffs = ampf * np.exp(1j*theta)
        eta = np.fft.ifft(np.roll(eta_fft_coeffs,ntime_steps+1)) *(2*ntime_steps+1)
        eta = np.roll(eta,ntime_steps).real # roll back to [-T, T], IFFT result should be real, imaginary part is very small ~10^-16

        time, eta = time[ntime_steps:2*ntime_steps+1], eta[ntime_steps:2*ntime_steps+1]

        if t:
            if t[1]-t[0] < time[1]-time[0]:
                raise ValueError('Higher frequency needed to achieve time domain resolution dt={:.4f}'.format(t[1]-t[0]))
            if t[-1] > time[-1]:
                raise ValueError('Higher frequency resolution needed to achieve time domain period, T={:.2e}'.format(t[-1]))
            time = t
            eta = np.interp(t, time, eta)
        return time, eta 

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
        Convert single side psd specified by (f, pxx) to double side
        Arguments:
            f   : single side frequency vector, could start from arbitrary value 
            pxx : single side power spectral density (PSD) estimate, pxx, 
        Returns:
            ff  : double side frequency vector
            pxx2: double side power spectral density (PSD) estimate, pxx2
        """
        assert self.f is not None and self.pxx is not None
        f, pxx = self.f, self.pxx
        ## sorting f in ascending order
        f   = f  [f.argsort()]
        pxx = pxx[f.argsort()]
        assert f[0] >= 0 and f[-1] >0 # make sure frequency vector in positive range
        df  = np.mean(f[1:]-f[:-1], dtype=np.float64) # df is the average frequency difference
        N   = len(np.arange(f[-1], 0, -df)) # does not include 0
        pxx2= np.zeros(2*N+1)
        ff  = np.zeros(2*N+1)
        # print(ff.shape, pxx2.shape)
        # 
        # pxx2: [0,|1,2,...,N | N+1,...,2N+1] 
        #       [0,| positive |   negative  ]

        
        ## padding psd from 0 to f[0] with 0
        # first element of pxx2[0] is power at frequency 0. If pxx(0) is not given, 0 is padded
        pxx2[0] = pxx[0] if f[0]==0 else 0
        # Positive frequencies part
        m = pxx.size
        pxx2[N+1-m:N+1] = pxx/2
        # Negative frequencies part
        pxx2[N+1:]  = np.flip(pxx2[1:N+1]) 
        ff[1:N+1]   = np.flip(np.arange(f[-1], 0, -df))
        ff[N+1:]    = -np.arange(f[-1], 0, -df)
        
        ## Reorder psd to [-inf, negative, 0, positive, +inf] 
        pxx2= np.roll(pxx2, N)
        ff  = np.roll(ff, N)

        return ff, pxx2

    def _gen_process_sum(self):
        if self.sides.lower() == 'double':
            raise NotImplementedError('Stay tuned')
        else:
            psd_f, psd_pxx = self.f, self.pxx
            fmax, df = psd_f[-1]  , psd_f[1]-psd_f[0]
            tmax, dt = 0.5/df , 0.5/fmax
            N = int(tmax/dt)
            t = np.arange(-N,N+1) * dt
            # tmax = t[-1]
            theta = np.random.uniform(-np.pi, np.pi, len(psd_f))
            # Direct sum with single side psd
            ampf = np.sqrt(2*psd_pxx*(df)) # amplitude

            # Reshape to matrix operation format
            ampf  = np.reshape(ampf,  (N+1,1))
            psd_f = np.reshape(psd_f, (N+1,1))
            theta = np.reshape(theta, (N+1,1))
            t     = np.reshape(t,     (1, 2*N+1))
            eta = np.sum(ampf * np.cos(2*np.pi*psd_f*t + theta),axis=0)

        return t, eta



