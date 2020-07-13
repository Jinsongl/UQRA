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

import museuq
from museuq.solver._solverbase import SolverBase
import os, numpy as np, scipy as sp
# from scipy.integrate import odeint
# from scipy.optimize import brentq
from .PowerSpectrum import PowerSpectrum
from museuq.environment import Kvitebjorn 
from tqdm import tqdm
import scipy.stats as stats


class surge_model(SolverBase):
    """
    Solving linear oscillator in frequency domain
        m x'' + c x' + k x = f 
    =>    x'' + 2*zeta*omega_n x' + omega_n**2 x = 1/m * f
    where, omega_n = sqrt(k/m), zeta = c/(2*sqrt(m*k))
    default value: omega_n = 1/pi Hz (2 rad/s), zeta = 0.05

    """
    def __init__(self, m=1, c=0.1/np.pi, k=1.0/np.pi/np.pi, environment=None,ltf=None, qtf=None,**kwargs):
        """
        excitation:
            1. None: free vibration case, equivalent to set excitation=0
            2. np.scalar: constant external force
            3. np.array of shape (2,): external force time series,[ t, f = excitation ], may need to interpolate
            4. string   : name of spectrum to be used
            5. callable() function
            
        f0: part of external force
        kwargs, dictionary, spectrum definitions for the input excitation functions

        """
        super().__init__()
        np.random.seed(100)
        seeds_st_cand   = np.random.randint(0, int(2**31-1), size=1000000)
        self.name       = 'Surge Model'
        self.nickname   = 'Surge'
        self.tmax       = kwargs.get('time_max', 100)
        self.tmax       = kwargs.get('tmax', 100)
        self.dt         = kwargs.get('dt', 0.01)
        self.t_transit  = kwargs.get('t_transit', 0)
        self.out_stats  = kwargs.get('out_stats', ['mean', 'std', 'skewness', 'kurtosis', 'absmax', 'absmin', 'up_crossing'])
        self.seeds_idx  = kwargs.get('phase', [0,])
        self.seeds_st   = [seeds_st_cand[idx] for idx in self.seeds_idx] 
        self.n_short_term= len(self.seeds_st) 
        self.out_responses= kwargs.get('out_responses', 'ALL')

        self.dict_rand_params = {}
        self.params_dist_names= 'None'
        self.is_param_rand    = self._validate_mck(m,c,k)
        self.environment      = self._validate_env(environment)
        self.ltf_w, self.ltf_c= self._validate_ltf(ltf) ## linear transfer function
        self.qtf_w, self.qtf_c= self._validate_qtf(qtf) ## quadratic transfer function
        self.ndim             = sum(self.is_param_rand)
        self.nparams          = np.size(self.is_param_rand)

    def __str__(self):
        message1 = 'Single Degree of Fredom Oscillator: \n' +\
                    '   - {:<25s} : {}\n'.format('tmax'  , self.tmax) +\
                    '   - {:<25s} : {}\n'.format('dt'    , self.dt) +\
                    '   - {:<25s} : {}\n'.format('n_short_term'  , self.n_short_term) +\
                    '   - {:<25s} : {}\n'.format('out_responses'  , self.out_responses) +\
                    '   - {:<25s} : {}\n'.format('out_stats'  , self.out_stats) 

        keys   = list(self.dict_rand_params.keys())
        value_names = [] 
        for ivalue in self.dict_rand_params.values():
            try:
                value_names.append(ivalue.name)
            except AttributeError:
                value_names.append(ivalue.dist.name)
        message2 = '   - {} : {}\n'.format(keys, value_names)
        message = message1 + message2
        return message

    def run(self, x, save_raw=False, save_qoi=False, seeds_st=None, out_responses=None, data_dir=None):
        """
        run linear_oscillator:
        Arguments:
            x, power spectrum parameters, ndarray of shape (n_parameters, nsamples)

        """
        seeds_st = self.seeds_st if seeds_st is None else seeds_st
        seeds_st = [seeds_st,] if np.ndim(seeds_st) == 0 else seeds_st
        n_short_term    = np.size(seeds_st) 
        out_responses   = self.out_responses if out_responses is None else out_responses
        data_dir        = os.getcwd() if data_dir is None else data_dir

        x = np.array(x.T, copy=False, ndmin=2)
        y_stats = []
        for iseed_idx, iseed in zip(self.seeds_idx, seeds_st):
            pbar_x  = tqdm(x, ascii=True, ncols=80, desc="    - {:d}/{:d} ".format(iseed_idx, n_short_term))
            ### Note that xlist and ylist will be tuples (since zip will be unpacked). 
            ### If you want them to be lists, you can for instance use:
            if save_raw:
                y_raw_, y_stats_ = map(list, zip(*[self._linear_oscillator(ix, random_seed=iseed,
                    out_responses=out_responses, ret_raw=True) for ix in pbar_x]))
                filename = '{:s}_yRaw_nst{:d}'.format(self.nickname,iseed_idx)
                np.save(os.path.join(data_dir, filename), np.array(y_raw_))
            else:
                y_stats_ = [self._linear_oscillator(ix, random_seed=iseed, 
                    out_responses=out_responses, ret_raw=False) for ix in pbar_x]

            if save_qoi:
                filename = '{:s}_yQoI_nst{:d}'.format(self.nickname,iseed_idx)
                np.save(os.path.join(data_dir, filename), np.array(y_stats_))

            y_stats.append(y_stats_)
        return np.array(y_stats)

    def _linear_oscillator(self, x, random_seed=None, ret_raw=False, out_responses='ALL'):
        """
        Solving linear oscillator in frequency domain
        m x'' + c x' + k x = f => 

        f: frequency in rad/s
        t: array, A sequence of time points for which to solve for y. 
        args, tuple, oscillator arguments in order of (mass, damping, stiffness) 
        kwargs, dictionary, spectrum definitions for the input excitation functions
        """
        if len(x) != self.nparams:
            raise ValueError("_linear_oscillator: Expecting {:d} parameters but {:d} given".format(self.nparams, len(x)))

        ## output time index
        time_out = np.arange(0, int(round((self.tmax + self.t_transit)/self.dt)+1)) * self.dt
        tmin, tmax, dt = time_out[0], time_out[-1], time_out[1]-time_out[0]
        ## frequency index
        if self.ltf_w is None:
            fft_freq_min, fft_freq_max, fft_freq_dw = 0, np.pi/dt, 2*np.pi/tmax 
            fft_freq_pos= np.arange(int(round(fft_freq_max/fft_freq_dw) +1)) * fft_freq_dw
            fft_tmin, fft_tmax, fft_dt = tmin, tmax, dt 
            fft_time= np.arange(fft_freq_pos.size * 2 - 1) * fft_dt 
            psd_w   = fft_freq_pos
        else:
            ltf_w_min, ltf_w_max, ltf_dw = self.ltf_w[0], self.ltf_w[-1], self.ltf_w[1] - self.ltf_w[0]
            if 2*np.pi/ltf_dw < tmax:
                raise ValueError('Surge Model: frequency step (dw) must smaller than 2*pi/T to return time series longer than T')
            fft_freq_min, fft_freq_max, fft_freq_dw = 0, max(np.pi/dt, ltf_w_max), ltf_dw 
            fft_freq_pos= np.arange(int(round(fft_freq_max/fft_freq_dw) +1)) * fft_freq_dw
            fft_tmin, fft_tmax, fft_dt = 0, 2*np.pi/fft_freq_dw, np.pi/fft_freq_max 
            fft_time= np.arange(fft_freq_pos.size * 2 - 1) * fft_dt 
            psd_w   = self.ltf_w

        ##--------- oscillator properties -----------
        params_mck, params_env = x[:3], x[3:]
        m,c,k = params_mck 
        np.random.seed(random_seed)
        theta       = stats.uniform.rvs(-np.pi, 2*np.pi, size=psd_w.size)
        spectrum_env= self.environment_psd(params_env, psd_w)
        env_A       = np.sqrt(2*spectrum_env.dw * spectrum_env.pxx)/2 * np.exp(-1j*theta) 
        frc_lin_A   = env_A * self.ltf_c
        H_sys       = k - m*psd_w**2 + 1j*c* psd_w 
        RAO         = self.ltf_c / H_sys
        response_A  = RAO * env_A

        time_start, time_end = int(round(self.t_transit/fft_dt)), int(round(tmax/fft_dt))
        y_raw   = [] 
        y_stats = [] 

        museuq.blockPrint()
        if 0 in out_responses:
            t0     = fft_time[time_start:time_end]
            istats = museuq.get_stats(t0, out_stats=self.out_stats, axis=0)
            y_stats.append(istats)
            if ret_raw:
                y_raw.append(time_out)

        if 1 in out_responses:
            #--------- environment time series -----------
            fft_A   = np.zeros(fft_freq_pos.size, dtype=complex)
            fft_A   = self._assign_value(fft_freq_pos, fft_A, psd_w, env_A)
            fft_A   = np.append(fft_A, np.flip(np.conj(fft_A[1:])))
            fft_eta = np.fft.ifft(fft_A * fft_A.size)
            eta     = fft_eta[time_start:time_end]
            istats  = museuq.get_stats(eta, out_stats=self.out_stats, axis=0)
            y_stats.append(istats)
            if ret_raw:
                eta = np.interp(time_out, fft_time, fft_eta)
                y_raw.append(eta)

        if 2 in out_responses:
            #--------- excitation time series -----------
            fft_A   = np.zeros(fft_freq_pos.size, dtype=complex)
            fft_A   = self._assign_value(fft_freq_pos, fft_A, psd_w, frc_lin_A)
            fft_A   = np.append(fft_A, np.flip(np.conj(fft_A[1:])))
            fft_frc = np.fft.ifft(fft_A * fft_A.size)
            x_t     = fft_frc[time_start:time_end]
            istats  = museuq.get_stats(x_t, out_stats=self.out_stats, axis=0)
            y_stats.append(istats)
            if ret_raw:
                x_t = np.interp(time_out, fft_time, fft_frc)
                y_raw.append(x_t)

        if 3 in out_responses:
            ##--------- response time series -----------
            fft_A   = np.zeros(fft_freq_pos.size, dtype=complex)
            fft_A   = self._assign_value(fft_freq_pos, fft_A, psd_w, response_A)
            fft_A   = np.append(fft_A, np.flip(np.conj(fft_A[1:])))
            fft_rsp = np.fft.ifft(fft_A * fft_A.size)
            y_t     = fft_rsp[time_start:time_end]
            istats  = museuq.get_stats(y_t, out_stats=self.out_stats, axis=0)
            y_stats.append(istats)
            if ret_raw:
                y_t = np.interp(time_out, fft_time, fft_rsp)
                y_raw.append(y_t)

        museuq.enablePrint()
        # y_stats = museuq.get_stats(y_raw, out_responses=out_responses, out_stats=self.out_stats, axis=0) 
        if ret_raw:
            y_raw = np.vstack(y_raw).T
            return y_raw, y_stats
        else:
            return y_stats

        # ampf    = np.sqrt(pxx*dw) # amplitude
        # eta_fft_coeffs = ampf * np.exp(1j*theta)
        # eta = np.fft.ifft(np.roll(eta_fft_coeffs,ntime_steps+1)) *(2*ntime_steps+1)
        # eta = np.roll(eta,ntime_steps).real # roll back to [-T, T], IFFT result should be real, imaginary part is very small ~10^-16

        # time, eta = time[ntime_steps:2*ntime_steps+1], eta[ntime_steps:2*ntime_steps+1]
        
        # spectrum_frc= self.force_psd() 
        # spectrum_response   = self.response_psd(params_mck, spectrum_frc)

        # t0, x_t, f_hz_x, theta_x = spectrum_frc.gen_process(t=self.t, phase=theta_x)
        # omega   = 2*np.pi*f_hz_x
        # A, B    = k - m * omega**2 , c * omega
        # theta_y =  np.arctan(-(A * np.tan(theta_x) + B) / (A - B * np.tan(theta_x)))
        # t1, y_t, f_hz_y, theta_y = spectrum_response.gen_process(t=self.t, phase=theta_y)
        # assert np.array_equal(t0,t1)
        # assert np.array_equal(f_hz_x,f_hz_y)
        # assert np.array_equal(self.t,t1)

            
    def map_domain(self, u, u_cdf):
        """
        mapping random variables u from distribution dist_u (default U(0,1)) to self.distributions 
        Argument:
            u and dist_u
        """

        if not isinstance(u_cdf, np.ndarray):
            u, dist_u = super().map_domain(u, u_cdf) ## check if dist from stats and change to list [dist,]
            u_cdf     = np.array([idist.cdf(iu) for iu, idist in zip(u, dist_u)])
            u_cdf[u_cdf>0.99999] = 0.99999
            u_cdf[u_cdf<0.00001] = 0.00001

        assert (u_cdf.shape[0] == self.ndim), '{:s} expecting {:d} random variables, {:s} given'.format(self.name, self.ndim, u_cdf.shape[0])
        x = []
        i = 0
        ### maping mck values
        for ikey in ['m', 'c', 'k']:
            try:
                x_ = self.dict_rand_params[ikey].ppf(u_cdf[i])
                x.append(x_)
                i += 1
            except KeyError:
                x_ = getattr(self, ikey) * np.ones((1,u_cdf.shape[1]))
                x.append(x_)
        ### maping env values
        try:
            x_ = self.dict_rand_params['env'].ppf(u_cdf[i:])
            if x_.shape[1] ==1:
                x_ = np.repeat(x_, u_cdf.shape[1], axis=1)
            elif x_.shape[1] == u_cdf.shape[1]:
                pass
            else:
                raise ValueError('map_domain: returned x shape not match u shape: {}!= {}'.format(x.shape, u.shape))
            x.append(x_)
        except KeyError:
            ### env is None
            pass
        x = np.vstack(x)
        if x.shape[0] != self.nparams:
            raise ValueError('map_domain: expecting {:d} parameters but only return {:d}'.format(self.nparams, x.shape[0]))
        return x

    # def random_params_mean

    def _validate_mck(self, m, c, k):
        """
        mck vould either be sclars or distributions from scipy.stats
        """

        is_param_rand = []
        if museuq.isfromstats(m):
            self.dict_rand_params['m'] = m
            is_param_rand.append(True)
        elif np.isscalar(m):
            is_param_rand.append(False)
        else:
            raise ValueError('SDOF: mass value type error: {}'.format(self.m))

        if museuq.isfromstats(c):
            self.dict_rand_params['c'] = c
            is_param_rand.append(True)
        elif np.isscalar(c):
            is_param_rand.append(False)
        else:
            raise ValueError('SDOF: dampling value type error: {}'.format(self.m))

        if museuq.isfromstats(k):
            self.dict_rand_params['k'] = k
            is_param_rand.append(True)
        elif np.isscalar(k):
            is_param_rand.append(False)
        else:
            raise ValueError('SDOF: stiffness value type error: {}'.format(self.m))

        self.m = m
        self.c = c
        self.k = k
        return is_param_rand

    def _validate_env(self, env):
        """
        env must be an object of museuq.Environemnt or None
        """
        if env is None:
            self.params_dist_names = 'EnvNone'
        elif isinstance(env, museuq.EnvBase):
            self.dict_rand_params['env'] = env 
            self.params_dist_names = '_'.join(env.name)
            for is_rand in env.is_arg_rand:
                self.is_param_rand.append(is_rand)
        else:
            raise ValueError('SDOF: environment type {} not defined'.format(type(self.environment)))
        return env 

    def _validate_ltf(self, ltf):
        if isinstance(ltf, np.ndarray):
            w_rad, ltf_cval = ltf 
            w_rad = np.abs(w_rad)
        elif ltf is None:
            w_rad, ltf_cval = None, 1
        else:
            raise NotImplementedError
        return w_rad, ltf_cval 

    def _validate_qtf(self, qtf):
        if isinstance(qtf, np.ndarray):
            w_rad, qtf_cval = qtf 
        elif qtf is None:
            w_rad, qtf_cval = None, 1
        else:
            raise NotImplementedError
        return w_rad, qtf_cval 

    # def _validate_ltf(self, trans_func):
        # if isinstance(trans_func, np.ndarray):
            # spectrum_frc = PowerSpectrum('ltf') 
            # w_rad, trans_func_cval = trans_func
            # pxx = abs(trans_func_cval)**2/(w_rad[1]-w_rad[0])
            # spectrum_frc.set_density(w_rad, pxx)
            # spectrum_frc.tf_cval = trans_func_cval
        # elif trans_func is None:
            # spectrum_frc = PowerSpectrum('ltf') 
            # spectrum_frc.tf_cval = 1 
            # # w_rad = self.w_rad
            # trans_func_cval = w_rad/w_rad
            # pxx = abs(trans_func_cval)**2/(w_rad[1]-w_rad[0])
            # spectrum_frc.set_density(w_rad, pxx)
        # else:
            # raise NotImplementedError
        # return spectrum_frc




    def generate_samples(self, n, random_seed=None):
        n = int(n)
        x = []
        np.random.seed(random_seed)
        ### mck samples 
        i = 0
        for ikey in ['m', 'c', 'k']:
            try:
                x_ = self.dict_rand_params[ikey].rvs(size=(1,n))
                x.append(x_)
                i += 1
            except KeyError:
                x_ = getattr(self, ikey) * np.ones((1,n))
                x.append(x_)

        ### env samples 
        try:
            x_ = self.dict_rand_params['env'].rvs(size=n)
            x_ = np.array(x_, ndmin=2)
            x.append(x_)
        except KeyError:
            pass

        x= np.vstack(x)
        return x 

    def environment_psd(self, x, w_rad):
        """
        Return the psd estimator of environment (e.g. wave elevation) on the right hand side 
        parameters:
            x: power spectrum parameters
            w_rad: frequency in rad/s
            
        Returns:
            a function take scalar argument

        """
        if self.environment is not None:
            try:
                spectrum= PowerSpectrum(self.environment.spectrum, *x)
                density = spectrum.cal_density(w_rad)
            except AttributeError as e:
                print('spectrum {:s} is not defined'.format(self.environment.spectrum))
        else:
            raise NotImplementedError

        return spectrum

        # if self.excitation is None:
            # f = lambda t: t * 0 
            # raise ValueError('SDOF is sovled in frequency domain, None type external force need to be transformed into frequency domain first') 

        # elif isinstance(self.excitation, (int, float, complex)) and not isinstance(x, bool):
            # f = lambda t: t/t * self.excitation 
            # raise ValueError('SDOF is sovled in frequency domain, constant external force need to be transformed into frequency domain first') 

        # elif isinstance(self.excitation, np.ndarray):
            # assert np.ndim(self.excitation) == 2
            # assert self.excitation.shape[0] == 2
            # t, y= self.excitation
            # f   = sp.interpolate.interp1d(t, y,kind='cubic')
            # raise ValueError('SDOF is sovled in frequency domain, time series external force need to be transformed into frequency domain first') 

        # elif isinstance(self.excitation, str):
            # spectrum= PowerSpectrum(self.excitation, *x)
            # density = spectrum.cal_density(w_rad)

        # else:
            # raise NotImplementedError


    def response_psd(self, mck, spectrum_frc):
        """
        Return the psd estimator of both input and output signals at frequency f for specified PowerSpectrum with given parameters x

        Arguments:
            f: frequency in rad/s
            x: PowerSpectrum parameters
        Returns:
            PowerSpectrum object of input and output signal
        """
        m,c,k = mck
        f_hz  = spectrum_frc.w_rad## f_hz in rad/s
        H_square = 1.0/np.sqrt( (k-m*f_hz**2)**2 + (c*f_hz)**2)
        psd_y = H_square * spectrum_frc.pxx
        spectrum_response = PowerSpectrum()
        spectrum_response.set_density(spectrum_frc.f_hz, psd_y, sides=spectrum_frc.sides)
        return spectrum_response 


    def _assign_value(self, fft_w, fft_A, w, Sw):

        dw = fft_w[1]-fft_w[0]
        for iw, iSw in zip(w, Sw):
            idx = np.where(abs(fft_w -iw) < dw/100)
            if idx[0].size > 0:
                fft_A[idx[0][0]] = iSw
            else:
                raise ValueError('Frequency {} not found in FFT'.format(iw))
        return fft_A
