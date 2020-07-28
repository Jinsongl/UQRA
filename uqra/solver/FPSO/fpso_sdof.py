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

import uqra
from uqra.solver._solverbase import SolverBase
import os, numpy as np, scipy as sp
from tqdm import tqdm
import scipy.io
import multiprocessing as mp
class FPSO(SolverBase):
    """
    Solving linear oscillator in frequency domain
        m x'' + c x' + k x = f 
    =>    x'' + 2*zeta*omega_n x' + omega_n**2 x = 1/m * f
    where, omega_n = sqrt(k/m), zeta = c/(2*sqrt(m*k))
    default value: omega_n = 1/pi Hz (2 rad/s), zeta = 0.05

    """
    def __init__(self, **kwargs):
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
        self.name       = 'FPSO'
        self.nickname   = 'FPSO_SDOF'
        self.dist_name  = 'None'
        self.ndim       = int(2)

        np.random.seed(100)
        seeds_st        = np.random.randint(0, int(2**31-1), size=10000)
        # self.tmax       = kwargs.get('time_max', 100)
        # self.tmax       = kwargs.get('tmax', 100)
        # self.dt         = kwargs.get('dt', 0.01)
        # self.t_transit  = kwargs.get('t_transit', 0)
        # self.out_stats  = kwargs.get('out_stats', ['mean', 'std', 'skewness', 'kurtosis', 'absmax', 'absmin', 'up_crossing'])
        self.seeds_idx  = kwargs.get('phase', [0,])
        self.seeds_st   = [seeds_st[idx] for idx in self.seeds_idx] 
        self.n_short_term= len(self.seeds_st) 
        self.out_responses= kwargs.get('out_responses', 'ALL')
        # self.t          = np.arange(0,int((self.tmax + self.t_transit)/self.dt) +1) * self.dt
        # self.f_hz       = np.arange(len(self.t)+1) *0.5/self.t[-1]
        # self.w_rad      = 2 * np.pi * self.f_hz

        self._load_freq_mat('freq_data.mat')
        self._FD_FPSO_LF()


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
        seeds_st = self.seeds_st if seeds_st is None else seeds_st
        seeds_st = [seeds_st,] if np.ndim(seeds_st) == 0 else seeds_st
        n_short_term    = np.size(seeds_st) 
        out_responses   = self.out_responses if out_responses is None else out_responses
        data_dir        = os.getcwd() if data_dir is None else data_dir

        x = np.array(x.T, copy=False, ndmin=2)
        y_QoI = []
        for iseed_idx, iseed in zip(self.seeds_idx, seeds_st):
            pbar_x  = tqdm(x, ascii=True, ncols=80, desc="    - {:d}/{:d} ".format(iseed_idx, n_short_term))
            ### Note that xlist and ylist will be tuples (since zip will be unpacked). 
            ### If you want them to be lists, you can for instance use:
            if save_raw:
                raise NotImplementedError
                # y_raw_, y_QoI_ = map(list, zip(*[self._linear_oscillator(ix, random_seed=iseed,
                    # out_responses=out_responses, ret_raw=True) for ix in pbar_x]))
                # filename = '{:s}_yRaw_nst{:d}'.format(self.nickname,iseed_idx)
                # np.save(os.path.join(data_dir, filename), np.array(y_raw_))
            else:
                with mp.Pool(processes=mp.cpu_count()) as p:
                    # y_QoI_ = [self._Glimitmax(ix, iseed) for ix in pbar_x]
                    y_QoI_ = np.array(list(tqdm(p.imap(self._Glimitmax , [(ix, iseed) for ix in pbar_x]),ncols=80, total=x.shape[1])))
                    # results = np.array(list(tqdm(p.starmap(self._Glimitmax, [(ix, iseed) for ix in pbar_x]), total=nsamples)))
            if save_qoi:
                filename = '{:s}_yQoI_nst{:d}'.format(self.nickname,iseed_idx)
                np.save(os.path.join(data_dir, filename), np.array(y_QoI_))

            y_QoI.append(y_QoI_)
        return np.array(y_QoI)

    def _run_FPSO(self, x, seed):
        y = self._Glimitmax(x)
        return y

    def _load_freq_mat(self, filename):
        data = scipy.io.loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), filename))
        self.diag_surge = np.squeeze(data['Diag_surge'])
        self.diag_sway  = np.squeeze(data['Diag_sway'])
        self.dw         = np.squeeze(data['dw'])
        self.dw_LF      = np.squeeze(data['dw_LF'])
        self.N          = np.squeeze(data['N'])
        self.N_LF       = np.squeeze(data['N_LF'])
        self.TFv        = np.squeeze(data['TFv'])
        self.w          = np.squeeze(data['w'])
        self.w_LF       = np.squeeze(data['w_LF'])
        self.wmax       = np.squeeze(data['wmax'])
        self.wmin       = np.squeeze(data['wmin'])

    def _FD_FPSO_LF(self):
        fac1=0.585809078570351
        fac2=1

        self.diag_surge  = self.diag_surge*fac1   #0.255554093091961

        self.TFv = self.TFv[0,:] 

        # -----------
        # Mass matrix
        # -----------
        self.M =1e8 

        # ----------------
        # Stiffness matrix
        # ----------------

        self.K = 280000

        # --------------
        # Damping matrix
        # --------------

        self.DR=.05 
        self.B = self.DR*2*np.sqrt(self.K*self.M) 

        # -----------
        # WF analysis
        # -----------

        self.RAO = self.TFv/(-self.w**2 *self.M +1j*self.w *self.B +self.K)

        self.Hx  = 1.0 / ( -self.w_LF**2 * self.M + 1j * self.w_LF * self.B + self.K )

    def _Glimitmax(self, args):


        # ------------------
        # Environmental data
        # ------------------
        Hs, Tp = args[0]
        seed = args[1]

        Snn=self._jonswap(Hs,Tp);

        N =960;
        Nt=125664;

        # ----------
        # Parameters
        # ----------

        np.random.seed(seed)
        x_rand= np.random.normal(loc=0,scale=1.0, size=(1,2*N))
        Re    = x_rand[0,:N];
        Im    = x_rand[0,N:];
        # print(x_rand[:10])
        A=(Re+1j*Im)*np.sqrt(self.dw*Snn);
        # print(A.shape)
        # X_0 =-Nt*np.real(np.fft.ifft(A,Nt));
        # print(' Hs: {:.2f}, 4*sigma: {:.2f}'.format(Hs, 4*np.std(X_0)))
        # -----------
        # WF response
        # -----------

        Z1 = np.zeros((int(self.wmin/self.dw,)))
        Z2 = self.RAO * A
        Z  = np.hstack((Z1,Z2))
        # print(Z.shape)


        # -----------
        # LF response
        # -----------
        X1 = np.zeros((self.N_LF,),dtype=np.complex_)
        for i in range(self.N_LF-1, -1,-1):
            A_Aconj = A[:N-i] * np.conj(A[i:N])
            X1[i] = self.Hx[i] * np.sum( 0.5 * (self.diag_surge[:N-i] + self.diag_surge[i:N]) * A_Aconj)

        Z [:len(X1)] = Z[:len(X1)] + 2 * X1
        X = -Nt * np.real(np.fft.ifft(Z, Nt)) 
        Gmax = np.max(X[:41888]) 
        return Gmax

        # for xx=160:-1:1
        # A_Aconj=A(1:N-xx).*conj(A(xx+1:N));
        # X1(xx+1)=Hx(xx+1)*sum(0.5*(diag_surge(1:N-xx)+diag_surge(xx+1:N)).*A_Aconj);
        # end

        # Z(1:length(X1))=Z(1:length(X1))+2*X1;

        # X=-Nt*real(ifft(Z,Nt,2));

        # Gmax=np.max(X(1:41888));

    def _jonswap(self, Hs, Tp):
        """ JONSWAP wave spectrum, IEC 61400-3
        w: frequencies to be sampled at, rad/s
        Hs: significant wave height, m
        Tp: wave peak period, sec
        """

        w = self.w 
        with np.errstate(divide='ignore'):
            # print "sample frequency: \n", w
            wp    = 2.0*np.pi/Tp
            gamma = 3.3 
            sigma = 0.07 * np.ones(w.shape)
            sigma[w > wp] = 0.09
            
            assert w[0].any() >= 0 ,'Single side power spectrum start with frequency greater or eqaul to 0, w[0]={:4.2f}'.format(w[0])

            JS1 = 5/16 * Hs**2 * Tp *(w/wp)**-5 /2/np.pi
            JS2 = np.exp(-1.25*(w/wp)**-4) * (1-0.287*np.log(gamma))
            JS3 = gamma**(np.exp( -0.5*((w/wp-1)/sigma)**2 ))

            JS1[np.isinf(JS1)] = 0
            JS2[np.isinf(JS2)] = 0
            JS3[np.isinf(JS3)] = 0
            # print(np.isnan(JS1).any())
            JS = JS1 * JS2 * JS3

        return JS

