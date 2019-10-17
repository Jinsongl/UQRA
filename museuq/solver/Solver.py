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
from tqdm import tqdm
from .dynamic_models import lin_oscillator, duffing_oscillator, linear_oscillator
from .benchmark import bench1, bench2, bench3, bench4, ishigami
from ..utilities.classes import ObserveError
from ..utilities import helpers as museuq_helpers
from ..utilities import constants as const
from itertools import compress

solvers_collections = {
    'ISHIGAMI'  : ishigami,
    'BENCH1'    : bench1,
    'BENCH2'    : bench2,
    'BENCH3'    : bench3,
    'BENCH4'    : bench4,
    'DUFFING'   : duffing_oscillator,
    'SDOF'      : linear_oscillator,
    'LINEAR_OSCILLATOR' : linear_oscillator,
    }

solvers_ndim  = {
    'ISHIGAMI'  : int(3),
    'BENCH1'    : int(1),
    'BENCH2'    : int(1),
    'BENCH3'    : int(1),
    'BENCH4'    : int(1),
    'DUFFING'   : int(1),
    'SDOF'      : int(2),
    'LINEAR_OSCILLATOR' : int(2),
        }

class Solver(object):
    """
    Solver object, served as a swrapper for all solvers contained in this package
    M(x,t,theta_m;y) = s(x,t; theta_s)
    - x: array -like (ndim, nsamples) ~ dist(x) 
        1. array of x samples (ndim, nsamples), y = M(x)
        2. ndarray of time series
    - t: time 
    - M: deterministic solver, defined by solver_name
    - theta_m: parameters defining solver M of shape (m,n)(sys_def_params)
        - m: number of set, 
        - n: number of system parameters per set
    - y: variable to solve 
    - s: excitation source function, defined by source_func (string) 
    - theta_s: parameters defining excitation function s 

    Return:
        List of simulation results (Different doe_order, sample size would change, so return list)
        if more than one set of system params: 
          - output      = [sys_params0,  sys_params1,  ... ]
          -- sys_param0 = [sim_res_DoE0, sim_res_DoE1, sim_res_DoE2,... ]
        1. len(res) = len(sys_def_params) 
        2. For each element in res: simulation results from 1 DoE set
        3. Return from solver:
            a). [*, ndofs]
                each column represents a time series for one dof
            b). 
    """

    def __init__(self,solver_name, x, **kwargs):
        self.solver_name= solver_name
        self.input_x    = x
        self.ndim       = solvers_ndim[solver_name.upper()] 

        ## kwargs: 
        self.theta_m    = kwargs.get('theta_m'  , [None])
        self.error      = kwargs.get('error'    , ObserveError())
        self.source_func= kwargs.get('source_func', None)
        self.theta_s    = kwargs.get('theta_s'  , None)

        self.output     = []
        self.output_stats=[]

        print(r'------------------------------------------------------------')
        print(r'>>> Initialize Solver Obejct...')
        print(r'------------------------------------------------------------')
        print(r' > Solver (system) properties:')
        print(r'   * {:<17s} : {:15s}'.format('Solver name', solver_name))


        if self.theta_m is None or self.theta_m[0] is None:
            print(r'   * {:<17s} : {:d}'.format('No. Solver set', 1))
        else:
            print(r'   * {:<17s} : {}'.format('No. Solver set', self.theta_m.T.shape))
            # print(r'   * Solver parameters: ndim={:d}, nsets={:d}'.format(self.theta_m.shape[1], self.theta_m.shape[0]))
        if self.source_func:
            print(r'   * System excitation functions:')
            print(r'     - {:<15s} : {}'.format('function'   , self.source_func))
            print(r'     - {:<15s} : {}'.format('parameters' , self.theta_s))

        if kwargs:
            print(r'   * Solver kwargs:')
            for key, value in kwargs.items():
                print(r'     - {:<15s} : {}'.format(key, value))

        ###------------- Error properties ----------------------------

    def run(self, *args, **kwargs):
        """
        run solver for given DoE object
        Parameters:

        Returns:

        """
        doe2run     = [self.input_x] if isinstance(self.input_x, (np.ndarray, np.generic)) else self.input_x
        self.output = [] # a list saving simulation results
        print(r' > Running Simulation...')

        for idoe, isamples_x in enumerate(doe2run):
            for isys_done, itheta_m in enumerate(self.theta_m): 
                input_vars = isamples_x[:self.ndim,:]
                ### Run simulations
                y = self._solver_wrapper(input_vars, *args, **kwargs)
                self.output.append(y)
                print(r'   ^ DoE set : {:d} / {:d}'.format(idoe, len(doe2run)), end='')
                print(r'    -> Solver output : {}'.format(y.shape))
        if len(self.output) == 1: # if only one doe set, just return the result, instead of list with length 1
            self.output = self.output[0]
        return self.output

    def get_stats(self, qoi2analysis='all', stats2cal=[1,1,1,1,1,1,0]):
        """
        Return column-wise statistic properties for given qoi2analysis and stats2cal
        Parameters:
            - qoi2analysis: array of integers, Column indices to analysis
            - stats2cal: array of boolen, indicators of statistics to calculate
              [mean, std, skewness, kurtosis, absmax, absmin, up_crossing]
        Return:
            list of calculated statistics [ np.array(nstats, nqoi2analysis)] 
        """

        print(r' > Calculating statistics...')
        print(r'   * {:<15s} '.format('post analysis parameters'))
        qoi2analysis = qoi2analysis if qoi2analysis is not None else 'All'
        print(r'     - {:<15s} : {} '.format('qoi2analysis', qoi2analysis))
        stats_list = ['mean', 'std', 'skewness', 'kurtosis', 'absmax', 'absmin', 'up_crossing']
        print(r'     - {:<15s} : {} '.format('statistics'  , list(compress(stats_list, stats2cal)) ))

        for idoe, idoe_output  in enumerate(self.output):
            print(r'     - DoE set : {:d} / {:d}'.format(idoe, len(self.output)), end='')
            idoe_output_stats = []
            for data in idoe_output:
                data = data if qoi2analysis.lower() == 'all' else data[qoi2analysis]
                stat = museuq_helpers.get_stats(np.squeeze(data), stats=stats2cal)
                idoe_output_stats.append(stat)
            print(r'    -> DoE Statistics output : {}'.format(np.array(idoe_output_stats).shape))
            self.output_stats.append(np.array(idoe_output_stats))

        return self.output_stats

    def _solver_wrapper(self, x, *args, **kwargs):
        """
        a wrapper for all solvers

        Parameters:
        M(x,t,theta_m;y) = s(x,t; theta_s)
        - x: array -like (ndim, nsamples) ~ dist(x) 
            1. array of x samples (ndim, nsamples), y = M(x)
            2. ndarray of time series
        - t: time 
        - M: deterministic solver, defined by solver_name
        - theta_m: parameters defining solver M of shape (m,n)(theta_m)
            - m: number of set, 
            - n: number of system parameters per set
        - y: variable to solve 
        - s: excitation source function, defined by source_func (string) 
        - theta_s: parameters defining excitation function s 

        Return: ndarray of shape (nsamples, [each solver return])
        [each solver return]: (:, nqoi)

        """

        # solver_name, sterm_dist = model_def #if len(model_def) == 2 else model_def[0], None
        # print(rsolver_name)
        try:
            solver = solvers_collections[self.solver_name.upper()]
        except KeyError:
            print(f"{self.solver_name.upper()} is not defined" )
        assert (callable(solver)), '{:s} not callable'.format(solver.__name__)
        
        if self.solver_name.upper() == 'ISHIGAMI':
            p = kwargs.get('theta_m', [7,0.1])
            # p = theta_m if theta_m else [7,0.1]
            y = solver(x, p=p)

        elif self.solver_name.upper()[:5] == 'BENCH':
            y = solver(x, self.error, **kwargs)

        elif self.solver_name.upper() == 'LIN_OSCILLATOR':
            time_max= simParameters.time_max
            dt      = simParameters.dt
            ## Default initial condition [0,0]
            if theta_m:
                x0, v0, zeta, omega0 = theta_m
            else:
                x0, v0, zeta, omega0 = (0,0, 0.2, 1)

            y = solver(time_max,dt,x0,v0,zeta,omega0,add_f=x[0],*x[1:])

        elif self.solver_name.upper() == 'LINEAR_OSCILLATOR':

            kwargs_default = {
                    'time_max'  : 1000,
                    'dt'        : 0.1,
                    'spec_name' : 'JONSWAP',
                    'return_all': False,
                    'm'         : 1,
                    'c'         : 0.02/np.pi,
                    'k'         : 1.0/np.pi**2, 
                    'mck'       : (1, 0.02/np.pi, 1.0/np.pi**2),
                    'zeta'      : 0.01,
                    'omega_n'   : 2
                    }
            kwargs_default.update(kwargs)

            tmax = kwargs_default['time_max']
            dt   = kwargs_default['dt']
            m    = kwargs_default['m']
            c    = kwargs_default['c']
            k    = kwargs_default['k']
            ### two ways defining mck
            if 'k' in kwargs.keys() and 'c' in kwargs.keys():
                k   = kwargs_default['k'] 
                c   = kwargs_default['c']
                zeta= c/(2*np.sqrt(m*k))
                kwargs_default['zeta'] = zeta
                omega_n = 2*np.pi*np.sqrt(k/m)
                kwargs_default['omega_n'] = omega_n
            elif 'zeta' in kwargs.keys() and 'omega_n' in kwargs.keys():
                zeta    = kwargs_default['zeta']
                omega_n = kwargs_default['omega_n'] # rad/s
                k       = (omega_n/2/np.pi) **2 * m
                c       = zeta * 2 * np.sqrt(m * k)
                kwargs_default['k'] = k
                kwargs_default['c'] = c
            kwargs_default['mck'] = (m,c,k)
            
            for key, value in kwargs_default.items():
                print(r'     - {:<15s} : {}'.format(key, value))

            t = np.arange(0,int(tmax/dt) +1) * dt
            x = np.array(x)
            if x.size == 2 or x.shape[1] == 1:
                y = linear_oscillator(t,x, **kwargs_default)
            else:
                pbar_x  = tqdm(x.T, ascii=True, desc="   - ")
                y = np.array([linear_oscillator(t,ix, **kwargs_default) for ix in pbar_x])


        elif self.solver_name.upper() ==  'DUFFING_OSCILLATOR':
            # x: [source_func, *arg, *kwargs]

            time_max  = simParameters.time_max
            dt        = simParameters.dt
            normalize = simParameters.normalize

            if len(x) == 3:
                source_func, source_kwargs, source_args = x
            elif len(x) == 2:
                source_func, source_kwargs, source_args = x[0], None, x[1]

            ## Default initial condition [0,0]
            if theta_m is not None:
                x0, v0, zeta, omega0, mu = theta_m
            else:
                x0, v0, zeta, omega0, mu = (0,0,0.02,1,1)

            # print(rsolver)
            # print(rsource_func, source_kwargs, source_args)
            # y = duffing_oscillator(time_max,dt,x0,v0,zeta,omega0,mu, *source_args,source_func=source_func,source_kwargs=source_kwargs)
            y,dt,pstep= solver(time_max,dt,x0,v0,zeta,omega0,mu, *source_args,source_func=source_func,source_kwargs=source_kwargs,normalize=normalize)

        else:
            raise ValueError('Function {} not defined'.format(solver.__name__)) 
        
        return y




