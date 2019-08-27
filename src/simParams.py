#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import sys
import chaospy as cp
import numpy as np
import os
from datetime import datetime
from doe.doe_generator import samplegen
## Define parameters class
DOE_METHOD_NAMES = {
    "GQ"    : "Quadrature"  , "QUAD"  : "Quadrature",
    "MC"    : "Monte Carlo" , "FIX"   : "Fixed point"
    } 

DOE_RULE_NAMES = {
    "c": "clenshaw_curtis"  , "e"   : "gauss_legendre"  , "p"   : "gauss_patterson",
    "z": "genz_keister"     , "g"   : "golub_welsch"    , "j"   : "leja",
    "h": "gauss_hermite"    ,"lag"  : "gauss_laguerre"  , "cheb": "gauss_chebyshev",
    "hermite"   :"gauss_hermite",
    "legendre"  :"gauss_legendre",
    "jacobi"    :"gauss_jacobi",
    "R": "Pseudo-Random", "RG": "Regular Grid", "NG": "Nested Grid", "L": "Latin Hypercube",
    "S": "Sobol", "H":"Halton", "M": "Hammersley",
    "FIX": "Fixed point"
    }

def _num2str(n):
    if n<100:
        return '{:2d}'.format(n)
    else:
        __str ='{:.0E}'.format(n) 
        return __str[0]+'E'+__str[-1] 

def _makelogfile(sys_definition_params, sys_input_params,idoe_filename,  sys_input_x_shape):
    """
    Create log files tracking simulations have been done
    """
    now = datetime.now()
    date_string = now.strftime("%d/%m/%Y %H:%M:%S")

    with open('logfile.txt', 'a') as filein_id:
        filein_id.write('-'*50+'\n')
        filein_id.write(date_string+'\n')
        # filein_id.write('{:<30s}:{:<25s}\n'.format('system definition parameters', str(sys_definition_params)))
        # filein_id.write('{}\n'.format('system input function parameters'))
        # filein_id.write('\t{:<30s}:{:<25s}\n'.format('systerm input function name', str(sys_input_params[0])))
        # filein_id.write('\t{:<30s}:{:<25s}\n'.format('systerm input function kwargs', str(sys_input_params[1])))
        # filein_id.write('{:<30s}\n'.format('systerm input variables (DoE)'))
        filein_id.write('\t{:<30s}:{:<25s}\n'.format('systerm input variables file name', idoe_filename))
        # filein_id.write('\t{:<30s}:{}\n'.format('systerm input variables shape',sys_input_x_shape ))
        # for isys_def_params in sys_definition_params:
            # l1 = list(isys_def_params) if isys_def_params else [np.inf]
            # for isys_input_func_name in  sys_input_params[0]:
                # l2 = list(isys_input_func_name) if isys_input_func_name else [np.inf]
                # for isys_input_kwargs in sys_input_params[1]:
                    # l3 = list(isys_input_kwargs ) if isys_input_kwargs else [np.inf]
                    # assert sys_input_x.shape == sys_input_zeta.shape, 'Number of elements are not same' 
                    # samples_input = np.concatenate((sys_input_x,sys_input_zeta))
                    # for isample_input in samples_input.T:
                        # l4 = list(isample_input) 
                        # data1line = l1 + l2 + l3 + l4
                        # filein_id.write('\t'.join(map(str, data1line)))
                        # # filein_id.write(data1line)
                        # filein_id.write("\n")

class ErrorType():
    def __init__(self, name=None, params=None, size=None):
        """
        name:   string, error distribution name or None if not defined
        params: list of error distribution parameters, float or array_like of floats 
                [ [mu1, sigma1, ...]
                  [mu2, sigma2, ...]
                  ...]
        size:   list of [int or tuple of ints], optional
            -- Output shape:
            If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. 
            If size is None (default), a single value is returned if loc and scale are both scalars. 
            Otherwise, np.broadcast(loc, scale).size samples are drawn.
        """
        assert name is None or isinstance(name, str) 
        if name is None:
            self.name   = 'Free' 
            self.params = None 
            self.size   = None 
        elif len(params) == 1:
            self.name   = name
            self.params = params[0]
            self.size   = size[0] if size else None
        else:
            self.name   = [] 
            for _ in range(len(params)):
                self.name.append(name) 
            self.size   = size if size else [None] * len(params)
            self.params = params

    def tolist(self, ndoe):
        """

        """
        if not isinstance(self.name, list):
            return [ErrorType(self.name, [self.params], [self.size])] * ndoe
        else:
            assert len(self.name) == ndoe,"ErrorType.name provided ({:2d}) doesn't match number of DoE expected ({:2d})".format(len(self.name), ndoe)
            error_type_list = []
            for i in range(ndoe):
                error_type_list.append(ErrorType(self.name[i], [self.params[i]], self.size[i]))
            return error_type_list

    def disp(self):

        if self.name.upper() == 'FREE':
            print('   ♦ Short-term/error distribution parameters: noise free')
        else:
            print('   ♦ Short-term/error distribution parameters:')
            print('     ∙ {:<15s} : {}'.format('dist_name', self.name))
            for i, ierror_params in enumerate(self.params):
                ierror_params_shape = []
                for iparam in ierror_params:
                    if np.isscalar(iparam):
                        ierror_params_shape.append(1)
                    else:
                        ierror_params_shape.append(np.array(iparam).shape)

                if i == 0:
                    print('     ∙ {:<15s} : {}'.format('dist_params',ierror_params_shape))
                else:
                    print('     ∙ {:<15s} : {}'.format('',ierror_params_shape))
            print('     ∙ {:<15s} : {}'.format('dist_size', self.size))


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

class simParameter(object):
    """
    Define general parameter settings for simulation running and post data analysis. 
    System parameters will be different between solver and solver

    Arguments:
        dist_zeta: list of selected marginal distributions from Wiener-Askey scheme
        *OPTIONAL:
        doe_params  = [doe_method, doe_rule, doe_order]
        time_params = [time_start, time_ramp, time_max, dt]
        post_params = [qoi2analysis=[0,], stats=[1,1,1,1,1,1,0]]
            stats: [mean, std, skewness, kurtosis, absmax, absmin, up_crossing]
        sys_def_params: array of parameter sets defining the solver
            if no sys_def_params is required, sys_def_params = None
            e.g. for duffing oscillator:
            sys_def_params = np.array([0,0,1,1,1]).reshape(1,5) # x0,v0, zeta, omega_n, mu 
        normalize: 
    """

    def __init__(self,dist_zeta, doe_params=['GQ','lag',[2,3]],\
            time_params=None, post_params=[[0,], [1,1,1,1,1,1,0]],\
            sys_def_params=None, normalize=False):
        sys.stdout = Logger()
        ###---------- Random variable parameters ------------------------
        self.seed       = [0,100]
        self.dist_zeta  = dist_zeta
        self.dist_zeta_J= dist_zeta #if len(dist_zeta) == 1 else cp.J(*dist_zeta) 

        ###-----------DoE parameters ------------------------------------
        self.doe_params = doe_params
        self.doe_method = DOE_METHOD_NAMES.get(doe_params[0])
        self.doe_rule   = doe_params[1]
        self.doe_order  = []
        self.doe_filenames  = 'DoE_filenames.txt'
        if np.isscalar(doe_params[2]): 
            self.doe_order.append(int(doe_params[2]))
        else:
            self.doe_order = np.array(doe_params[2])
        self.ndoe           = len(self.doe_order)   # number of doe sets

        ###-------------Directories setting -----------------------------
        self.pwd            = ''
        self.data_dir       = ''
        self.figure_dir     = ''
        ###-------------Systerm input params ----------------------------
        ### sys_def_params is of shape (m,n)
        ##  - m: number of set, 
        ##  - n: number of system parameters per set
        self.sys_def_params = sys_def_params if sys_def_params else [None]
        self.sys_input_params = [[None], [None]]### sys_input_params = [sys_input_func_name, sys_input_func_kwargs]
        self.sys_input_x    = []
        self.sys_input_zeta = []

        # self.ndim_sys_inputs= len(self.dist_zeta_J)                     # dimension of solver inputs 
        # self.nsamples_per_doe   = []                    # number of samples for each doe sets
        # self.nsamples_done  = 0
        # self.nsamples_per_doe.append(samp_zeta[0].shape[1])

        ###-------------Error type paramters ----------------------------
        self.error = ErrorType()

        ###-------------Others ------------------------------------------
        self.time_start, self.time_ramp, self.time_max, self.dt = time_params if time_params else [None]*4
        self.qoi2analysis, self.stats = post_params
        self.normalize      = normalize 

    def get_doe_samples_zeta(self, sys_input_x=None, dist_x=None):
        """
        Return design sample points both in zeta spaces based on specified doe_method
        Return:
            Experiment samples in zeta space (idoe_order,[samples for each doe_order])
        """

        print('------------------------------------------------------------')
        print('►►► Design of Experiments (DoEs)')
        print('------------------------------------------------------------')
        print(' ► DoE parameters: ')
        print('   ♦ {:<15s} : {:<15s}'.format('DoE method', self.doe_method))
        print('   ♦ {:<15s} : {:<15s}'.format('DoE rule', DOE_RULE_NAMES[self.doe_rule]))
        print('   ♦ {:<15s} : {}'.format('DoE order', list(map(_num2str, self.doe_order))))
        print(' ► Running DoEs: ')

        self.sys_input_zeta = []
        doe_completed = 0
        if self.doe_method.upper() == 'FIXED POINT':
            assert sys_input_x is not None, " For 'fixed point' method, samples in physical space must be given" 
            for isample_x in sys_input_x:
                self.sys_input_zeta.append(self._dist_transform(dist_x, self.dist_zeta, isample_x))
        else: ## could be MC or quadrature
            for idoe_order in self.doe_order: 
                # DoE for different selected doe_order 
                samp_zeta = samplegen(self.doe_method, idoe_order, self.dist_zeta_J, rule=self.doe_rule)
                self.sys_input_zeta.append(samp_zeta)
                doe_completed +=1
                print('   ♦ {:<15s}: {:d}/{:d}'.format( 'DoE completed', doe_completed, self.ndoe ))

                # if self.doe_method.upper() == 'QUADRATURE':
                    # ## if quadrature, samp_zeta = [coord(ndim, nsamples), weights(nsamples,)]
                    # # self.ndim_sys_inputs = samp_zeta[0].shape[0] 
                    # self.nsamples_per_doe.append(samp_zeta[0].shape[1])
                # else:
                    # ## if mc , samp_zeta = [coord(ndim, nsamples)]
                    # # self.ndim_sys_inputs = samp_zeta.shape[0] 
                    # self.nsamples_per_doe.append(samp_zeta.shape[1])

        # print(' ► ----   Done (Design of Experiment)   ----')
        print(' ► DoE Summary:')
        print('   ♦ Number of sample sets : {:d}'.format(len(self.sys_input_zeta)))
        print('   ♦ Sample shape: ')
        for isample_zeta in self.sys_input_zeta:
            if self.doe_method.upper() == 'QUADRATURE':
                print('     ∙ Coordinate: {}; weights: {}'\
                        .format(isample_zeta[0].shape, isample_zeta[1].shape))
            else:
                print('     ∙ Coordinate: {}'.format(isample_zeta.shape))

    def get_doe_samples(self, dist_x, is_both=True, filename_leading='DoE'):
        """
        Return and save design sample points both in physical spaces based on specified doe_method
        Return:
        sys_input_x: parameters defining the inputs for the solver
            General format of a solver
                y =  M(x,sys_def_params)
                M: system solver, taking a set of system parameters and input x
                x: system inputs, ndarray of shape (ndim, nsamples)
                    1. M takes x[:,i] as input, y = M(x.T)
                    2. M takes time series generated from input_func(x[:,i]) as input
            Experiment samples in physical space (idoe_order,[samples for each doe_order])
        """

        now = datetime.now()
        date_string = now.strftime("%d/%m/%Y %H:%M:%S")


        # # self.sys_input_x = []
        assert len(dist_x) == len(self.dist_zeta)

        ## Get samples in zeta space first
        self.get_doe_samples_zeta() 

        # n_sys_input_func_name = len(self.sys_input_params[0]) if self.sys_input_params[0] else 1
        n_sys_input_func_name = len(self.sys_input_params[0]) 
        n_sys_input_kwargs    = len(self.sys_input_params[1]) 
        n_sys_def_params      = len(self.sys_def_params) 
        n_total_simulations   = n_sys_input_func_name * n_sys_input_kwargs *n_sys_def_params 

        ## Then calcuate corresponding samples in physical variable space 
        # with open(os.path.join(self.data_dir, self.doe_filenames), 'w') as doe_filenames:

        logtext ='-'*50+'\n' + date_string +'\n' 
        for idoe, isample_zeta in enumerate(self.sys_input_zeta):
            if self.doe_method.upper() == 'QUADRATURE':
                zeta_cor, zeta_weights = isample_zeta 
                x_cor = self._dist_transform(self.dist_zeta, dist_x, zeta_cor)
                x_weights = zeta_weights#.reshape(zeta_cor.shape[1],1)
                isample_x = np.array([x_cor, x_weights])
            else:
                zeta_cor, zeta_weights = isample_zeta, None
                x_cor = self._dist_transform(self.dist_zeta, dist_x, zeta_cor)
                isample_x = x_cor
            # print('isample_x shape: {}, coord shape: {}, weight shape {}'.format(isample_x.shape, isample_x[0].shape, isample_x[1].shape))

            self.sys_input_x.append(isample_x)

            ### save input variables to file
            idoe_filename = '_'.join([filename_leading, self.doe_params[0], self.doe_params[1], _num2str(self.doe_order[idoe])]) 
            samples_input = np.concatenate((isample_x,isample_zeta))
            logtext += idoe_filename
            logtext += '\n'

            idoe_filename = os.path.join(self.data_dir, idoe_filename)
            np.save(idoe_filename,samples_input)

        with open('DoE_logfile.txt', 'a') as filein_id:
            filein_id.write(logtext)

            # _save2file(idoe_filename, self.sys_def_params, self.sys_input_params, isample_x, isample_zeta)
            # print(isample_x)
            # print(isample_zeta)
            # print(' ► Data saved {}'.format(idoe_filename))

        # print('   ♦ Total number of simulations: {:d}'.format(int(n_total_simulations)))
        # self.sys_input_params[2] = self.sys_input_x

    def set_doe_samples(self, input_x, dist_x):

        ## Set samples in physical space first
        self.sys_input_x = input_x
        self.get_doe_samples_zeta(input_x, dist_x) 

        self.sys_input_params[2].append(self.sys_input_x)


    def set_doe_method(self,doe_method):
        self.doe_method = doe_method

    def set_doe_order(self,p):
        self.doe_order = p

    def set_error(self, params=None):
        """
        Set error distributions 
        """
        if params is None:
            self.error = ErrorType()
        else:
            name, params, size = params
            self.error = ErrorType(name=name, params=params, size=size)

    def add_nsamples_done(self,n):
        self.nsamples_done += n

    def set_seed(self, s):
        self.seed = s

    def update_filename(self,filename):
        self.doe_filenames = filename

    def update_dir(self, **kwargs):
        """
        Updating directories of
            pwd: present working directory, self.pwd
            data_dir: directory saving all data, self.data_dir
            figure_dir: directory saving all figures, self.figure_dir
        """
        self.pwd = kwargs.get('pwd',os.getcwd())
        self.data_dir = kwargs.get('data_dir', '')
        self.figure_dir = kwargs.get('figure_dir', '')

    def set_qoi2analysis(self, qois):
        # what is this??
        self.qoi2analysis = qois
    def set_sys_input_params(self, sys_input_func_name, sys_input_kwargs):
        self.sys_input_params[0] = sys_input_func_name
        self.sys_input_params[1] = sys_input_kwargs

    # def set_nsamples_need(self, n):
        # self.nsamples_need= n
        # self.doe_method = xrange(self.nsamples_need)



    def _dist_transform(self, dist1, dist2, var1):
        """
        Transform variables (var1) from list of dist1 to correponding variables in dist2. based on same icdf

        Arguments:
            dist2: list of independent distributions (destination)
            var1 : variables in dist1 of shape[ndim, nsamples]

        Return:
            
        """
        var1 = np.array(var1)

        dist1  = [dist1,] if len(dist1) == 1 else dist1
        dist2  = [dist2,] if len(dist2) == 1 else dist2
        assert (len(dist1) == len(dist2) == var1.shape[0]), 'Dimension of variable not equal. dist1.ndim={}, dist2.ndim={}, var1.ndim={}'.format(len(dist1), len(dist2), var1.shape[0])
        var2 = []
        for i in range(var1.shape[0]):
            _dist1 = dist1[i]
            _dist2 = dist2[i]
            _var = _dist2.inv(_dist1.cdf(var1[i,:]))
            var2.append(_var)
        var2 = np.array(var2)
        assert var1.shape == var2.shape
        return var2

