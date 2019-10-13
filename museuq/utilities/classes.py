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
import sys, os
from datetime import datetime
class ObserveError():
    def __init__(self, name=None, **kwargs):
        """
        name:   string, error distribution name or None if not defined
        """
        self.name   = name if name else 'None' 
        self.loc    = kwargs.get('loc', 0)
        self.scale  = kwargs.get('scale', 1)
        self.cov    = kwargs.get('cov', False)
        # self.params = None 
        # self.size   = None 

    def error(self, size, *params):
        """
        Generate observe errors from specified error type 

        Arguments:
        params: args
        list of error distribution parameters, float or array_like of floats 
                [ [mu1, sigma1, ...]
                  [mu2, sigma2, ...]
                  ...]
        size : int or tuple of ints, optional
            -- Output shape:
            If the given shape is, e.g., (m, n, k), then m * n * k observe errors. 
            If size is None (default), a single value is returned if loc and scale are both scalars. 
            Otherwise, np.broadcast(loc, scale).size observe errors.

        """
        loc, scale = (0, params[1]) if self.cov else (self.loc, self.scale)

        if self.name.upper() == 'NONE':
            observe_errors = 0

        elif self.name.upper() == 'NORMAL':
            observe_errors = np.random.normal(loc=loc, scale=scale, size=size) 

        elif self.name.upper() == 'GUMBEL':
            observe_errors = np.random.gumbel(loc=loc, scale=scale, size=size)

        # elif self.name.upper() == 'WEIBULL':
            # observe_errors = scale * np.random.weibull(shape, size=size)
        else:
            raise NotImplementedError("{:s} error is not implemented".format())
        return observe_errors
    
    # def tolist(self, ndoe):
        # """

        # """
        # if not isinstance(self.name, list):
            # return [ObserveError(self.name, [self.params], [self.size])] * ndoe
        # else:
            # assert len(self.name) == ndoe,"ObserveError.name provided ({:2d}) doesn't match number of DoE expected ({:2d})".format(len(self.name), ndoe)
            # error_type_list = []
            # for i in range(ndoe):
                # error_type_list.append(ObserveError(self.name[i], [self.params[i]], self.size[i]))
            # return error_type_list


    def __repr__(self):
        if self.cov:
            return "ObserveError({:s}(mu=0, cov={:.2f}))".format(self.name, self.cov)
        else:
            return "ObserveError({:s}(loc={:.2f}, scale={:.2f}))".format(self.name, self.loc, self.scale)
    def __str__(self):
        if self.cov:
            return "ObserveError({:s}(mu=0, cov={:.2f}))".format(self.name, self.cov)
        else:
            return "ObserveError({:s}(loc={:.2f}, scale={:.2f}))".format(self.name, self.loc, self.scale)

    # def disp(self):
        # if self.name.upper() == 'NONE':
            # print(r'   * Short-term/error distribution parameters: None')
        # else:
            # print(r'   * Short-term/error distribution parameters:')
            # print(r'     ∙ {:<15s} : {}'.format('dist_name', self.name))
            # for i, ierror_params in enumerate(self.params):
                # ierror_params_shape = []
                # for iparam in ierror_params:
                    # if np.isscalar(iparam):
                        # ierror_params_shape.append(1)
                    # else:
                        # ierror_params_shape.append(np.array(iparam).shape)

                # if i == 0:
                    # print(r'     ∙ {:<15s} : {}'.format('dist_params',ierror_params_shape))
                # else:
                    # print(r'     ∙ {:<15s} : {}'.format('',ierror_params_shape))
            # print(r'     ∙ {:<15s} : {}'.format('dist_size', self.size))

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logfile.log", "w+")

        now = datetime.now()
        date_string = now.strftime("%d/%m/%Y %H:%M:%S")
        logtext ='-'*50+'\n' + date_string +'\n' 
        self.log.write(logtext)


    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    
