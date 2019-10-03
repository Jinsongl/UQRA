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
            print(r'   ♦ Short-term/error distribution parameters: noise free')
        else:
            print(r'   ♦ Short-term/error distribution parameters:')
            print(r'     ∙ {:<15s} : {}'.format('dist_name', self.name))
            for i, ierror_params in enumerate(self.params):
                ierror_params_shape = []
                for iparam in ierror_params:
                    if np.isscalar(iparam):
                        ierror_params_shape.append(1)
                    else:
                        ierror_params_shape.append(np.array(iparam).shape)

                if i == 0:
                    print(r'     ∙ {:<15s} : {}'.format('dist_params',ierror_params_shape))
                else:
                    print(r'     ∙ {:<15s} : {}'.format('',ierror_params_shape))
            print(r'     ∙ {:<15s} : {}'.format('dist_size', self.size))

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
