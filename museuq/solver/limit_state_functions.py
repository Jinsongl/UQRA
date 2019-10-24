#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""
Collections of limit state functions
input x is of shape (ndim nsamples)

References:
"""
import numpy as np


##### fucntion g1-g5 are fom reference: "Papaioannou, Iason, Costas Papadimitriou, and Daniel Straub. "Sequential importance sampling for structural reliability analysis." Structural safety 62 (2016): 66-75"

def g1(x):
    """
    Convex limit-state function
    """
    x0,x1 = np.array(x)
    g = 0.1*(x0-x1) **2 - 1.0/np.sqrt(2)(x1+x0) + 2.5
    return g


def g2(x, b=5, k=0.5, e=0.1):
    """
    parabolic/concave limit-state function

    """
    x0,x1 = np.array(x)
    g = b-x1-k*(x0-e)**2
    return g 
    

def g3(x):
    """
    Series system reliability problem

    """
    x0,x1 = np.array(x)

    g1 = 0.1 * (x0-x1)**2 - (x0+x1)/np.sqrt(2) + 3
    g2 = 0.1 * (x0-x1)**2 + (x0+x1)/np.sqrt(2) + 3
    g3 = x0-x1 + 7/np.sqrt(2)
    g3 = x1-x0 + 7/np.sqrt(2)

    g12 = np.minimum(g1, g2)
    g34 = np.minimum(g3, g4)
    g   = np.minimum(g12, g34)
    return g

def g4(x):
    """
    Noisy limit-state function

    """
    x = np.array(x)
    g = x[0] + 2*x[1] + 2 * x[2] + x[3] -5*x[4] -5*x[5] + 0.001*np.sum(np.sin(100*x),axis=0)
    return g


def g5(x, beta=3.5):
    """
    Linear limit-state function in high dimensions
    """
    x = np.array(x)
    n = x.shape[0]
    g = -1/np.sqrt(n) * np.sum(x, axis=0) + beta
    return g
    

##### fucntion g1-g5 are fom reference: "Papaioannou, Iason, Costas Papadimitriou, and Daniel Straub. "Sequential importance sampling for structural reliability analysis." Structural safety 62 (2016): 66-75"

def g6(x, a=3, mu=1,sigma=0.2):
    x = np.array(x)
    n = x.shape[0]
    g = n + a * sigma * np.sqrt(n) - np.sum(x, axis=0)
    return g


def g7(x, c):
    """
    Multiple design points
    """
    g1 = c -1 - x[1] + np.exp(-x[0]**2/10.0) + (x[0]/5.0)**4
    g2 = c**2/2.0 - x[0] * x[1]
    g  = np.minimum(g1,g2)
    return g


def g8():
    """
This structural reliability example was first proposed in the report by Der Kiureghian[1]. It was then used for benchmark purposes in the recent article by Bourinet et al. [2]. It consists in studying the failure of a two-degree-of-freedom damped oscillator under a white-noise base excitation. The probabilistic model is composed of n=8 independent random variables whose distributions are defined in Table below. Fs mean value is varied from 15.0 to 27.5 as in Bourinet et al. [2]. The limit-state function reads as follows:
    [1]: Der Kiureghian A, de Stefano M. Efficient algorithms for second order reliability analysis. Journal of Engineering Mechanics 1991;117(12): 2906–23.
    [2]: Bourinet J-M, Deheeger F, Lemaire M. Assessing small failure probabilities by combined subset simulation and support vector machines. Structural Safety 2011;33(6):343–53.
    -------------------------------
    Variable    | Distribution  |   Mean            | c.o.v. (%)
    mp          | Lognormal     |   1.5             | 10
    ms          | Lognormal     |   0.01            | 10 
    kp          | Lognormal     |   1               | 20 
    ks          | Lognormal     |   0.01            | 20
    zp          | Lognormal     |   0.05            | 40
    zs          | Lognormal     |   0.02            | 50 
    FS          | Lognormal     |   {15,21.5,27.5}  | 10
    S0          | Lognormal     |   100             | 10
    """
    return 0
