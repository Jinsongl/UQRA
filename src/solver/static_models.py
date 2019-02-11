#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import numpy as np


def ishigami(x, p=None):
    """
    The ishigami function of ishigami & Homma (1990) is used as an example for uncertainty and sensitivity analysis methods, because it exhibits strong nonlinearity and nonmonotonicity. It also has a peculiar dependence on x3, as described by Sobol' & Levitan (1999). 

    The values of a and b used by Crestaux et al. (2007) and Marrel et al. (2009) are: a = 7 and b = 0.1. Sobol' & Levitan (1999) use a = 7 and b = 0.05. 

    Input Distributions:
    The independent distributions of the input random variables are usually: xi ~ Uniform[-π, π], for all i = 1, 2, 3.

    References:
    Crestaux, T., Martinez, J.-M., Le Maitre, O., & Lafitte, O. (2007). Polynomial chaos expansion for uncertainties quantification and sensitivity analysis [PowerPoint slides]. Retrieved from SAMO 2007 website: http://samo2007.chem.elte.hu/lectures/Crestaux.pdf.

    I3shigami, T., & Homma, T. (1990, December). An importance quantification technique in uncertainty analysis for computer models. In Uncertainty Modeling and Analysis, 1990. Proceedings., First International Symposium on (pp. 398-403). IEEE.

    Marrel, A., Iooss, B., Laurent, B., & Roustant, O. (2009). Calculations of sobol indices for the gaussian process metamodel. Reliability Engineering & System Safety, 94(3), 742-751.

    Saltelli, A., Chan, K., & Scott, E. M. (Eds.). (2000). Sensitivity analysis (Vol. 134). New York: Wiley.

    Sobol', I. M., & Levitan, Y. L. (1999). On the use of variance reducing multipliers in Monte Carlo computations of a global sensitivity index. Computer Physics Communications, 117(1), 52-61.

    Arguments:
        x: array-like of shape(ndim, nsamples)
        p: parameters for ishigami
    Return:
        y: array-like (nsamples,)
    """
    x = np.array(x)
    p = p or [7, 0.1]
    assert x.shape[0] == int(3), 'x.shape={}'.format(x.shape)

    if x.ndim == 1:
        y = np.sin(x[0]) + p[0] * np.sin(x[1])**2 + p[1]*x[2]**4 * np.sin(x[0])
    else:
        y = np.sin(x[0,:]) + p[0] * np.sin(x[1,:])**2 + p[1]*x[2,:]**4 * np.sin(x[0,:])
    return y

def poly5(x):
    x = np.array(x)
    y = x**5 - 10* x**3 + 15*x
    return float(y) if y.ndim == 1 else y  

def benchmark1_func(x):
    x = np.array(x)
    return x**2*np.sin(5*x)

def sigma_x(x,sigma=0.5):
    x = np.array(x)
    return sigma*(x**2+1)

def benchmark1_normal(x,mu=0,sigma=0.5):
    """
    Benchmar problem #1 with normal error
    Arguments:
        x: ndarray of shape(ndim, nsamples)
        optional:
        (mu, sigma): parameters defining normal error
    Return:
        ndarray of shape(nsamples,)
    """
    x = np.array(x)
    sigmas = sigma_x(x,sigma=sigma)
    y0 = benchmark1_func(x)
    e = np.array([np.random.normal(mu, sigma, 1) for sigma in sigmas.T]).T
    y = y0 + e
    return y

def benchmark1_gumbel(x,mu=0,beta=0.5):
    x = np.array(x)
    x0 = x**2*np.sin(5*x)
    e = np.array([np.random.gumbel(mu, beta*abs(s),1) for s in x]).reshape((len(x),))
    mu, std
    return x0+ e
