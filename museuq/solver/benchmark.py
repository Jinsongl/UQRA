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

"""
Benchmark problems:
    y = M(x)
    - x: array -like (ndim, nsamples) ~ dist(x), N dimensional
    - M could be deterministic or stochastic solver, i.e.
        If M is deterministic, for given x, y = M(x) is same
        If M is stochastic, for given x, Y = M(x) ~ dist(y), Y is a realization of random number
Return:
    y: array-like (nsamples,) or (nsamples, nQoI)
"""


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
    assert x.shape[0] == int(3), 'Ishigami function expecting 3 random variables, {} given'.format(x.shape[0])

    if x.ndim == 1:
        y = np.sin(x[0]) + p[0] * np.sin(x[1])**2 + p[1]*x[2]**4 * np.sin(x[0])
    else:
        y = np.sin(x[0,:]) + p[0] * np.sin(x[1,:])**2 + p[1]*x[2,:]**4 * np.sin(x[0,:])
    return y

def bench1(x, error_type):
    """
    y = x*sin(x) + e
    """
    x = np.array(x)
    e = gen_error(error_type)
    y = x * np.sin(x)
    y = y + e
    y = y.reshape(x.shape[1],-1)
    return np.squeeze(y)

def bench2(x, error_type):
    """
    y = x^2 * sin(x) + e
    """
    x = np.array(x)
    e = gen_error(error_type)
    y = x**2*np.sin(5*x)
    y = y + e
    return y

def bench3(x, error_type):
    """
    y = sin(pi/5 * x) + 0.2 * cos(0.8 * pi * x)
    """
    x = np.array(x)
    e = gen_error(error_type)
    y = np.sin(np.pi/5.0 * x) + 1/5.0 * np.cos(4*np.pi*x/5.0) 
    y = y + e
    return y

def bench4(x, error):
    """
    y = 5 + -5*x + 2.5*x^2 -0.36*x^3 + 0.015*x^4
    """
    x = np.array(x)
    y = 5 + -5*x + 2.5*x**2 -0.36*x**3 + 0.015*x**4
    if error.name.upper() == 'NORMAL':
        mu    = y
        sigma = abs(mu) * error.cov
        params= (np.squeeze(mu), np.squeeze(sigma))
    elif error.name.upper() == 'GUMBEL':
        mu   = y
        beta = error.cov * abs(mu) / (np.pi/np.sqrt(6) - error.cov * 0.5772) 
        params = (np.squeeze(mu), np.squeeze(beta))
    else:
        pass
    e = error.error(x.size, *params)
    # print(np.mean(e), np.std(e))
    y = y + e
    return y
# def benchmark1_normal(x,mu=0,sigma=0.5):
    # """
    # Benchmar problem #1 with normal error
    # Arguments:
        # x: ndarray of shape(ndim, nsamples)
        # optional:
        # (mu, sigma): parameters defining normal error
    # Return:
        # ndarray of shape(nsamples,)
    # """
    # x = np.array(x)
    # sigmas = sigma_x(x,sigma=sigma)
    # y0 = benchmark1_func(x)
    # e = np.array([np.random.normal(mu, sigma, 1) for sigma in sigmas.T]).T
    # y = y0 + e
    # return y

# def benchmark1_gumbel(x,mu=0,beta=0.5):
    # x = np.array(x)
    # x0 = x**2*np.sin(5*x)
    # e = np.array([np.random.gumbel(mu, beta*abs(s),1) for s in x]).reshape((len(x),))
    # mu, std
    # return x0+ e
