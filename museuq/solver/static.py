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
from museuq.solver.base import Solver

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



class Ishigami(Solver):
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
    def __init__(self, p=[7,0.1], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'Ishigami'
        self.p    = p
        self.x    = []
     
    def __str__(self):
        return 'Solver: Ishigami function (p={})'.format(self.p)


    def run(self, x):
        x = np.array(x)

        assert x.shape[0] == int(3), 'Ishigami function expecting 3 random variables, {} given'.format(x.shape[0])
        if x.ndim == 1:
            self.y = np.sin(x[0]) + self.p[0] * np.sin(x[1])**2 + self.p[1]*x[2]**4 * np.sin(x[0])
        else:
            self.y = np.sin(x[0,:]) + self.p[0] * np.sin(x[1,:])**2 + self.p[1]*x[2,:]**4 * np.sin(x[0,:])



class xsinx(Solver):
    """
    y = x*sin(x) + e
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'xsinx'
        self.x    = []
        # self.error= kwargs.get('error', 'None') 

    def __str__(self):
        return 'Solver: x*sin(x)'

    def run(self, x):
        x = np.array(x)
        # e = self.error.samples()
        self.y = x * np.sin(x)
        # y = y + e

class poly4th(Solver):
    """
    y = 5 + -5*x + 2.5*x^2 -0.36*x^3 + 0.015*x^4
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'polynomial'
        self.x    = []

    def __str__(self):
        return 'Solver: 4th-order polynomial function'

    def run(self, x):
        x = np.squeeze(np.array(x))
        self.y = 5 + -5*x + 2.5*x**2 -0.36*x**3 + 0.015*x**4
        # e = error.samples()
        # y = y + e

class polynomial_square_root_function(Solver):
    """
    y = - [ (-x1+10)**2 + (x2+7)**2 + 10*(x1+x2)  **2 ]**0.5 + 14 
    x1,x2 ~ N(0,1)
    
    Benchmarks:
    prob(y>6) = 2.35E-6
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'polynomial square root function'
        self.x    = []

    def __str__(self):
        return 'Solver: Polynomial square root function'

    def run(self, x):
        x = np.squeeze(np.array(x))
        self.y = 5 + -5*x + 2.5*x**2 -0.36*x**3 + 0.015*x**4
        # e = error.samples()
        # y = y + e

class four_branch_system(Solver):
    """
    y = 10 - min{ 3 + 0.1(x1 - x2)**2 - (x1+x2)/sqrt(2)
                  3 + 0.1(x1 - x2)**2 + (x1+x2)/sqrt(2)
                  (x1 - x2) + 7/sqrt(2) 
                  (x2 - x1) + 7/sqrt(2) 
                  }

    This toy case allows us to test the ability of the rare event estimation methods to accurately estimate the probability in the case of disconnected failure region 􏱎f .

    Benchmarks:
    prob(y > 10) = 2.22E-3
    prob(y > 12) = 1.18E-6

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'four branch system'
        self.x    = []

    def __str__(self):
        return 'Solver: four branch system'

    def run(self, x):

        x = np.array(x).reshape(2,-1)
        x1 = x[0,:]
        x2 = x[1,:]

        y1 = 3 + 0.1*(x1 - x2)**2 - (x1+x2)/np.sqrt(2)
        y2 = 3 + 0.1*(x1 - x2)**2 + (x1+x2)/np.sqrt(2)
        y3 = (x1 - x2) + 7.0/np.sqrt(2) 
        y4 = (x2 - x1) + 7.0/np.sqrt(2) 

        self.y = np.array([y1, y2, y3, y4]).min(axis=0)
        self.y = 10 - self.y 

        # e = error.samples()
        # y = y + e

    
class polynomial_product_function(Solver):
    """
    y = 1/2 * sum( xi**4 + xi**2 + 5xi), i = 1, ..., d

    This toy case is useful to evaluate the ability of the methods to cope with high dimensional problems. 
    x: ndarray of shape(ndim, n_samples)

    Benchmarks:
    d        T        prob(y > T)
    ____________________________
    5       400         8.44E-7
    20      500         1.09E-6
    50      700         3.56E-7
    200     1000        4.85E-6

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'Polynomial product function'
        self.x    = []

    def __str__(self):
        return 'Solver: polynomial product function'

    def run(self, x):
        x = np.array(x)
        y = np.array([ix**4 + ix**2 + 5*ix for ix in x])
        self.y = 0.5 * np.sum(y, axis=0)
        # e = error.samples()
        # self.y = self.y + e

class papaioannou2016sequential(Solver):
    """
    examples tested in paper:
        "Papaioannou, Iason, Costas Papadimitriou, and Daniel Straub. "Sequential importance sampling for structural reliability analysis." Structural safety 62 (2016): 66-75"
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'papaioannou2016sequential'
        self.x    = []

    def __str__(self):
        return "Examples in [papaioannou2016sequential]" 

    def g1(self, x):
        """
        Convex limit-state function
        """
        x0,x1 = np.array(x)
        g = 0.1*(x0-x1) **2 - 1.0/np.sqrt(2)(x1+x0) + 2.5
        return g


    def g2(self,x, b=5, k=0.5, e=0.1):
        """
        parabolic/concave limit-state function

        """
        x0,x1 = np.array(x)
        g = b-x1-k*(x0-e)**2
        return g 
        

    def g3(self,x):
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

    def g4(self,x):
        """
        Noisy limit-state function

        """
        x = np.array(x)
        g = x[0] + 2*x[1] + 2 * x[2] + x[3] -5*x[4] -5*x[5] + 0.001*np.sum(np.sin(100*x),axis=0)
        return g


    def g5(self,x, beta=3.5):
        """
        Linear limit-state function in high dimensions
        """
        x = np.array(x)
        n = x.shape[0]
        g = -1/np.sqrt(n) * np.sum(x, axis=0) + beta
        return g
        

##### fucntion g1-g5 are fom reference: "Papaioannou, Iason, Costas Papadimitriou, and Daniel Straub. "Sequential importance sampling for structural reliability analysis." Structural safety 62 (2016): 66-75"

    def g6(self,x, a=3, mu=1,sigma=0.2):
        x = np.array(x)
        n = x.shape[0]
        g = n + a * sigma * np.sqrt(n) - np.sum(x, axis=0)
        return g


    def g7(self, x, c):
    """
    Multiple design points
    """
    g1 = c -1 - x[1] + np.exp(-x[0]**2/10.0) + (x[0]/5.0)**4
    g2 = c**2/2.0 - x[0] * x[1]
    g  = np.minimum(g1,g2)
    return g



# def bench1(x, error_type):
    # """
    # y = x*sin(x) + e
    # """
    # x = np.array(x)
    # e = gen_error(error_type)
    # y = x * np.sin(x)
    # y = y + e
    # y = y.reshape(x.shape[1],-1)
    # return np.squeeze(y)

# def bench2(x, error_type):
    # """
    # y = x^2 * sin(x) + e
    # """
    # x = np.array(x)
    # e = gen_error(error_type)
    # y = x**2*np.sin(5*x)
    # y = y + e
    # return y

# def bench3(x, error_type):
    # """
    # y = sin(pi/5 * x) + 0.2 * cos(0.8 * pi * x)
    # """
    # x = np.array(x)
    # e = gen_error(error_type)
    # y = np.sin(np.pi/5.0 * x) + 1/5.0 * np.cos(4*np.pi*x/5.0) 
    # y = y + e
    # return y

# def bench4(x, error):
    # """
    # y = 5 + -5*x + 2.5*x^2 -0.36*x^3 + 0.015*x^4
    # """
    # x = np.squeeze(np.array(x))
    # y = 5 + -5*x + 2.5*x**2 -0.36*x**3 + 0.015*x**4
    # e = error.samples()
    # y = y + e
    # return y

# def polynomial_square_root_function(x, error):
    # """
    # y = - [ (-x1+10)**2 + (x2+7)**2 + 10*(x1+x2)  **2 ]**0.5 + 14 
    # x1,x2 ~ N(0,1)
    
    # Benchmarks:
    # prob(y>6) = 2.35E-6
    # """
    # x = np.array(x).reshape(2, -1)
    # x1 = x[0,:]
    # x2 = x[1,:]

    # y = -np.sqrt( (-x1+10)**2 + (x2+7)**2 + 10*(x1+x2)) + 14
    # e = error.samples()
    # y = y + e
    # return y

# def four_branch_system(x, error):
    # """
    # y = 10 - min{ 3 + 0.1(x1 - x2)**2 - (x1+x2)/sqrt(2)
                  # 3 + 0.1(x1 - x2)**2 + (x1+x2)/sqrt(2)
                  # (x1 - x2) + 7/sqrt(2) 
                  # (x2 - x1) + 7/sqrt(2) 
                  # }

    # This toy case allows us to test the ability of the rare event estimation methods to accurately estimate the probability in the case of disconnected failure region 􏱎f .

    # Benchmarks:
    # prob(y > 10) = 2.22E-3
    # prob(y > 12) = 1.18E-6

    # """
    # x = np.array(x).reshape(2,-1)
    # x1 = x[0,:]
    # x2 = x[1,:]

    # y1 = 3 + 0.1*(x1 - x2)**2 - (x1+x2)/np.sqrt(2)
    # y2 = 3 + 0.1*(x1 - x2)**2 + (x1+x2)/np.sqrt(2)
    # y3 = (x1 - x2) + 7.0/np.sqrt(2) 
    # y4 = (x2 - x1) + 7.0/np.sqrt(2) 

    # y = np.array([y1, y2, y3, y4]).min(axis=0)
    # y = 10 - y 

    # e = error.samples()
    # y = y + e
    # return y


# def polynomial_product_function(x, error):
    # """
    # y = 1/2 * sum( xi**4 + xi**2 + 5xi), i = 1, ..., d

    # This toy case is useful to evaluate the ability of the methods to cope with high dimensional problems. 
    # x: ndarray of shape(ndim, n_samples)

    # Benchmarks:
    # d        T        prob(y > T)
    # ____________________________
    # 5       400         8.44E-7
    # 20      500         1.09E-6
    # 50      700         3.56E-7
    # 200     1000        4.85E-6

    # """
    # x = np.array(x)
    # y = np.array([ix**4 + ix**2 + 5*ix for ix in x])
    # y = 0.5 * np.sum(y, axis=0)
    # e = error.samples()
    # y = y + e

    # return y






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
