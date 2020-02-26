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
import scipy.stats as stats
from museuq.solver._solverbase import SolverBase
from museuq.utilities.helpers import isfromstats
import random

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

class Ishigami(SolverBase):
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
    def __init__(self, p=[7,0.1]):
        super().__init__()
        self.name = 'Ishigami'
        self.nickname = 'Ishigami'
        self.ndim = int(3)
        self.p    = p
        self.distributions = [stats.uniform(-np.pi, np.pi),] * self.ndim

     
    def __str__(self):
        return 'solver: Ishigami function (p={})'.format(self.p)

    def run(self, x1, x2, x3):
        y = np.sin(x1) + self.p[0] * np.sin(x2)**2 + self.p[1]*x3**4 * np.sin(x1)
        return y

    def map_domain(self, u, dist_u):
        """
        mapping random variables u from distribution dist_u (default U(0,1)) to self.distributions 
        Argument:
            two options:
            1. cdf(u)
            2. u and dist_u
        """
        u = np.array(u, copy=False, ndmin=2)
        if isinstance(dist_u, (list, tuple)):
            ## if a list is given but not enough distributions, appending with Uniform(0,1)
            for idist in dist_u:
                assert isfromstats(idist)
            for _ in range(len(dist_u), self.ndim):
                dist_u.append(stats.uniform(0,1))
        else:
            assert isfromstats(dist_u)
            dist_u = [dist_u,] * self.ndim
        u_cdf = np.array([idist.cdf(iu) for iu, idist in zip(u, dist_u)])
        assert (u_cdf.shape[0] == self.ndim), '{:s} expecting {:d} random variables, {:d} given'.format(self.name, self.ndim, u_cdf.shape[0])
        x = np.array([idist.ppf(iu_cdf)  for iu_cdf, idist in zip(u_cdf, self.distributions)])
        return x

class xsinx(SolverBase):
    """
    y = x*sin(x)
    """
    def __init__(self):
        super().__init__()
        self.name = 'xsinx'
        self.nickname = 'xsinx'
        self.ndim = int(1)
        self.distributions = [stats.uniform(-np.pi, np.pi),] * self.ndim

    def __str__(self):
        return 'solver: x*sin(x)'

    def run(self, x):
        x = np.array(x,copy=False, ndmin=1)
        y = x * np.sin(x)
        return y

    def map_domain(self, u_cdf=None, u=None, dist_u=None):
        """
        mapping random variables u from distribution dist_u (default U(0,1)) to self.distributions 
        Argument:
            two options:
            1. cdf(u)
            2. u and dist_u
        """

        u_cdf = []
        u = np.array(u, copy=False, ndmin=2)
        if isinstance(dist_u, (list, tuple)):
            for _ in range(len(dist_u), self.ndim):
                dist_u.append(stats.uniform(0,1))
        else:
            assert hasattr(stats, dist_u.dist.name)
            dist_u = [dist_u,] * self.ndim

        for iu, idist in zip(u, dist_u):
            assert hasattr(stats, idist.dist.name)
            u_cdf.append(idist.cdf(iu))
        u_cdf = np.array(u_cdf)
        assert (u_cdf.shape[0] == self.ndim), '{:s} expecting {:d} random variables, {:d} given'.format(self.name, self.ndim, u_cdf.shape[0])
        x = np.array([idist.ppf(iu_cdf)  for iu_cdf, idist in zip(u_cdf, self.distributions)])
        return x

class sparse_poly(SolverBase):
    """
    Sparse Polynomial
    """
    def __init__(self, poly, coef= stats.norm, sparsity='full'):
        self.name = 'sparse polynomial'
        self.nickname = 'sparse_poly'
        self.poly = poly
        self.ndim = poly.ndim
        self.deg  = poly.deg
        self.num_basis = poly.num_basis
        if self.poly.name == 'Legendre':
            self.distributions = [stats.uniform(-1,1),] * self.ndim
        elif self.poly.name == 'Hermite':
            self.distributions = [stats.norm(), ] * self.ndim
        else:
            raise NotImplementedError

        if isfromstats(coef):
            k = self.num_basis if sparsity == 'full' else sparsity
            coef = self._random_coef(k)
            self.poly.set_coef(coef)
        else:
            self.poly.set_coef(coef)

    def __str__(self):
        return 'solver: sparse polynomial function'

    def run(self, x):
        y = self.poly(x)
        return y

    def _random_coef(self, k, dist=stats.norm, seed=None, theta=(0,1)):
        """
        p: total order    
        s: sparsity
        """
        np.random.seed(seed)
        coef = dist.rvs(loc=theta[0], scale=theta[1], size=self.num_basis)
        coef[random.sample(range(0, self.num_basis), self.num_basis - k)] = 0.0
        return coef

    def map_domain(self, u, dist_u):
        """
        mapping random variables u from distribution dist_u (default U(0,1)) to self.distributions 
        Argument:
            two options:
            1. cdf(u)
            2. u and dist_u
        """
        u = np.array(u, copy=False, ndmin=2)
        if isinstance(dist_u, (list, tuple)):
            ## if a list is given but not enough distributions, appending with Uniform(0,1)
            for idist in dist_u:
                assert isfromstats(idist)
            for _ in range(len(dist_u), self.ndim):
                dist_u.append(stats.uniform(0,1))
        else:
            assert isfromstats(dist_u)
            dist_u = [dist_u,] * self.ndim
        u_cdf = np.array([idist.cdf(iu) for iu, idist in zip(u, dist_u)])
        assert (u_cdf.shape[0] == self.ndim), '{:s} expecting {:d} random variables, {:d} given'.format(self.name, self.ndim, u_cdf.shape[0])
        x = np.array([idist.ppf(iu_cdf)  for iu_cdf, idist in zip(u_cdf, self.distributions)])
        return x

class poly4th(SolverBase):
    """
    y = 5 + -5*x + 2.5*x^2 -0.36*x^3 + 0.015*x^4
    """

    def __init__(self):
        super().__init__()
        self.name = 'polynomial'
        self.nickname = 'Poly4'
        self.ndim = int(1)
        self.distributions = [stats.norm(2, 4),] * self.ndim

    def __str__(self):
        return 'solver: 4th-order polynomial function'

    def run(self, x):
        x = np.array(x, copy=False, ndmin=1)
        y = 5 + -5*x + 2.5*x**2 -0.36*x**3 + 0.015*x**4
        return y

    def map_domain(self, u_cdf=None, u=None, dist_u=None):
        """
        mapping random variables u from distribution dist_u (default U(0,1)) to self.distributions 
        Argument:
            two options:
            1. cdf(u)
            2. u and dist_u
        """

        u_cdf = []
        u = np.array(u, copy=False, ndmin=2)
        if isinstance(dist_u, (list, tuple)):
            for _ in range(len(dist_u), self.ndim):
                dist_u.append(stats.uniform(0,1))
        else:
            assert hasattr(stats, dist_u.dist.name)
            dist_u = [dist_u,] * self.ndim

        for iu, idist in zip(u, dist_u):
            assert hasattr(stats, idist.dist.name)
            u_cdf.append(idist.cdf(iu))
        u_cdf = np.array(u_cdf)
        assert (u_cdf.shape[0] == self.ndim), '{:s} expecting {:d} random variables, {:d} given'.format(self.name, self.ndim, u_cdf.shape[0])
        x = np.array([idist.ppf(iu_cdf)  for iu_cdf, idist in zip(u_cdf, self.distributions)])
        return x


class polynomial_square_root_function(SolverBase):
    """
    y = - [ (-x1+10)**2 + (x2+7)**2 + 10*(x1+x2)  **2 ]**0.5 + 14 
    x1,x2 ~ N(0,1)
    
    Benchmarks:
    prob(y>6) = 2.35E-6
    """

    def __init__(self):
        super().__init__()
        self.name = 'polynomial square root function'
        self.nickname = 'PolySqrt'
        self.ndim = int(2)
        self.distributions = [stats.norm(),] * self.ndim

    def __str__(self):
        return 'solver: Polynomial square root function'

    def run(self, x):
        x = np.array(x, copy=False)
        x1 = x[0,:]
        x2 = x[1,:]
        y = -((-x1+10)**2 + (x2+7)**2 + 10*(x1+x2)  **2)**0.5 + 14 
        return y


    def map_domain(self, u_cdf=None, u=None, dist_u=None):
        """
        mapping random variables u from distribution dist_u (default U(0,1)) to self.distributions 
        Argument:
            two options:
            1. cdf(u)
            2. u and dist_u
        """

        u_cdf = []
        u = np.array(u, copy=False, ndmin=2)
        if isinstance(dist_u, (list, tuple)):
            for _ in range(len(dist_u), self.ndim):
                dist_u.append(stats.uniform(0,1))
        else:
            assert hasattr(stats, dist_u.dist.name)
            dist_u = [dist_u,] * self.ndim

        for iu, idist in zip(u, dist_u):
            assert hasattr(stats, idist.dist.name)
            u_cdf.append(idist.cdf(iu))
        u_cdf = np.array(u_cdf)
        assert (u_cdf.shape[0] == self.ndim), '{:s} expecting {:d} random variables, {:d} given'.format(self.name, self.ndim, u_cdf.shape[0])
        x = np.array([idist.ppf(iu_cdf)  for iu_cdf, idist in zip(u_cdf, self.distributions)])
        return x

class four_branch_system(SolverBase):
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

    def __init__(self):
        super().__init__()
        self.name = 'four branch system'
        self.nickname = 'Branches'
        self.ndim = int(2)
        self.distributions = [stats.norm(),] * self.ndim

    def __str__(self):
        return 'Solver: four branch system'

    def run(self, x):

        x = np.array(x, ndim=2, copy=False).reshape(2,-1)
        x1 = x[0,:]
        x2 = x[1,:]

        y1 = 3 + 0.1*(x1 - x2)**2 - (x1+x2)/np.sqrt(2)
        y2 = 3 + 0.1*(x1 - x2)**2 + (x1+x2)/np.sqrt(2)
        y3 = (x1 - x2) + 7.0/np.sqrt(2) 
        y4 = (x2 - x1) + 7.0/np.sqrt(2) 

        y = np.array([y1, y2, y3, y4]).min(axis=0)
        y = 10 - y 
        return y

    def map_domain(self, u_cdf=None, u=None, dist_u=None):
        """
        mapping random variables u from distribution dist_u (default U(0,1)) to self.distributions 
        Argument:
            two options:
            1. cdf(u)
            2. u and dist_u
        """

        u_cdf = []
        u = np.array(u, copy=False, ndmin=2)
        if isinstance(dist_u, (list, tuple)):
            for _ in range(len(dist_u), self.ndim):
                dist_u.append(stats.uniform(0,1))
        else:
            assert hasattr(stats, dist_u.dist.name)
            dist_u = [dist_u,] * self.ndim

        for iu, idist in zip(u, dist_u):
            assert hasattr(stats, idist.dist.name)
            u_cdf.append(idist.cdf(iu))
        u_cdf = np.array(u_cdf)
        assert (u_cdf.shape[0] == self.ndim), '{:s} expecting {:d} random variables, {:s} given'.format(self.name, self.ndim, u_cdf.shape[0])
        x = np.array([idist.ppf(iu_cdf)  for iu_cdf, idist in zip(u_cdf, self.distributions)])
        return x

    
class polynomial_product_function(SolverBase):
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

    def __init__(self, d):
        super().__init__()
        self.name = 'Polynomial product function'
        self.nickname = 'PolyProd'
        self.ndim = int(d)
        self.distributions = [stats.norm(),] * self.ndim

    def __str__(self):
        return 'Solver: polynomial product function'

    def run(self, x):
        x = np.array(x, copy=False)
        assert x.shape[0] == self.ndim
        y = np.array([ix**4 + ix**2 + 5*ix for ix in x])
        y = 0.5 * np.sum(y, axis=0)
        return y

    def map_domain(self, u_cdf=None, u=None, dist_u=None):
        """
        mapping random variables u from distribution dist_u (default U(0,1)) to self.distributions 
        Argument:
            two options:
            1. cdf(u)
            2. u and dist_u
        """
        u_cdf = []
        u = np.array(u, copy=False, ndmin=2)
        if isinstance(dist_u, (list, tuple)):
            for _ in range(len(dist_u), self.ndim):
                dist_u.append(stats.uniform(0,1))
        else:
            assert hasattr(stats, dist_u.dist.name)
            dist_u = [dist_u,] * self.ndim

        for iu, idist in zip(u, dist_u):
            assert hasattr(stats, idist.dist.name)
            u_cdf.append(idist.cdf(iu))
        u_cdf = np.array(u_cdf)

        assert (u_cdf.shape[0] == self.ndim), '{:s} expecting {:d} random variables, {:d} given'.format(self.name, self.ndim, u_cdf.shape[0])
        x = np.array([idist.ppf(iu_cdf)  for iu_cdf, idist in zip(u_cdf, self.distributions)])
        return x

class papaioannou2016sequential(SolverBase):
    """
    examples tested in paper:
        "Papaioannou, Iason, Costas Papadimitriou, and Daniel Straub. "Sequential importance sampling for structural reliability analysis." Structural safety 62 (2016): 66-75"
    """
    def __init__(self):
        super().__init__()
        self.name = 'papaioannou2016sequential'

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


class franke(SolverBase):
    """
    Franke Function

    """
    def __init__(self):
        super(franke, self).__init__()
        self.name = 'Franke'
        self.nickname = 'Franke'
        self.ndim = int(2)
        self.distributions = [stats.uniform(-1,2),] * self.ndim

    def __str__(self):
        return 'solver: franke function'

    def run(self, x):

        x = np.array(x, copy=0, ndmin=2)
        x1,x2 = x
        f1 = 0.75 * np.exp(-(9.0*x1 -2.0)**2/ 4.0 - (9.0*x2 -2)**2/4.0)
        f2 = 0.75 * np.exp(-(9.0*x1 +1.0)**2/49.0 - (9.0*x2 +1.0)/10.0)
        f3 = 0.50 * np.exp(-(9.0*x1 -7.0)**2/ 4.0 - (9.0*x2 -3.0)**2 /4.0)
        f4 = 0.20 * np.exp(-(9.0*x1 -4.0)**2 - (9.0*x2 -7.0)**2)
        f  =  f1 + f2 + f3 - f4
        return f


    def map_domain(self, u, dist_u):
        """
        mapping random variables u from distribution dist_u (default U(0,1)) to self.distributions 
        Argument:
            two options:
            1. cdf(u)
            2. u and dist_u
        """

        u_cdf = []
        u = np.array(u, copy=False, ndmin=2)
        if isinstance(dist_u, (list, tuple)):
            for _ in range(len(dist_u), self.ndim):
                dist_u.append(stats.uniform(0,1))
        else:
            assert hasattr(stats, dist_u.dist.name)
            dist_u = [dist_u,] * self.ndim

        for iu, idist in zip(u, dist_u):
            assert hasattr(stats, idist.dist.name)
            u_cdf.append(idist.cdf(iu))
        u_cdf = np.array(u_cdf)
        assert (u_cdf.shape[0] == self.ndim), '{:s} expecting {:d} random variables, {:d} given'.format(self.name, self.ndim, u_cdf.shape[0])
        x = np.array([idist.ppf(iu_cdf)  for iu_cdf, idist in zip(u_cdf, self.distributions)])
        return x



