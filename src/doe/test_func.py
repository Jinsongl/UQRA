#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""
Testing function for DOE 


"""
import numpy as np
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
def conner_peak(x, *kwargs):
    """
    Arguments:
    x: array of shape (nsamples,dimensions)

    Return:
    y: array of shape(nsamples,)

    """
    n,d = x.shape
    if len(kwargs) == 0:
        c = np.array([1/(i**2) for i in range(1,d+1)]).reshape((d,1))
    else:
        c = kwargs['c']
        c = np.array(c).reshape((1,d))

    x = (x + 1.0)/2.0
    s = np.squeeze(1.0 + np.dot(x,c))
    assert len(s) == n
    y = s**(-d-1)

    return y

def product_peak(x,*kwargs):
    """
    Arguments:
    x: array of shape (nsamples, dimensions) (n,d)
    c: parameters controlling the difficulty of the functions (1,d)
    w: shifting parameters (1,d)

    Return:
    y: array of shape (n,1)

    """
    n,d = x.shape
    if len(kwargs) == 0:
        c = np.ones(1,d)
        w = np.ones(1,d)
    else:
        c = kwargs['c']
        w = kwargs['w']
        c = np.array(c).reshape((1,d))
        w = np.array(w).reshape((1,d))

    x = ((x+1.0)/2 - w)**2
    c = 1.0/c**(2)
    p = (c + x)**(-1)

    assert (n,d) == p.shape
    y = np.prod(p,axis=1)

    return y

def gaussian_peak(x,*kwargs):

    """
    Arguments:
    x: array of shape (nsamples, dimensions) (n,d)
    c: parameters controlling the difficulty of the functions (1,d)
    w: shifting parameters (1,d)

    Return:
    y: array of shape (n,1)

    **************************************************************************  
    Description:
    Dimensions: d 

    The Gaussian Peak family of functions is quickly integrated analytically, to high precision. In its two-dimensional form, as shown in the plot above, the function shows a distinct two-dimensional Gaussian peak in the centre of the integration region. 

    The d uneffective u-parameters are chosen randomly from [0, 1], and their values don't significantly affect the difficulty of integration. 

    Larger values of the d affective a-parameters result in a sharper Gaussian peak, thus resulting in more difficult integration. 


    Reference:
    Genz, A. (1984, September). Testing multidimensional integration routines. In Proc. of international conference on Tools, methods and languages for scientific and engineering computation (pp. 81-94). Elsevier North-Holland, Inc..
    **************************************************************************  
    """
    n,d = x.shape

    if len(kwargs) == 0:
        c = np.ones(1,d)
        w = np.ones(1,d)
    else:
        c = kwargs['c']
        w = kwargs['w']
        c = np.array(c).reshape((1,d))
        w = np.array(w).reshape((1,d))

    # print('\t Gaussian peak function: dimension = {:5d}, number of samples {:5d}'.format(d,n))
    c = np.array(c).reshape((d,1))**2
    w = np.array(w).reshape((d,1))
    x = ((x + 1.0)/2.0 - w.T)**2
    s = -np.squeeze(np.dot(x,c))
    assert len(s) == n
    y = np.exp(s) 
    return y



def ishigami(x,p=None):
    if len(args) == 0:
        p = [7, 0.1]
    else:
        p = args[0]
        assert len(p) == 2

    y = np.sin(x[:,0]) + p[0] * np.sin(x[:,1])**2 + p[1]*x[:,2]**4 * np.sin(x[:,0])
    return y
    
def franke2d(xx, *args):
    """
    Argument:
    xx: array of shape (n,2)
    
    Return:
    y: array of shape(n,), evaluated at each (xx[i,0],xx[i,1]) pair
    Description:
    Dimensions: 2 
    Franke's function has two Gaussian peaks of different heights, and a smaller dip. It is used as a test function in interpolation problems. 

    References:
    Franke, R. (1979). A critical comparison of some methods for interpolation of scattered data (No. NPS53-79-003). NAVAL POSTGRADUATE SCHOOL MONTEREY CA.

    Haaland, B., & Qian, P. Z. (2011). Accurate emulators for large-scale computer experiments. The Annals of Statistics, 39(6), 2974-3002.
    """
    xx = np.array(xx, dtype=float)
    if xx.shape[1] != 2:
        xx = xx.T
    assert xx.shape[1]==2, "2 variables required for franke function"
    x1 = xx[:,0]
    x2 = xx[:,1]

    term1 = 0.75 * np.exp(-(9*x1-2.0)**2/4.0 - (9*x2-2.0)**2/4.0);
    term2 = 0.75 * np.exp(-(9*x1+1.0)**2/49.0 - (9*x2+1.0)/10.0);
    term3 = 0.50 * np.exp(-(9*x1-7.0)**2/4.0 - (9*x2-3.0)**2/4.0);
    term4 = -0.2 * np.exp(-(9*x1-4.0)**2.0 - (9*x2-7.0)**2);

    y = term1 + term2 + term3 + term4;
    return y

def plot_func(func, *args):
    x = np.linspace(-1,1,1000, endpoint=True)
    xx, yy = np.meshgrid(x,x)
    m,n = xx.shape
    z = np.zeros(xx.shape)
    z = np.array([func(np.array([x,y]).reshape(2,1), *args) for x, y in zip(np.ravel(xx), np.ravel(yy))])
    Z = z.reshape(xx.shape)
    # for i in range(m):
        # for j in range(n):
            # z[i,j] = func([xx[i,j],yy[i,j]], *arg)

    # z = func(xx,*args)

    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.scatter(x,x,z)
    ax.plot_surface(xx,yy,z)

    plt.show()
