#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

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
"""
import numpy.linalg as la 
import numpy as np
import chaospy as cp
from sklearn import linear_model
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

scoring = ['mean_squared_error','r2_score']


def ishigami(x,p=None):
    p = p or [7, 0.1]
    y = np.sin(x[:,0]) + p[0] * np.sin(x[:,1])**2 + p[1]*x[:,2]**4 * np.sin(x[:,0])
    return y
    
def testfunc(x):
    return 3.0 + 2.3*x[:,0] + 5*x[:,0]*x[:,1] 


def fit_model(X, y):
    '''
    X: numpy array of shape [n_samples,n_features]]
    y: numpy array of shape [n_samples,]
    '''
    H = la.inv(np.dot(X.T, X))
    H = np.dot(H, X.T)
    beta_hat = np.dot(H, y)
    H = np.dot(X, H)
    e = y - np.dot(X, beta_hat) # residual
    l2_error = la.norm(e)

    loo_error = 1 - np.diagonal(H)
    loo_error = e / loo_error 
    loo_error = (loo_error**2).mean()
    rloo_error =  loo_error/ (np.std(y)**2)

    N,P = X.shape
    C = 1.0/N * np.dot(X.T, X)
    T = N/(N-P) * (1+np.trace(la.inv(C)) /N)
    loo_error_corrected = loo_error * T
    rloo_error_corrected = loo_error_corrected / (np.std(y)**2)
    return beta_hat, l2_error, loo_error, rloo_error, loo_error_corrected, rloo_error_corrected


# def standardize(x):
    # '''
    # Assume only first column could be constant
    # if x design matrix
        # x: numpy array of shape [n_samples,n_features]]
    # if x observations
        # y: numpy array of shape [n_samples,]
    # '''
    # if x.ndim == 1:
        # mu = np.mean(x)
        # sigma = None if np.std(x)==0 else np.std(x)  
        # if sigma is None:
            # return np.ones(x.shape), mu, sigma
        # else:
            # return (x-mu)/sigma, mu, sigma 
    # else:
        # mu = np.mean(x, axis=0)
        # sigma = np.std(x, axis=0)

        # if sigma[0] == 0:
            # x[:,0] = x[:,0] / x[0,0]
            # x[:,1:] = (x[:,1:] - mu[1:]) / sigma[1:]
        # else:
            # x = (x - mu)/sigma
        # return x, mu, sigma

    # zero_sigma_index = np.where(sigma==0)
    # if not len(zero_sigma_index[0]):
# # have zero sigma
        
    # if any(sigma==0):
        # if x.ndim == 1:
            # print("constant array")
            # return np.zeros(x.shape), x[0], None
        # else:
            # return (x-mu)/sigma, mu, sigma

    # else:
        # return (x-mu)/sigma, mu ,sigma


def ErrCorr(Errloo, X):
    N, P = X.shape
    C = 1.0/N * np.dot(X.T,X)
    T = N/(N-P) * (1 + np.trace(la.inv(C))/N)
    return Errloo * T



dist_x = cp.J(cp.Uniform(-np.pi, np.pi), cp.Uniform(-np.pi, np.pi), cp.Uniform(-np.pi, np.pi))
# dist_x = cp.J(cp.Uniform(-np.pi, np.pi), cp.Uniform(-np.pi, np.pi))
# dist_x = cp.J(cp.Uniform(0, 1), cp.Uniform(0, 1), cp.Uniform(0, 1))
samp_x = dist_x.sample(1200,'S')

y = ishigami(samp_x.T)
# y = testfunc(samp_x.T)
# poly, norm = cp.orth_ttr(5,dist_x, retall=True)
# X = poly(*samp_x).T

# # Standardize X and y
# y = (y - np.mean(y))/np.std(y)
# X = (X[:,1:] - np.mean(X[:,1:], axis=0))/np.std(X[:,1:], axis=0)
# print(X.mean(axis=0))

# XYCoef = []
# for i in range(X.shape[0]):
    # row = X[i,:]
    # coef = abs(np.dot(row,y))/(la.norm(y)*la.norm(row))
    # XYCoef.append([i,coef])

# XYCoef = np.nan_to_num(XYCoef)
# XYCoef = np.array(XYCoef)
# print XYCoef[np.argsort(XYCoef[:,1])][-10:,:]

for p in range(13,14):
    poly,norm = cp.orth_ttr(p, dist_x, retall=True)
    # print(poly)
    # print(norm)
    N = 2*len(poly)
    # print(samp_x[:,:N].T[:6,:])
    X = poly(*samp_x[:,:N]).T
    # print(X[:6,:6])
    check = np.dot(X.T, X)[:6,:6]
    # print('---Check----')
    # print(check/N)
    X = X/np.sqrt(norm)
    # print(X[:6,:6])
    # print(np.dot(X.T,X))
    Y = y[:N]
    # print(np.std(Y)**2)
    # # Standardize X and y
    # X, X_mu, X_sigma = standardize(X) 
    # print('Normalized X:\n')
    # print(X)
    # print('--'*20)
    # print(X_mu)
    # print('--'*20)
    # print(X_sigma)
    # Y, Y_mu, Y_sigma = standardize(Y)
    # print('Normalized Y:\n', Y, Y_mu, Y_sigma)
    beta_hat, _, _, rloo, loo_cor, rloo_cor = fit_model(X,Y)
    # print("relative $L^2$ Error:", rloo/N)
    # print("Corrected $L^2$ Error:", loo_cor/N)
    print("Corrected relative $L^2$ Error: {:.2e}".format(rloo_cor))


    # print e
    # f_hat = cp.fit_regression(poly,X,Y)
    # print f_hat
    # y_hat = f_hat(*samp_x[:,:P])
    # print 'Predict Y:', y_hat
    # print '-' * 20
    # res = Y - y_hat
    # print y_hat
    # e = Errloo(X.T,Y)
    # e = ErrCorr(e,X)
    
    # y_hat = f_hat(*X)
    # L2 = la.norm(Y-y_hat)
    
    

# print("Computing regularization path using the LARS ...")
# alphas, _, coefs = linear_model.lars_path(X.T, y, method='lasso', verbose=True)

# for i in range(10):
    # col = coefs[:,i]
    # print np.nonzero(col)
# xx = np.sum(np.abs(coefs.T), axis=1)
# xx /= xx[-1]


# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111)
# number_of_plots=10
# # colormap = plt.cm.nipy_spectral #I suggest to use nipy_spectral, Set1,Paired
# colormap = plt.cm.jet #I suggest to use nipy_spectral, Set1,Paired
# ax1.set_color_cycle([colormap(i) for i in np.linspace(0, 1,number_of_plots)])

# lineObjects = ax1.plot(xx, coefs.T)
# # labels = tuple(str(i) for i in range(coefs.shape[1]))
# # plt.legend(iter(lineObjects), labels)
# ymin, ymax = plt.ylim()
# plt.vlines(xx, ymin, ymax, linestyle='dashed')
# plt.xlabel('|coef| / max|coef|')
# plt.ylabel('Coefficients')
# plt.title('LASSO Path')
# plt.axis('tight')
# plt.show()


# reg =linear_model.Lars()
# reg.fit(X,y)
# print reg.coef_path_.shape
# print 'poly:', poly
# print 'poly_evals', poly(*samp_x).shape
# f_hat = cp.fit_regression(poly,samp_x,y)

# print f_hat.coeffs()
