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
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
import csv
# from sklearn.metrics import r2_score as r2
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


def ishigami(x,p=None):
    p = p or [7, 0.1]
    y = np.sin(x[:,0]) + p[0] * np.sin(x[:,1])**2 + p[1]*x[:,2]**4 * np.sin(x[:,0])
    return y
    
def plot_ic_criterion(model, name, color):
    alpha_ = model.alpha_
    alphas_ = model.alphas_
    criterion_ = model.criterion_
    plt.plot(-np.log10(alphas_), criterion_, '--', color=color,
             linewidth=3, label='%s criterion' % name)
    plt.axvline(-np.log10(alpha_), color=color, linewidth=3,
                label='alpha: %s estimate' % name)
    plt.xlabel('-log(alpha)')
    plt.ylabel('criterion')

def ExPlot(data,q=1e-4, R=1,labels=[],color='k',figHandle=None,figname='ExceedencePlot'):
    if figHandle:
        fig, ax = figHandle
    else:
        fig = plt.figure()  # create a figure object
        ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
    # data = iter_loadtxt(filename)
    M = int(len(data)/R)
    if M < 1.0/q:
        raise ValueError('Not enough samples to get specified quantile ', str(q))

    data = data.reshape((M,R))
    conf=[]

    for i in xrange(R):
        ecdf = ECDF(data[:,i])
        conf.append(next(ecdf.x[i] for i, xx in enumerate(1-ecdf.y) if xx<q))
        x,y = ecdf.x, ecdf.y
        M1  = int(M * 0.9)
        M2  = M - M1
        ind1 = np.linspace(0,M1,num=int(M1/10),dtype=int)
        ind2 = np.linspace(M1+1, M, num=M2, dtype=int)
        ind = np.append(ind1, ind2)
        if labels:
            ax.plot(x, 1-y, color=color, label=labels[i])
        else:
            ax.plot(x, 1-y, color=color)      

    conf.sort()
    q0 = 1.0/M
    print ("Exceedence interval: [", conf[0], conf[-1], " ]")
    ax.plot([conf[0],conf[0]],[q0,q], '--', color='Gray')
    ax.plot([conf[-1],conf[-1]],[q0,q], '--', color='Gray')
    ax.plot([0, conf[-1]],[q,q], '--', color='Gray')
    ax.text(2,1e-5,'$10^{-'+ '{:.1E}'.format(q)[-1] +'}$' +' Exceedence Interval:\n ['+ '{:.2f}'.format(conf[0]) +' , '+ '{:.2f}'.format(conf[-1])+']')
    # ax.text(2,1e-5, '{:.1E}'.format(q) +' Exceedence Interval:\n ['+ '{:.2f}'.format(conf[0]) +' , '+ '{:.2f}'.format(conf[-1])+']')
    ax.set_yscale('log')
    ax.set_xlabel('QoI: $f^{T}_{max}$')
    ax.set_xlim(0,22)
    ax.set_ylabel('Exceedence')
    ax.set_title('Exceedence plot of SDOF system with fixed phases')
    plt.savefig(figname + '.eps')
    return (fig,ax)


def lasso_model_selection(X,y):
    # Normalization of X
    # X -= np.mean(X, axis=0)
    # X /= np.sqrt(np.sum(X**2, axis=0))
    
    
    model_bic = linear_model.LassoLarsIC(criterion='bic')
    model_bic.fit(X, y)
    # alpha_bic_ = model_bic.alpha_

    model_aic = linear_model.LassoLarsIC(criterion='aic')
    model_aic.fit(X, y)
    # alpha_aic_ = model_aic.alpha_
    
    model_cv = linear_model.LassoCV(cv=5).fit(X,y)
    model_lars_cv = linear_model.LassoCV(cv=5).fit(X,y)

    return model_aic, model_bic, model_cv, model_lars_cv


##dist_x = cp.J(cp.Uniform(-np.pi, np.pi), cp.Uniform(-np.pi, np.pi), cp.Uniform(-np.pi, np.pi))
##
##for i in range(10):
##        
##    p=13
##    samp_x = dist_x.sample(1e4,'R')
##    ##print(samp_x[:3,:])
##    Y = ishigami(samp_x.T)
##    np.savetxt(str(p)+'_'+ str(i)+'_y_true.csv', Y, delimiter=',')
##
##    poly,norm = cp.orth_ttr(p, dist_x, retall=True)
##    N = 3*len(poly)
##    print(len(poly))
##    X = poly(*samp_x[:,:N]).T
##    y = Y[:N]
##
##
##    scaler = preprocessing.StandardScaler().fit(X)
##    X = scaler.transform(X)
##    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=0)
##    print('Design Matrix...')
##    X = poly(*samp_x).T
##    
##    print('ols Training...')
##    model_ols = linear_model.LinearRegression()
##    model_ols.fit(X_train,y_train)
##    print('ols Predicting...')
##    y_pred = model_ols.predict(X)
##    np.savetxt(str(p)+'_'+ str(i)+'_ols_y_pred.csv', y_pred, delimiter=',')
##    
##    print('lasso CV Training...')  
##    model_cv = linear_model.LassoCV(cv=5).fit(X_train,y_train)
##    model_cv.fit(X_train, y_train)
##    # alpha_aic_ = model_aic.alpha_
##    y_pred = model_cv.predict(X)
##    np.savetxt(str(p)+'_'+ str(i)+'_lasso_y_pred.csv', y_pred, delimiter=',')
##
##    print('lasso Lars Training...')   
##    model_lars_cv = linear_model.LassoLarsCV(cv=5).fit(X_train,y_train)
##    model_lars_cv.fit(X_train, y_train)
##    # alpha_aic_ = model_aic.alpha_
##    y_pred = model_lars_cv.predict(X)
##    np.savetxt(str(p)+'_'+ str(i)+'_lasso_lars_y_pred.csv', y_pred, delimiter=',')


fin = '13_y_true.csv'
with open(fin) as csvfile:
    fig = plt.figure()  # create a figure object
    ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
    figname='ExceedencePlot_'+fin[:-4]
    data = np.genfromtxt(fin, delimiter=',') 
    q = 1e-3 
    conf=[]
    M,R = data.shape
    labels=[]
    color='k'
    for i in range(R):
        ecdf = ECDF(data[:,i])
        conf.append(next(ecdf.x[i] for i, xx in enumerate(1-ecdf.y) if xx<q))
        x,y = ecdf.x, ecdf.y
        M1  = int(M * 0.9)
        M2  = M - M1
        ind1 = np.linspace(0,M1,num=int(M1/10),dtype=int)
        ind2 = np.linspace(M1+1, M, num=M2, dtype=int)
        ind = np.append(ind1, ind2)
        if labels:
            ax.plot(x, 1-y, color=color, label=labels[i])
        else:
            ax.plot(x, 1-y, color=color)      

    conf.sort()
    q0 = 1.0/M
    print ("Exceedence interval: [", conf[0], conf[-1], " ]")
    ax.plot([conf[0],conf[0]],[q0,q], '--', color='Gray') # first vertical 
    ax.plot([conf[-1],conf[-1]],[q0,q], '--', color='Gray') # second vertical
    ax.plot([-150, conf[-1]],[q,q], '--', color='Gray') # horizontal
    ax.text(-100,1e-2,'$10^{-'+ '{:.1E}'.format(q)[-1] +'}$' +' Exceedence Interval:\n ['+ '{:.2f}'.format(conf[0]) +' , '+ '{:.2f}'.format(conf[-1])+']')
    # ax.text(2,1e-5, '{:.1E}'.format(q) +' Exceedence Interval:\n ['+ '{:.2f}'.format(conf[0]) +' , '+ '{:.2f}'.format(conf[-1])+']')
    ax.set_yscale('log')
    # ax.set_xlabel('QoI: $f^{T}_{max}$')
    ax.set_xlim(-150,200)
    ax.set_ylabel('Exceedence')
    ax.set_title('Exceedence plot for Ishigami' + fin[:-4])
    plt.savefig(figname + '.eps')

fin = '13_lasso_lars_y_pred.csv'
with open(fin) as csvfile:
    fig = plt.figure()  # create a figure object
    ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
    figname='ExceedencePlot'+fin[:-4]
    data = np.genfromtxt(fin, delimiter=',') 
    q = 1e-3 
    conf=[]
    M,R = data.shape
    labels=[]
    color='k'
    for i in range(R):
        ecdf = ECDF(data[:,i])
        conf.append(next(ecdf.x[i] for i, xx in enumerate(1-ecdf.y) if xx<q))
        x,y = ecdf.x, ecdf.y
        M1  = int(M * 0.9)
        M2  = M - M1
        ind1 = np.linspace(0,M1,num=int(M1/10),dtype=int)
        ind2 = np.linspace(M1+1, M, num=M2, dtype=int)
        ind = np.append(ind1, ind2)
        if labels:
            ax.plot(x, 1-y, color=color, label=labels[i])
        else:
            ax.plot(x, 1-y, color=color)      

    conf.sort()
    q0 = 1.0/M
    print ("Exceedence interval: [", conf[0], conf[-1], " ]")
    ax.plot([conf[0],conf[0]],[q0,q], '--', color='Gray') # first vertical 
    ax.plot([conf[-1],conf[-1]],[q0,q], '--', color='Gray') # second vertical
    ax.plot([-150, conf[-1]],[q,q], '--', color='Gray') # horizontal
    ax.text(-100,1e-2,'$10^{-'+ '{:.1E}'.format(q)[-1] +'}$' +' Exceedence Interval:\n ['+ '{:.2f}'.format(conf[0]) +' , '+ '{:.2f}'.format(conf[-1])+']')
    # ax.text(2,1e-5, '{:.1E}'.format(q) +' Exceedence Interval:\n ['+ '{:.2f}'.format(conf[0]) +' , '+ '{:.2f}'.format(conf[-1])+']')
    ax.set_yscale('log')
    # ax.set_xlabel('QoI: $f^{T}_{max}$')
    ax.set_xlim(-150,200)
    ax.set_ylabel('Exceedence')
    ax.set_title('Exceedence plot for Ishigami' + fin[:-4])
    plt.savefig(figname + '.eps')


fin = '13_lasso_y_pred.csv'
with open(fin) as csvfile:
    fig = plt.figure()  # create a figure object
    ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
    figname='ExceedencePlot'+fin[:-4]
    data = np.genfromtxt(fin, delimiter=',') 
    q = 1e-3 
    conf=[]
    M,R = data.shape
    labels=[]
    color='k'
    for i in range(R):
        ecdf = ECDF(data[:,i])
        conf.append(next(ecdf.x[i] for i, xx in enumerate(1-ecdf.y) if xx<q))
        x,y = ecdf.x, ecdf.y
        M1  = int(M * 0.9)
        M2  = M - M1
        ind1 = np.linspace(0,M1,num=int(M1/10),dtype=int)
        ind2 = np.linspace(M1+1, M, num=M2, dtype=int)
        ind = np.append(ind1, ind2)
        if labels:
            ax.plot(x, 1-y, color=color, label=labels[i])
        else:
            ax.plot(x, 1-y, color=color)      

    conf.sort()
    q0 = 1.0/M
    print ("Exceedence interval: [", conf[0], conf[-1], " ]")
    ax.plot([conf[0],conf[0]],[q0,q], '--', color='Gray') # first vertical 
    ax.plot([conf[-1],conf[-1]],[q0,q], '--', color='Gray') # second vertical
    ax.plot([-150, conf[-1]],[q,q], '--', color='Gray') # horizontal
    ax.text(-100,1e-2,'$10^{-'+ '{:.1E}'.format(q)[-1] +'}$' +' Exceedence Interval:\n ['+ '{:.2f}'.format(conf[0]) +' , '+ '{:.2f}'.format(conf[-1])+']')
    # ax.text(2,1e-5, '{:.1E}'.format(q) +' Exceedence Interval:\n ['+ '{:.2f}'.format(conf[0]) +' , '+ '{:.2f}'.format(conf[-1])+']')
    ax.set_yscale('log')
    # ax.set_xlabel('QoI: $f^{T}_{max}$')
    ax.set_xlim(-150,200)
    ax.set_ylabel('Exceedence')
    ax.set_title('Exceedence plot for Ishigami' + fin[:-4])
    plt.savefig(figname + '.eps')

fin = '13_ols_y_pred.csv'
with open(fin) as csvfile:
    fig = plt.figure()  # create a figure object
    ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
    figname='ExceedencePlot'+fin[:-4]
    data = np.genfromtxt(fin, delimiter=',') 
    q = 1e-3 
    conf=[]
    M,R = data.shape
    labels=[]
    color='k'
    for i in range(R):
        ecdf = ECDF(data[:,i])
        conf.append(next(ecdf.x[i] for i, xx in enumerate(1-ecdf.y) if xx<q))
        x,y = ecdf.x, ecdf.y
        M1  = int(M * 0.9)
        M2  = M - M1
        ind1 = np.linspace(0,M1,num=int(M1/10),dtype=int)
        ind2 = np.linspace(M1+1, M, num=M2, dtype=int)
        ind = np.append(ind1, ind2)
        if labels:
            ax.plot(x, 1-y, color=color, label=labels[i])
        else:
            ax.plot(x, 1-y, color=color)      

    conf.sort()
    q0 = 1.0/M
    print ("Exceedence interval: [", conf[0], conf[-1], " ]")
    ax.plot([conf[0],conf[0]],[q0,q], '--', color='Gray') # first vertical 
    ax.plot([conf[-1],conf[-1]],[q0,q], '--', color='Gray') # second vertical
    ax.plot([-150, conf[-1]],[q,q], '--', color='Gray') # horizontal
    ax.text(-100,1e-2,'$10^{-'+ '{:.1E}'.format(q)[-1] +'}$' +' Exceedence Interval:\n ['+ '{:.2f}'.format(conf[0]) +' , '+ '{:.2f}'.format(conf[-1])+']')
    # ax.text(2,1e-5, '{:.1E}'.format(q) +' Exceedence Interval:\n ['+ '{:.2f}'.format(conf[0]) +' , '+ '{:.2f}'.format(conf[-1])+']')
    ax.set_yscale('log')
    # ax.set_xlabel('QoI: $f^{T}_{max}$')
    ax.set_xlim(-150,200)
    ax.set_ylabel('Exceedence')
    ax.set_title('Exceedence plot for Ishigami' + fin[:-4])
    plt.savefig(figname + '.eps')

