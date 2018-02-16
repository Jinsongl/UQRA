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

from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
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


dist_x = cp.J(cp.Uniform(-np.pi, np.pi), cp.Uniform(-np.pi, np.pi), cp.Uniform(-np.pi, np.pi))
samp_x = dist_x.sample(2000,'R')
samp_xmc = dist_x.sample(1e6, 'R')
print(samp_x[:3,:])
Y = ishigami(samp_x.T)

scoring = ['neg_mean_squared_error','r2']

## Ordinary least square without normalization X
# methods = ['ols', 'lasso_aic', 'lasso_bic', 'lasso_cv', 'lasso_lars_cv']

# regressors = {'ols':linear_model.LinearRegression(),
        # 'lasso_aic':linear_model.Lasso(),
        # 'lasso_bic':linear_model.Lasso(),
        # 'lasso_cv': linear_model.Lasso(),
        # 'lasso_lars_cv':linear_model.Lasso()} 

# mse_test = {k:[] for k in methods}

n_samples = []
rel_err = []
n_features = []

for p in range(11,12,2):
    poly,norm = cp.orth_ttr(p, dist_x, retall=True)
    N = 3*len(poly)
    n_samples.append(N)
    print(len(poly))
    X = poly(*samp_x[:,:N]).T
    y = Y[:N]
    # err_ = []
    # n_features_ = []
    xmc = poly(*samp_xmc).T

    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=0)

    model_ols = linear_model.LinearRegression()
    model_ols.fit(X_train,y_train)
    y_mc = model_ols.predict(xmc)
    np.savetxt(str(p)+'_ols_y.csv', y_mc, delimiter=',')

    # y_pred = model_ols.predict(X_test)
    # err =  mse(y_pred, y_test)/np.std(y_test)**2
    # print('OLS MSE for p={:2d}:  {:.2e}'.format(p,err)) 
    # err_.append(err)
    # n_features_.append(len(poly))


    # model_cv = linear_model.LassoCV(cv=5).fit(X_train,y_train)
    # model_cv.fit(X_train, y_train)
    # # alpha_aic_ = model_aic.alpha_
    # y_pred = model_cv.predict(X_test)
    # err =  mse(y_pred, y_test)/np.std(y_test)**2
    # print('Lasso CD for p={:2d}:  {:.2e}'.format(p,err)) 

    # err_.append(err)
    # n_features_.append(sum(model_cv.coef_!=0))
    # print(sum(model_cv.coef_!=0))


    # model_lars_cv = linear_model.LassoLarsCV(cv=5).fit(X_train,y_train)
    # model_lars_cv.fit(X_train, y_train)
    # # alpha_aic_ = model_aic.alpha_
    # y_pred = model_lars_cv.predict(X_test)
    # err =  mse(y_pred, y_test)/np.std(y_test)**2
    # # index = np.argwhere(model_lars_cv.coef_path_.T != 0)

    # err_.append(err)
    # n_features_.append(sum(model_lars_cv.coef_!=0))
    # print(sum(model_lars_cv.coef_!=0))

    # print('Lasso Lars for p={:2d}:  {:.2e}'.format(p,err)) 
    # rel_err.append(err_)
    # n_features.append(n_features_)
    # print('-'*30)

# rel_err = np.array(rel_err)
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111)
# number_of_plots=10

# lineObjects = ax1.plot(n_samples, rel_err, '-o')
# labels = ('Full PCE', 'SPCE (Lasso CD)', 'SPCE (Lasso Lars)')
# plt.legend(iter(lineObjects), labels)
# for i in range(len(n_samples)):
    # ax1.text(n_samples[i],rel_err[i,0], str(n_features[i][0]), va='bottom')
    # ax1.text(n_samples[i],rel_err[i,1], str(n_features[i][1]), va='center')
    # ax1.text(n_samples[i],rel_err[i,2], str(n_features[i][2]), va='top' )
# # ymin, ymax = plt.ylim()
# # plt.vlines(xx, ymin, ymax, linestyle='dashed')
# plt.xlabel('Number of model evaluations')
# plt.ylabel('Relative MSE')
# ax1.set_yscale('log')
# plt.grid(True)
# plt.axis('tight')
# # plt.show()
# plt.savefig('SparsePCE4.eps')


# reg =linear_model.Lars()
# reg.fit(X,y)
# print reg.coef_path_.shape
# print 'poly:', poly
# print 'poly_evals', poly(*samp_x).shape
# f_hat = cp.fit_regression(poly,samp_x,y)

# print f_hat.coeffs()
