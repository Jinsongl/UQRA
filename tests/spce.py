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
import operator as op
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

def ncr(n, r):
    from functools import reduce
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, range(n, n-r, -1))
    denom = reduce(op.mul, range(1, r+1))
    return numer//denom


def find_p_max(x,m, alpha=3):
    """
    solve: 
        alpha * comb(m+p,p) <= x
    """
    x = int(x/alpha)
    p = 0
    z = ncr(m+p,p)
    while z < x:
        p += 1
        z = ncr(m+p,p)
    return p
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
def error_loo(X, y_true, y_pred):
    H = np.dot(X.T,X)
    H = la.inv(H)
    H = np.dot(H,X.T)
    H = np.dot(X,H)
    h = np.diag(H)
    h = 1 - h
    e = ((y_true - y_pred)/h)**2
    return e.mean()


scoring = ['neg_mean_squared_error','r2']
## Set up input random variables: 3 independent uniform [-pi,pi]]
dist_unif = cp.Uniform(-np.pi, np.pi)
dist_x  = cp.J(cp.Uniform(-np.pi, np.pi),cp.Uniform(-np.pi, np.pi),cp.Uniform(-np.pi, np.pi))
n_dim   = len(dist_x)
## Initialization of experimental design
alpha = 3
n_cv    = 5
n_samp  = 10000
e_target = 1e1
max_iter = 10
i_iter = 0
err = np.inf

n_samp_all = []
err_all = []
n_features = []
p_max   = find_p_max(n_samp,n_dim) 
samp_x  = dist_x.sample(n_samp,'S')


poly,norm = cp.orth_ttr(2, dist_x, retall=True)
X = poly(*samp_x).T
y = ishigami(samp_x.T)
test = la.inv(np.dot(X.T, X))
np.set_printoptions(precision=3)
print(test)



poly,norm = cp.orth_ttr(p_max, dist_x, retall=True)
X = poly(*samp_x).T
y = ishigami(samp_x.T)

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)
sigma2 = np.std(y_test)**2
p = 1
while err > e_target and (i_iter < max_iter):
    err_=[]
    n_features_ = []

    print('-'*30)
    print('Experimental design size: {:4d}'.format(n_samp))
    n_samp_all.append(n_samp)

    poly,norm   = cp.orth_ttr(p, dist_x, retall=True)
    n_poly      = len(poly)

    if n_samp < alpha * n_poly:
        new_samp = dist_x.sample(alpha * n_poly - n_samp, 'S') 
        samp_x = np.hstack((samp_x, new_samp))
        n_samp = alpha * n_poly
        # print(new_samp[:,:4])
        X = poly(*samp_x).T
        y = ishigami(samp_x.T)

        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)
        sigma2 = np.std(y_test)**2
        
    X_train_ = X_train[:,:n_poly]
    X_test_  = X_test[:,:n_poly]
    model_ols = linear_model.LinearRegression().fit(X_train_,y_train)
    y_pred = model_ols.predict(X_test_)
    err_ols =  mse(y_pred, y_test)/sigma2
    n_features_.append(np.count_nonzero(model_ols.coef_))

    err_.append(err_ols)


    model_lars_cv = linear_model.LassoLarsCV(cv=n_cv).fit(X_train_,y_train)
    y_pred  = model_lars_cv.predict(X_test_)
    err_lar   = mse(y_pred, y_test)/sigma2
    # print(np.nonzero(model_lars_cv.coef_))
    n_features_.append(np.count_nonzero(model_lars_cv.coef_))

    err_.append(err_lar)

    nonzero_ind = np.nonzero(model_lars_cv.coef_)[0]
    X_ = X_train_[:,nonzero_ind]
    model_olsr = linear_model.LinearRegression().fit(X_,y_train)
    y_pred = model_olsr.predict(X_test_[:,nonzero_ind])
    err_olsr =  mse(y_pred, y_test)/sigma2
    # print(np.count_nonzero(model_ols.coef_))

    err_.append(err_olsr)
    err_all.append(err_)
    n_features.append(n_features_)
    print('p={:2d}, ols test dataset relative mse={:.4e}'.format(p,err_ols))
    print('p={:2d}, olsr test dataset relative mse={:.4e}'.format(p,err_olsr))
    print('p={:2d}, lasso test dataset relative mse={:.4e}'.format(p,err_lar))
    err = max(err_lar, err_ols)
    p += 1


err_all = np.array(err_all)
n_samp_all = np.array(n_samp_all)
n_features = np.array(n_features)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
number_of_plots=10

lineObjects = ax1.plot(n_samp_all, err_all, '-o')
labels = ('Full PCE', 'SPCE (Lasso Lar)', 'SPCE (Lasso OLS)')
plt.legend(iter(lineObjects), labels)
for i in range(len(n_samp_all)):
    ax1.text(n_samp_all[i],err_all[i,0], str(n_features[i][0]), va='bottom')
    # ax1.text(n_samp_all[i],err_all[i,1], str(n_features[i][1]), va='center')
    ax1.text(n_samp_all[i],err_all[i,2], str(n_features[i][1]), va='top' )
# ymin, ymax = plt.ylim()
# plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('Number of model evaluations')
plt.ylabel('Relative test MSE')
ax1.set_yscale('log')
plt.grid(True)
plt.axis('tight')
plt.show()
# plt.savefig('SparsePCE4.eps')























## Ordinary least square without normalization X
# methods = ['ols', 'lasso_aic', 'lasso_bic', 'lasso_cv', 'lasso_lars_cv']

# regressors = {'ols':linear_model.LinearRegression(),
        # 'lasso_aic':linear_model.Lasso(),
        # 'lasso_bic':linear_model.Lasso(),
        # 'lasso_cv': linear_model.Lasso(),
        # 'lasso_lars_cv':linear_model.Lasso()} 

# mse_test = {k:[] for k in methods}


# n_samples = []
# rel_err = []
# n_features = []

# for p in range(3,14,2):
    # poly,norm = cp.orth_ttr(p, dist_x, retall=True)
    # N = 3*len(poly)
    # n_samples.append(N)
    # print(len(poly))
    # X = poly(*samp_x[:,:N]).T
    # y = Y[:N]
    # err_ = []
    # n_features_ = []

    # scaler = preprocessing.StandardScaler().fit(X)
    # X = scaler.transform(X)
    # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=0)

    # model_ols = linear_model.LinearRegression()
    # # scores = cross_validate(regressor, X, y,scoring=scoring,cv=5,return_train_score=False)
    # # test_mse_avg = abs(scores['test_neg_mean_squared_error']).mean()
    # # print('p ==%2d: %.2e' %(p, test_mse_avg/(np.std(y)**2)))
    # model_ols.fit(X_train,y_train)
    # y_pred = model_ols.predict(X_test)
    # err =  mse(y_pred, y_test)/np.std(y_test)**2
    # print('OLS MSE for p={:2d}:  {:.2e}'.format(p,err)) 
    # err_.append(err)
    # n_features_.append(len(poly))

   
    # # model_bic = linear_model.LassoLarsIC(criterion='bic')
    # # model_bic.fit(X_train, y_train)
    # # # alpha_bic_ = model_bic.alpha_
    # # y_pred = model_bic.predict(X_test)
    # # err =  mse(y_pred, y_test)/np.std(y_test)**2
    # # print('BIC MSE for p={:2d}:  {:.2e}'.format(p,err)) 

    # # model_aic = linear_model.LassoLarsIC(criterion='aic')
    # # model_aic.fit(X_train, y_train)
    # # # alpha_aic_ = model_aic.alpha_
    # # y_pred = model_aic.predict(X_test)
    # # err =  mse(y_pred, y_test)/np.std(y_test)**2
    # # print('AIC MSE for p={:2d}:  {:.2e}'.format(p,err)) 
    
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
    # # print(index)
    # # print(model_lars_cv.coef_path_.shape)
    # # print(model_lars_cv.alpha_)
    # # print(model_lars_cv.coef_)
    # # print(model_lars_cv.alphas_)
    # # print(model_lars_cv.mse_path_.shape)
    # # index = [3,5,7,1,2,14,15,17,19]
    # # X0_train = X_train[:,index]
    # # X0_test = X_test[:,index]

    # # model_ols_sparse = linear_model.LinearRegression()
    # # model_ols_sparse.fit(X0_train,y_train)
    # # y_pred = model_ols_sparse.predict(X0_test)
    # # err =  mse(y_pred, y_test)/np.std(y_test)
    # # print('sparse OLS MSE for p={:2d}:  {:.2e}'.format(p,err)) 

    # rel_err.append(err_)
    # n_features.append(n_features_)
    # print('-'*30)

    # # model_aic, model_bic, model_cv, model_lars_cv = lasso_model_selection(X,y)
    # # regressors['lasso_aic'] = linear_model.Lasso(alpha=model_aic.alpha_)
    # # regressors['lasso_cic'] = linear_model.Lasso(alpha=model_bic.alpha_)
    # # regressors['lasso_cv'] = linear_model.Lasso(alpha=model_cv.alpha_)
    # # regressors['lasso_lars_cv'] = linear_model.Lasso(alpha=model_lars_cv.alpha_)

    # # reg0.fit(X_train, y_train)
    # # pred0 = reg0.predict(X_test)
    # # R2.append(reg0.score(X_train, y_train))
    # # mse_test.append(mse(y_test,pred0))

    # # for key, regressor in regressors.items():
        # # scores = cross_validate(regressor, X, y,scoring=scoring,cv=5,return_train_score=False)
        # # test_mse_avg = abs(scores['test_neg_mean_squared_error']).mean()
        # # print('p ==%2d: %.2e' %(p, test_mse_avg/(np.std(y)**2)))
        # # mse_test[key].append(test_mse_avg)

# rel_err = np.array(rel_err)
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111)
# number_of_plots=10


# reg =linear_model.Lars()
# reg.fit(X,y)
# print reg.coef_path_.shape
# print 'poly:', poly
# print 'poly_evals', poly(*samp_x).shape
# f_hat = cp.fit_regression(poly,samp_x,y)

# print f_hat.coeffs()