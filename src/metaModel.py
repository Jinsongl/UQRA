# vim:fenc=utf-8
#
# Copyright © 2017 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import chaospy as cp
import numpy as np
import sklearn.gaussian_process as skgp
import scipy.stats as scistats
from sklearn.utils import check_random_state
from scipy.stats.kde import gaussian_kde
from statsmodels.distributions.empirical_distribution import ECDF
import warnings
warnings.filterwarnings(action="ignore",  message="^internal gelsd")
from sklearn.gaussian_process import GaussianProcessRegressor
cal_coeffs_methods = {
        'GQ'    : 'Gauss Quadrature',
        'QUAD'  : 'Gauss Quadrature',
        'SP'    : 'Gauss Quadrature',
        'RG'    : 'Linear Regression',
        }
class metaModel(object):
    """
    Meta model class object 
    General options:
    """
    def __init__(self, metamodel_class, metamodel_basis, dist_zeta_list, **kwargs):
        """
        metamodel_class: 
            surrogate model classes to be used, e.g. PCE, aPCE or Gaussian Process (GPR) etc 
        metamodel_basis:

        dist_zeta_list: list, marginal distributions of underlying random variables from selected Wiener-Askey polynomial, mutually independent
        metamodel_basis, orthpoly_norms: orthogonal polynomial basis and their corresponding normalization constant. E[Phi_i, Phi_i] = nomrs[i]
        
        """
        self.metamodel_class = metamodel_class
        # self.cal_coeffs      = cal_coeffs
        self.dist_zeta_list  = dist_zeta_list
        self.distJ_zeta      = dist_zeta_list if len(dist_zeta_list) == 1 else cp.J(*dist_zeta_list)
        self.metamodel_params= kwargs if kwargs else {} 
        self.metamodels      = []    # list of built metamodels
        self.metamodel_coeffs= []    # list of parameter sets for each built meta model
        self.orthpoly_norms  = []    # Only for PCE model, list of norms for basis functions 
        self.cv_l2errors     = []
        self.metric_names    = ['value','moments','norms','upper fractile','ECDF','pdf']
        self.metrics         = [[] for _ in range(len(self.metric_names))] ## each element is of shape: [sample sets, metaorders, repeated samples, metric]
        self.__get_metamodel_basis(metamodel_basis)    # list of basis functions/kernels for meta models 

        print('------------------------------------------------------------')
        print('►►► Build Surrogate Models')
        print('------------------------------------------------------------')
        print(' ► Surrogate model properties:')
        print('   ♦ {:<15s} : {:<15s}'.format('Model class', self.metamodel_class))
        if self.metamodel_class.upper() == 'PCE':
            cal_coeffs = cal_coeffs_methods.get(self.metamodel_params.get('cal_coeffs', 'NA')) 
            print('   ♦ {:<15s} : {:<15s}'.format('Coeffs Meth', cal_coeffs ))
            print('   ♦ {:<15s} : {}'.format('Basis order', metamodel_basis))
        elif self.metamodel_class.upper() == 'GPR':
            print('   ♦ {:<15s} : {:<15s}'.format('Optimizer', self.metamodel_params.get('optimizer', 'fmin_l_bfgs_b')))
            for i, ikernel in enumerate(self.metamodel_basis):
                # ikernel_params = ikernel.get_params()
                # ikernel_name   = ikernel_params['']
                if i==0:
                    print('   ♦ {:<15s} : {}'.format('Kernels', ikernel))
                else:
                    print('   ♦ {:<15s} : {}'.format('', ikernel['kernel']))
        elif self.metamodel_class.upper() == 'APCE':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def fit_model(self,training_data,target_values,weight=None):
        """
        Fit specified meta model with given observations (training_data,target_values, weight for quadrature)

        Arguments:
            training_data: sample input values in zeta (selected Wiener-Askey distribution) space
                array-like, shape= (ndim, nsamples)
            target_values: QoI observations, array-like, shape = (nsamples,)
            weight: 
                PCE: array-like of shape(nsamples), weight for quadrature 
                GPR: list of kernels 
        """
        print(' ► Building surrogate models ...')

        
        if not self.metamodel_basis:
            self.__get_metamodel_basis() 

        training_data = np.array(training_data)
        target_values = np.array(target_values)

        if self.metamodel_class.upper() == 'PCE':
            target_values = np.squeeze(target_values)
            weight = np.squeeze(weight)
            assert training_data.shape[1] == len(target_values),\
                    "PCE training data incompatible dimensions, training_data shape: {}, target_values shape{}".format(training_data.shape, target_values.shape)
            self.__build_pce_model(training_data,target_values,w=weight,) 
        elif self.metamodel_class.upper() == "GPR":
            " For GPR model:  \
            Training data: array-like, shape = (n_samples, n_features) \
            Target values: array-like, shape = (n_samples, [n_output_dims]) \
            "
            training_data = training_data.T
            target_values = target_values.reshape(training_data.shape[0],-1) 
            # print(training_data.shape)
            # print(target_values.shape)
            # if self.metamodel_params:
            print('   ♦ GPR model parameters:')
            for key, value in self.metamodel_params.items():
                print('     ∙ {:<25s} : {}'.format(key,value))

            alpha        = self.metamodel_params.get('alpha'       , 1e-10          )
            optimizer    = self.metamodel_params.get('optimizer'   , 'fmin_l_bfgs_b')
            normalize_y  = self.metamodel_params.get('normalize_y' , False          )    
            copy_X_train = self.metamodel_params.get('copy_X_train', True           )
            random_state = self.metamodel_params.get('random_state', None           )
            n_restarts_optimizer = self.metamodel_params.get('n_restarts_optimizer', 0)

            for kernel in self.metamodel_basis:
                gp = GaussianProcessRegressor(kernel=kernel,alpha=alpha, 
                        optimizer=optimizer, n_restarts_optimizer=n_restarts_optimizer, 
                        normalize_y=normalize_y, copy_X_train=copy_X_train,random_state=random_state)
                gp.fit(training_data, target_values)
                self.metamodels.append(gp)
                kernel_params = gp.kernel_.get_params()
                kernel_params['opt_theta'] = gp.kernel_.theta
                kernel_params['opt_theta_lml'] = gp.log_marginal_likelihood()
                self.metamodel_coeffs.append(kernel_params)

                # kernel_params = gp.get_params()
                for key, value in kernel_params.items():
                    print('     ∙ {:<25} : {}'.format(key,value))

        elif self.metamodel_class.upper() == 'APCE':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def predict(self,X, return_std=False, return_cov=False):
        """
        Predict using surrogate models 

        Arguments:	
        X : array-like, shape = (n_features/ndim, n_samples)
            Query points where the surrogate model are evaluated

        Returns:	
        y : list of array, array shape = (n_samples, )
            predicted value from surrogate models at query points
        """
        if not self.metamodels:
            raise ValueError('No surrogate model exists')

        predres = []
        print(' ► Evaluating with surrogate models ... ')
        print('   ♦ {:<15s} : {}'.format('Query points ', X.shape))
        for i, imetamodel in enumerate(self.metamodels):
            print('   ♦ Surrogate model {:d}/{:d} '.format(i, len(self.metamodels)))
            if self.metamodel_class.upper() == 'PCE':
                f_pred = np.array(imetamodel(X))
            elif self.metamodel_class.upper() == 'GPR':
                f_mean, f_std = imetamodel.predict(X.T, return_std=return_std, return_cov=return_cov)
                f_pred = np.hstack((f_mean, f_std.reshape(f_mean.shape))).T
            predres.append(f_pred)
        return np.squeeze(np.array(predres))

    def score(self,x,p=[0.01,0.001],retmetrics=[1,1,1,1,1,1]):
        """
        Calculate a set of error metrics used to evaluate the accuracy of the approximation
        Reference: "On the accuracy of the polynomial chaos approximation R.V. Field Jr., M. Grigoriu"
        arguments: 
            x: estimate data of shape(n,)
            y: real data of shape(n,)
        [values, moments, norms, upper tails, ecdf, pdf] 
        -----------------------------------------------------------------------
        0 | values 
        1 | moments, e = [1 - pce_moment(i)/real_moment(i)], for i in range(n)
        2 | norms : numpy.linalg.norm(x, ord=None, axis=None, keepdims=False)
        3 | upper fractile 
        4 | ecdf 
        5 | kernel density estimator pdf
        """
        metrics = []
        self.metric_names=[]
        x = np.squeeze(np.array(x))
        if retmetrics[0]:
            metrics.append(x)
            self.metric_names.append['value']
        if retmetrics[1]:
            metrics.append(self.__error_moms(x))
            self.metric_names.append('moments')
        if retmetrics[2]:
            metrics.append(self.__error_norms(x))
            self.metric_names.append('norms')
        if retmetrics[3]:
            metrics.append(self.__error_tail(x,p=p))
            self.metric_names.append('upper_tail')
        if retmetrics[4]:
            metrics.append(ECDF(x))
            self.metric_names.append('ECDF')
        if retmetrics[5]:
            metrics.append(gaussian_kde(x))
            self.metric_names.append('kde')
        # else:
            # NotImplementedError('Metric not implemented')
        return metrics
    
    def log_marginal_likelihood(self, theta, eval_gradient=False):
        """
        Returns log-marginal likelihood of theta for training data.

        Parameters:	
        theta : list of array-like, [1-D arrays representing the coordinates of a grid]
        Kernel hyperparameters for which the log-marginal likelihood is evaluated. 
        If None, the precomputed log_marginal_likelihood of self.kernel_.theta is returned.

        eval_gradient : bool, default: False
        If True, the gradient of the log-marginal likelihood with respect to the kernel hyperparameters at position theta is returned additionally. If True, theta must not be None.

        Returns:	
        log_likelihood : float
        Log-marginal likelihood of theta for training data.

        log_likelihood_gradient : array, shape = (n_kernel_params,), optional
        Gradient of the log-marginal likelihood with respect to the kernel hyperparameters at position theta. Only returned when eval_gradient is True.

        """

        if self.metamodel_class.upper() != 'GPR':
            raise ValueError('log_marginal_likelihood funtion defined only for GPR model for now')
        if not self.metamodels:
            raise ValueError('No surrogate model exists')

        log_marginal_likelihood_theta = []
        theta_meshgrid = np.meshgrid(*theta, indexing='ij')
        theta_flatten  = []
        for theta_i in theta_meshgrid:
            theta_flatten.append(theta_i.flatten())
        theta_flatten = np.array(theta_flatten).T
        theta_shape = theta_meshgrid[0].shape 


        print(' ► Evaluating log marginal likelihood with surrogate models ... ')
        print('   ♦ {:<15s} : {}'.format('Kernel hyperparameters ', theta_shape))
        print('   ♦ {:<15s} : {}'.format('Evaluate gradient', eval_gradient))
        for i, imetamodel in enumerate(self.metamodels):
            print('   ♦ Surrogate model {:d}/{:d} '.format(i, len(self.metamodels)))
            imetamodel_log_marginal_likelihood = []
            for itheta in theta_flatten:
                imetamodel_log_marginal_likelihood.append(imetamodel.log_marginal_likelihood(theta=np.log(itheta), eval_gradient=eval_gradient))
            theta_meshgrid.append(np.reshape(imetamodel_log_marginal_likelihood, theta_shape))

        return theta_meshgrid
    
    def sample_y(self,X, n_samples=1, random_state=0):
        """
        Draw samples from Gaussian process and evaluate at X.

        Parameters:	
        X : array-like, shape = (n_features,n_samples_X)
        Query points where the GP samples are evaluated

        n_samples : int, default: 1
        The number of samples drawn from the Gaussian process

        random_state : int, RandomState instance or None, optional (default=0)
        If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.

        Returns:	
        y_samples : list, list item shape = (n_samples_X, [n_output_dims], n_samples)
        Values of n_samples samples drawn from Gaussian process and evaluated at query points.

        """

        if self.metamodel_class.upper() != 'GPR':
            raise ValueError('log_marginal_likelihood funtion defined only for GPR model for now')
        if not self.metamodels:
            raise ValueError('No surrogate model exists')

        X = np.array(X)
        print(' ► Draw samples from Gaussian process models... ')
        print('   ♦ {:<15s} : {}'.format('n_samples', n_samples))
        print('   ♦ {:<15s} : {}'.format('random_state', random_state))
        y_samples = []
        for i, imetamodel in enumerate(self.metamodels):
            print('   ♦ Surrogate model {:d}/{:d} '.format(i, len(self.metamodels)))
            imeta_y_samples = imetamodel.sample_y(X.T, n_samples=n_samples, random_state=random_state)
            n_output_dims = imeta_y_samples.shape[1]
            if n_output_dims == 1:
                imeta_y_samples = np.squeeze(imeta_y_samples).T
                # print(X.shape)
                # print(imeta_y_samples.shape)
                y_samples.append(np.vstack((X, imeta_y_samples)))
                # print(y_samples[-1].shape)
            else:
                X = X[np.newaxis,:]
                y_samples.append(np.vstack((X, imeta_y_samples.T)))
        if len(y_samples) == 1:
            y_samples = y_samples[0]
        return y_samples


    def cross_validate(self, x, y):
        """
        Cross validation of the fitted metaModel with given data
        """
        if not self.metamodels:
            raise ValueError('No meta model exists')
        for _, metamodels in enumerate(self.metamodels):
            _cv_l2error = []
            print('\t\tCross validation {:s} metamodel of order \
                    {:d}'.format(self.metamodel_class, max([sum(order) for order in f.keys])))
            for _, f in enumerate(metamodels):
                f_fit = [f(*val) for val in x.T]
                f_cv.append(f_fit)
                cv_l2error = np.sqrt(np.mean((np.asfarray(f_fit) - y)**2),axis=0)
                _cv_l2error.append(cv_l2error)
            self.cv_l2errors.append(_cv_l2error)
# sampling=[1e5,10,'R'], retmetrics=[1,1,1,1,1,1]

    def __error_moms(self,x):
        x = np.squeeze(x)
        assert x.ndim == 1
        xx = np.array([x**i for i in np.arange(7)])
        xx_moms = np.mean(xx, axis=1)
        return xx_moms

    def __error_norms(self, x):
        """
        ord	norm for matrices	        norm for vectors
        None	Frobenius norm	                2-norm
        ‘fro’	Frobenius norm	                –
        ‘nuc’	nuclear norm	                –
        inf	max(sum(abs(x), axis=1))	max(abs(x))
        -inf	min(sum(abs(x), axis=1))	min(abs(x))
        0	–	                        sum(x != 0)
        1	max(sum(abs(x), axis=0))	as below
        -1	min(sum(abs(x), axis=0))	as below
        2	2-norm (largest sing. value)	as below
        -2	smallest singular value	i       as below
        other	–	                        sum(abs(x)**ord)**(1./ord)
        """
        x = np.squeeze(x)
        # y = np.squeeze(y)
        assert x.ndim == 1
        e = []

        e.append(np.linalg.norm(x,ord=0))
        e.append(np.linalg.norm(x,ord=1))
        e.append(np.linalg.norm(x,ord=2))
        e.append(np.linalg.norm(x,ord=np.inf))
        e.append(np.linalg.norm(x,ord=-np.inf))
        return e

    def __error_tail(self,x, p=[0.01,]):
        """
        """
        e = []
        for ip in p:
            p_invx = scistats.mstats.mquantiles(x,1-ip)
            # p_invy = scistats.mstats.mquantiles(y,1-ip)
            e.append(p_invx[0])
        return e 

    def __build_pce_model(self, x, y, w=None):
        """
        Build Polynomial chaos expansion surrogate model with x, y, optional w

        Arguments:
            x: sample input values in zeta (selected Wiener-Askey distribution) space
                array-like of shape(ndim, nsamples)
            y: array-like of shape(nsamples,), QoI observations
            w: array-like of shape(nsamples,), weight for quadrature 

        Returns: None
            Append surrogate models to self.metamodels 
            Append surrogate model params 
              (coeffs for orthogonal poly in PCE and kernel params for GPR) to self.metamodel_coeffs
        """
        # iorthpoly: ndim
        # x: (ndim, n_samples)
        # y: (n_samples, n_output)
        # len(f_hat): n_output

        cal_coeffs = self.metamodel_params.get('cal_coeffs')
        for i, iorthpoly in enumerate(self.metamodel_basis):
            print('   ♦ {:<15s} : {:d}'.format('PCE model order', self.metamodel_pce_orders[i]))
            if cal_coeffs.upper() in ['SP','GQ','QUAD']:
                assert w is not None
                assert iorthpoly.dim == x.shape[0]
                f_hat, orthpoly_coeffs = cp.fit_quadrature(iorthpoly, x, w, y, retall=True)
                self.metamodels.append(f_hat)
                self.metamodel_coeffs.append(orthpoly_coeffs)
            elif cal_coeffs.upper() == 'RG':
                f_hat, orthpoly_coeffs= cp.fit_regression(iorthpoly, x, y, retall=True)
                self.metamodels.append(f_hat)
                self.metamodel_coeffs.append(orthpoly_coeffs)
            else:
                raise ValueError('Method to calculate PCE coefficients {:s} is not defined'.format(cal_coeffs))

    def __build_apce_model(self, x, y):
        print('\tBuilding aPCE surrogate model')
        metamodels = []
        l2_ers = []



        return (metamodels, l2_ers)

    def __get_metamodel_basis(self, metamodel_basis):
        """
        Return meta model basis functions
        For PCE: orthogonal basis functions for the specified underlying distributions (dist_zeta_list) with specifed orders (metamodel_orders)
        For GPR: list of kernels
        """
        if self.metamodel_class.upper() == 'PCE':
            self.metamodel_basis = []
            metamodel_orders     = []
            if np.isscalar(metamodel_basis):
                metamodel_orders.append(int(metamodel_orders))
            else:
                metamodel_orders = list(int(x) for x in metamodel_basis)
            self.metamodel_pce_orders = metamodel_orders
            for p_order in metamodel_orders:
                poly, norm = cp.orth_ttr(p_order, self.distJ_zeta, retall=True)
                self.metamodel_basis.append(poly)
                self.orthpoly_norms.append(norm)
        elif self.metamodel_class.upper() == 'GPR':
            self.metamodel_basis = metamodel_basis 



        
