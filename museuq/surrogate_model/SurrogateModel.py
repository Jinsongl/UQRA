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

CAL_COEFFS_METHODS = {
        'GQ'    : 'Gauss Quadrature',
        'QUAD'  : 'Gauss Quadrature',
        'SP'    : 'Gauss Quadrature',
        'RG'    : 'Linear Regression',
        }
# class dataClass(object):
    # """
    # data object
    # """
    # def __init__(self, x, y, w=None):
        # self.x = x
        # self.y = y
        # self.w = w


# Chaospy.Poly
# 1. Any constructed polynomial is a callable. The argument can either be inserted positional or as keyword arguments q0,
#   q1, . . . :
#   >>> poly = cp.Poly([1, x**2, x*y])
#   >>> print(poly(2, 3))
#   [1 4 6]
#   >>> print(poly(q1=3, q0=2))
#   [1 4 6]
# 2. The input can be a mix of scalars and arrays, as long as the shapes together can be joined to gether in a common
# compatible shape:
#   >>> print(poly(2, [1, 2, 3, 4]))
#   [[1 1 1 1]
#   [4 4 4 4]
#   [2 4 6 8]]
# 3. It is also possible to perform partial evaluation, i.e. evaluating some of the dimensions. To tell the polynomial that
# a dimension should not be evaluated either leave the argument empty or pass a masked value numpy.ma.masked.
#   For example:
#   >>> print(poly(2))
#   [1, 4, 2q1]
#   >>> print(poly(np.ma.masked, 2))
#   [1, q0^2, 2q0]
#   >>> print(poly(q1=2))
#   [1, q0^2, 2q0]
# 4. y = foo_hat(*sample) 
#   When foo_hat contains multiple surrogate models, it will apply each content of sample (*sample) to each of them and return a ndarray of shape(n,)
# Returns:
#   The type of return value for the polynomial is numpy.ndarray if all dimensions are filled. If not, it returns a new polynomial.

class SurrogateModel(object):
    """
    Meta model class object 
    General options:
    """
    def __init__(self, metamodel_class, basis_setting, **kwparams):
        """
        metamodel_class: string, surrogate model classes to be used
            - PCE
            - aPCE
            - Gaussian Process (GPR)

        basis_setting: list, used to generate the basis_setting functions for surrogate models
            PCE: list of orders, e.g. [8,9,10]. 
            GPR: list of kernel functions, e.g. [RBF + WN, RBF] 
            aPCE:


        dist_zeta: class, chaospy.distributions
            Distributions of underlying random variables from selected Wiener-Askey polynomial, mutually independent

        kwparams: dictionary containing parameters for specified metamodel class
            PCE: {
                dist_x:
                dist_zeta}
            GPR:{
                alpha         : 1e-10          
                optimizer     : 'fmin_l_bfgs_b'
                normalize_y   : False              
                copy_X_train  : True           
                random_state  : None           
                n_restarts_optimizer  : 0
                ...
                }

        basis_setting, orthpoly_norms: orthogonal polynomial basis_setting and their corresponding normalization constant. E[Phi_i, Phi_i] = nomrs[i]
        
        """
        self.metamodel_class = metamodel_class
        # define  a list of parameters to get metamodel basis functions. 
        # - PCE: list of basis function orders, redefine an attribute self.basis_orders
        # - GPR: list of kernels, redefine an attribute self.kernels  
        self.basis_setting   = []   
        self.kwparams        = kwparams if kwparams else {} 

        ## define list saving outputs
        self.model_basis     = []    # list of metamodel basis functions. For PCE, list of polynomials, for GPR, list of kernels
        self.metamodels      = []    # list of built metamodels
        self.metamodel_coeffs= []    # list of coefficient sets for each built meta model
        self.orthpoly_norms  = []    # Only for PCE model, list of norms for basis_setting functions 
        self.cv_l2errors     = []
        self.metric_names    = ['value','moments','norms','upper fractile','ECDF','pdf']
        self.metrics         = [[] for _ in range(len(self.metric_names))] ## each element is of shape: [sample sets, metaorders, repeated samples, metric]

        ## Initialize metaModel methods
        self.__metamodel_setting(basis_setting) 
        self.__get_metamodel_basis()    # list of basis_setting functions/kernels for meta models 


    def fit_model(self,x_train,y_train,weight=None):
        """
        Fit specified meta model with given observations (x_train,y_train, [weight for quadrature])

        Arguments:
            x_train: array-like of shape (ndim, nsamples) or (nsamples,) 
                sample input values in zeta (selected Wiener-Askey distribution) space
            y_train: array-like of shape (nsamples,[n_output_dims])
                QoI observations

            optional arguments:
            weight: array-like of shape(nsamples), 
                quadrature weights for PCE 
        """

        print(' ► Building surrogate models ...')
        if not self.basis_setting:
            self.__get_metamodel_basis() 

        x_train = np.squeeze(np.array(x_train))
        y_train = np.squeeze(np.array(y_train))
        if x_train.ndim == y_train.ndim == 1:
            assert len(x_train) == len(y_train), 'Number of data points in x and y is not same: {}!={}'.format(len(x_train), len(y_train))
        elif x_train.ndim == 1 and y_train.ndim != 1: 
            assert len(x_train) == y_train.shape[0],  'Number of data points in x and y is not same: {}!={} '.format(len(x_train), y_train.shape[0])
        elif x_train.ndim != 1 and y_train.ndim == 1:
            assert x_train.shape[1] == len(y_train),  'Number of data points in x and y is not same: {}!={} '.format(x_train.shape[1], len(y_train))
        elif x_train.ndim != 1 and y_train.ndim != 1:
            assert x_train.shape[1] == y_train.shape[0], 'Number of data points in x and y is not same: {}!={} '.format(x_train.shape[1], y_train.shape[0])
        else:
            raise ValueError

        if self.metamodel_class.upper() == 'PCE':
            weight = np.squeeze(weight)
            self.__build_pce_model(x_train,y_train,w=weight) 
        elif self.metamodel_class.upper() == "GPR":
            self.__build_gpr_model(x_train,y_train)

        elif self.metamodel_class.upper() == 'APCE':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def predict(self,X, return_std=False, return_cov=False):
        """
        Predict using surrogate models 

        Arguments:	
        X : array-like, shape = (n_features/ndim, nsamples)
            Query points where the surrogate model are evaluated

        Returns:	
        y : list of array, array shape = (nsamples, )
            predicted value from surrogate models at query points
        """
        if not self.metamodels:
            raise ValueError('No surrogate model exists')

        res_pred = []
        print(' ► Evaluating with surrogate models ... ')
        print('   ♦ {:<17s} : {}'.format('Query points ', X.shape))
        for i, imetamodel in enumerate(self.metamodels):
            print('   ♦ {:<17s} : {:d}/{:d} '.format('Surrogate model', i, len(self.metamodels)))
            if self.metamodel_class.upper() == 'PCE':
                ## See explainations about Poly above
                f_pred = imetamodel(*X)
            elif self.metamodel_class.upper() == 'GPR':
                f_mean, f_std = imetamodel.predict(X.T, return_std=return_std, return_cov=return_cov)
                f_pred = np.hstack((f_mean, f_std.reshape(f_mean.shape))).T
            res_pred.append(f_pred)
        return res_pred

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
    
    def sample_y(self,X, nsamples=1, random_state=0):
        """
        Draw samples from Gaussian process and evaluate at X.

        Parameters:	
        X : array-like, shape = (n_features,n_samples_X)
        Query points where the GP samples are evaluated

        nsamples : int, default: 1
        The number of samples drawn from the Gaussian process

        random_state : int, RandomState instance or None, optional (default=0)
        If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.

        Returns:	
        y_samples : list, list item shape = (n_samples_X, [n_output_dims], nsamples)
        Values of nsamples samples drawn from Gaussian process and evaluated at query points.

        """

        if self.metamodel_class.upper() != 'GPR':
            raise ValueError('log_marginal_likelihood funtion defined only for GPR model for now')
        if not self.metamodels:
            raise ValueError('No surrogate model exists')

        X = np.array(X)
        print(' ► Draw samples from Gaussian process models... ')
        print('   ♦ {:<15s} : {}'.format('nsamples', nsamples))
        print('   ♦ {:<15s} : {}'.format('random_state', random_state))
        y_samples = []
        for i, imetamodel in enumerate(self.metamodels):
            print('   ♦ Surrogate model {:d}/{:d} '.format(i, len(self.metamodels)))
            imeta_y_samples = imetamodel.sample_y(X.T, nsamples=nsamples, random_state=random_state)
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

    def __metamodel_setting(self, basis_setting):
        """
        Define parameters to build the metamodels
        """
        # __________________________________________________________________________________________________________________
        #           |                      Parameters
        # metamodel |-------------------------------------------------------------------------------------------------------
        #           |           required            |            optional 
        # ------------------------------------------------------------------------------------------------------------------
        # PCE       | dist_zeta, cal_coeffs         | dist_zeta_J, dist_x, dist_x_J, basis_orders
        # __________________________________________________________________________________________________________________
        # __________________________________________________________________________________________________________________
        # For PCE model, following parameters are required:
        print('------------------------------------------------------------')
        print('►►► Build Surrogate Models')
        print('------------------------------------------------------------')
        print(' ► Surrogate model properties:')
        print('   ♦ {:<17s} : {:<15s}'.format('Model class', self.metamodel_class))

        if self.metamodel_class.upper() == 'PCE':
            ## make sure basis_setting is a list
            self.basis_setting = []
            if np.isscalar(basis_setting):
                self.basis_setting.append(int(metamodel_orders))
            else:
                self.basis_setting= list(int(x) for x in basis_setting)
            # change basis_setting to basis_order for code reading easily
            self.basis_orders = self.basis_setting

            # Following parameters are required
            print('   ♦ Requried parameters:')
            try:
                print('     ∙ {:<15s} : {}'.format('Coeffs Meth'  , CAL_COEFFS_METHODS.get(self.kwparams['cal_coeffs'])))
                print('     ∙ {:<15s} : {}'.format('zeta dist'    , self.kwparams['dist_zeta']))
                print('     ∙ {:<15s} : {}'.format('Basis order'  , self.basis_orders))
            except KeyError:
                pass # Key is not present

            # Following parameters are optional
            self.kwparams['dist_zeta_J' ] = self.kwparams.get('dist_zeta_J'   ,self.kwparams['dist_zeta'])
            self.kwparams['dist_x'      ] = self.kwparams.get('dist_x'        , None)
            self.kwparams['dist_x_J'    ] = self.kwparams.get('dist_x_J'      , None)
            print('   ♦ Optional parameters:')
            for key, value in self.kwparams.items():
                if key in ['cal_coeffs', 'dist_zeta','Basis order']:
                    pass
                else:
                    print('     ∙ {:<15s} : {}'.format(key,value))

        elif self.metamodel_class.upper() == 'GPR':
            ## make sure basis_setting is a list
            if isinstance(basis_setting, list):
                self.basis_setting = basis_setting 
            else:
                self.basis_setting = [basis_setting,]
            # change basis_setting to kernels for code reading easily
            self.kernels = self.basis_setting

            # For GPR models, only list of kernel functions are required

            self.kwparams[ 'alpha'        ] = self.kwparams.get('alpha'       , 1e-10          )
            self.kwparams[ 'optimizer'    ] = self.kwparams.get('optimizer'   , 'fmin_l_bfgs_b')
            self.kwparams[ 'normalize_y'  ] = self.kwparams.get('normalize_y' , False          )    
            self.kwparams[ 'copy_X_train' ] = self.kwparams.get('copy_X_train', True           )
            self.kwparams[ 'random_state' ] = self.kwparams.get('random_state', None           )
            self.kwparams[ 'n_restarts_optimizer' ] = self.kwparams.get('n_restarts_optimizer', 0)

            print('   ♦ Parameters:')
            print('   ♦ {:<15s} : {}'.format('Kernels', self.kernels))
            print('   ♦ Optional parameters:')
            for key, value in self.kwparams.items():
                print('     ∙ {:<15s} : {}'.format(key,value))

        elif self.metamodel_class.upper() == 'APCE':
            raise NotImplementedError
        else:
            raise NotImplementedError

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

    def __build_gpr_model(self, x_train, y_train):

        # For GPR model:
        # Training data: array-like, shape = (nsamples, n_features) 
        # Target values: array-like, shape = (nsamples, [n_output_dims])
       
        # Reshape input data array
        x_train = x_train.T
        if x_train.ndim == 1:
            x_train = x_train[:, np.newaxis]
        y_train = y_train.reshape(x_train.shape[0],-1) 
        # print(x_train.shape)
        # print(y_train.shape)
        # if self.kwparams:

        alpha        = self.kwparams.get('alpha'       )
        optimizer    = self.kwparams.get('optimizer'   )
        normalize_y  = self.kwparams.get('normalize_y' )    
        copy_X_train = self.kwparams.get('copy_X_train')
        random_state = self.kwparams.get('random_state')
        n_restarts_optimizer = self.kwparams.get('n_restarts_optimizer')

        for ikernel in self.kernels:
            gp = GaussianProcessRegressor(kernel=ikernel,alpha=alpha, 
                    optimizer=optimizer, n_restarts_optimizer=n_restarts_optimizer, 
                    normalize_y=normalize_y, copy_X_train=copy_X_train,random_state=random_state)
            gp.fit(x_train, y_train)
            self.metamodels.append(gp)
            kernel_params = gp.kernel_.get_params()
            kernel_params['opt_theta'] = gp.kernel_.theta
            kernel_params['opt_theta_lml'] = gp.log_marginal_likelihood()
            self.metamodel_coeffs.append(kernel_params)

            # kernel_params = gp.get_params()
            print('   ♦ {:<15s} : {:d}'.format('Kernel (Initial)', ikernel))
            print('   ♦ Optimum values:')
            for key, value in kernel_params.items():
                print('     ∙ {:<25} : {}'.format(key,value))

    def __build_pce_model(self, x, y, w=None):
        """
        Build Polynomial chaos expansion surrogate model with x, y, optional w

        Arguments:
            x: sample input values in zeta (selected Wiener-Askey distribution) space
                array-like of shape(ndim, nsamples) or (nsamples,)
            y: array-like of shape(nsamples,), QoI observations
            w: array-like of shape(nsamples,), weight for quadrature 

        Returns: None
            Append surrogate models to self.metamodels 
            Append surrogate model params 
              (coeffs for orthogonal poly in PCE and kernel params for GPR) to self.metamodel_coeffs
        """
        # iorthpoly: ndim
        # x: (ndim, nsamples)
        # y: (nsamples, n_output)
        # len(f_hat): n_output

        cal_coeffs = self.kwparams.get('cal_coeffs')
        for i, iorthpoly in enumerate(self.model_basis):
            print('   ♦ {:<17s} : {:d}'.format('PCE model order', self.basis_orders[i]))
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

    def __get_metamodel_basis(self):
        """
        Return meta model basis functions
        For PCE: orthogonal basis functions for the specified underlying distributions (dist_zeta) with specifed orders (metamodel_orders)
        For GPR: list of kernels
        """
        if self.metamodel_class.upper() == 'PCE':
            for p_order in self.basis_orders:
                poly, norm = cp.orth_ttr(p_order, self.kwparams['dist_zeta_J'], retall=True)
                self.model_basis.append(poly)
                self.orthpoly_norms.append(norm)
        elif self.metamodel_class.upper() == 'GPR':
            self.kernels = self.basis_setting 



        

