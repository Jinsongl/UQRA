# vim:fenc=utf-8
#
# Copyright © 2017 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import chaospy as cp, numpy as np, sklearn.gaussian_process as skgp
import scipy.stats as scistats
import random
from scipy.stats.kde import gaussian_kde
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.gaussian_process import GaussianProcessRegressor
from ..utilities import metrics_collections 
from ..utilities.helpers import ordinal, blockPrint, enablePrint
import warnings
warnings.filterwarnings(action="ignore",  message="^internal gelsd")

# Chaospy.Poly
# 1. Any constructed polynomial is a callable. The argument can either be inserted positional or as keyword arguments q0,
#   q1, . . . :
#   >>> poly = cp.Poly([1, x**2, x*y])
#   >>> print(rpoly(2, 3))
#   [1 4 6]
#   >>> print(rpoly(q1=3, q0=2))
#   [1 4 6]
# 2. The input can be a mix of scalars and arrays, as long as the shapes together can be joined to gether in a common
# compatible shape:
#   >>> print(rpoly(2, [1, 2, 3, 4]))
#   [[1 1 1 1]
#   [4 4 4 4]
#   [2 4 6 8]]
# 3. It is also possible to perform partial evaluation, i.e. evaluating some of the dimensions. To tell the polynomial that
# a dimension should not be evaluated either leave the argument empty or pass a masked value numpy.ma.masked.
#   For example:
#   >>> print(rpoly(2))
#   [1, 4, 2q1]
#   >>> print(rpoly(np.ma.masked, 2))
#   [1, q0^2, 2q0]
#   >>> print(rpoly(q1=2))
#   [1, q0^2, 2q0]
# 4. y = foo_hat(*sample) 
#   When foo_hat contains multiple surrogate models, it will apply each content of sample (*sample) to each of them and return a ndarray of shape(n,)
# Returns:
#   The type of return value for the polynomial is numpy.ndarray if all dimensions are filled. If not, it returns a new polynomial.

CAL_COEFFS_METHODS = {
  'GALERKIN': 'Galerkin Projection',
    'GP'     : 'Galerkin Projection',
    'OLS'   : 'Ordinary Least Square ',
    }
class SurrogateModel(object):
    """
    Surrogate model class object 
    General options:
    """
    def __init__(self, name, setting, **kwparams):
        """
        name: string, surrogate model classes to be used
            - PCE
            - aPCE (to be implemented)
            - Gaussian Process (GPR)

        setting: list, used to generate the setting functions for surrogate models
            PCE: list of polynomial orders, e.g. [8,9,10]. 
            GPR: list of kernel functions, e.g. [RBF + WN, RBF] 
            aPCE:

        dist_zeta: class, chaospy.distributions
            Distributions of underlying random variables from selected Wiener-Askey polynomial, assumed IID now 

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

        setting, orthpoly_norms: orthogonal polynomial setting and their corresponding normalization constant. E[Phi_i, Phi_i] = nomrs[i]
        
        """
        # define  a list of parameters to get metamodel basis functions. 
        # - PCE: list of basis function orders, redefine an attribute self.basis_orders
        # - GPR: list of kernels, redefine an attribute self.kernels  
        self.name            = name  # string, PCE, aPCE, GPR
        self.setting         = []   # list, used to generate the setting functions for surrogate models
        self.kwparams        = kwparams if kwparams else {} 
        ## define list saving outputs
        self.basis           = []    # list of metamodel basis functions. For PCE, list of polynomials, for GPR, list of kernels
        self.metamodels      = []    # list of built metamodels
        self.basis_coeffs    = []    # list of coefficient sets for each built meta model
        self.poly_coeffs     = []    # list of coefficient sets for each built meta model
        self.orthpoly_norms  = []    # Only for PCE model, list of norms for setting functions 
        self.metrics     = ['mean_squared_error']
        self.metrics_value   = [] ## each element is of shape: [sample sets, metaorders, repeated samples, metric]
        ## Initialize metaModel methods
        self.__metamodel_setting(setting) 
        self.__get_metamodel_basis()    # list of setting functions/kernels for meta models 

    def fit(self,x_train,y_train,weight=None):
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

        print(r' > Building surrogate models ...')
        if not self.setting:
            self.__get_metamodel_basis() 

        x_shape = x_train[0].shape if isinstance(x_train, list) else x_train.shape
        y_shape = y_train[0].shape if isinstance(y_train, list) else y_train.shape
        if weight is not None:
            w_shape = weight[0].shape if isinstance(weight, list) else weight.shape
            print(r'   * {:<17s} : (X, Y, W) = {} x {} x {}'.format('Train data shape', x_shape, y_shape, w_shape))
        else:
            print(r'   * {:<17s} : (X, Y) = {} x {} '.format('Train data shape', x_shape, y_shape))
        if self.name.upper() == 'PCE':
            weight = np.squeeze(weight)
            self.__build_pce_model(x_train,y_train,w=weight) 
        elif self.name.upper() == "GPR":
            self.__build_gpr_model(x_train,y_train)
        elif self.name.upper() == 'MPCE':
            self.__build_mpce_model(x_train,y_train,w=weight)
        elif self.name.upper() == 'APCE':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def predict(self,X, y_true=None, **kwargs):
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
            raise ValueError('No surrogate models exist')
        # dist_zeta = self.kwparams['dist_zeta']
        # if len(dist_zeta) != X.shape[0]:
            # raise ValueError('Model expecting {} random variables but {} was given'.format(len(dist_zeta), X.shape[0]))
        

        print(r' > Evaluating with surrogate models ... ')

        surrogates_pred = []
        if self.name.upper() == 'GPR':
            print(r'   * {:<17s} : {}'.format('Query points ', X.shape))
            return_std = kwargs.get('return_std', False)
            return_cov = kwargs.get('return_cov', False)
            for i, imetamodel in enumerate(self.metamodels):
                y_pred_ = imetamodel.predict(X.T, return_std=return_std, return_cov=return_cov)
                # if return_cov:
                    # y_mean, y_cov = y_pred_ 
                # elif return_std:
                    # y_mean, y_std = y_pred_
                # else:
                    # y_mean = y_pred_
                print(r'   * {:<17s} : {:d}/{:d}    -> Output: {}'.format('Surrogate model (GPR)', i, len(self.metamodels), y_pred_.shape))
                y_pred = np.squeeze(np.array(y_pred_))
            surrogates_pred.append(y_pred)

        elif self.name.upper() == 'PCE':
            print(r'   * {:<17s} : {}'.format('Query points ', X.shape))
            ## See explainations about Poly above
            for i, imetamodel in enumerate(self.metamodels):
                y_pred = imetamodel(*X)
                print(r'   * {:<17s} : {:d}/{:d}    -> Output: {}'.format('Surrogate model (PCE)', i, len(self.metamodels), y_pred.shape))
                surrogates_pred.append(y_pred)

        elif self.name.upper() == 'MPCE':
            print(r'   * {:<17s} : {}'.format('Query points ', X.shape))
            metamodel_candidate = [] 
            for i, imetamodel in enumerate(self.metamodels):
                y_pred = imetamodel(*X)
                metamodel_candidate.append(y_pred)
            metamodel_candidate = np.array(metamodel_candidate)
            idx = np.random.randint(0, metamodel_candidate.shape[0],size=metamodel_candidate.shape[1])
            y_pred = np.choose(idx, metamodel_candidate)
            print(r'   * {:<17s} : m = {:d}    -> Output: {}'.format('Surrogate model (mPCE)', metamodel_candidate.shape[0], y_pred.shape))
            surrogates_pred.append(y_pred)

        if y_true is not None:
            scores = self.score(surrogates_pred, y_true, **kwargs)
            if len(surrogates_pred) == 1:
                surrogates_pred = surrogates_pred[0]
            return surrogates_pred, scores
        else:
            if len(surrogates_pred) == 1:
                surrogates_pred = surrogates_pred[0]
            return surrogates_pred

    
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

        if self.name.upper() != 'GPR':
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


        print(r' > Evaluating log marginal likelihood with surrogate models ... ')
        print(r'   * {:<15s} : {}'.format('Kernel hyperparameters ', theta_shape))
        print(r'   * {:<15s} : {}'.format('Evaluate gradient', eval_gradient))
        for i, imetamodel in enumerate(self.metamodels):
            print(r'   * Surrogate model {:d}/{:d} '.format(i, len(self.metamodels)))
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

        if self.name.upper() != 'GPR':
            raise ValueError('log_marginal_likelihood funtion defined only for GPR model for now')
        if not self.metamodels:
            raise ValueError('No surrogate model exists')

        X = np.array(X)
        print(r' > Draw samples from Gaussian process models... ')
        print(r'   * {:<15s} : {}'.format('nsamples', nsamples))
        print(r'   * {:<15s} : {}'.format('random_state', random_state))
        y_samples = []
        for i, imetamodel in enumerate(self.metamodels):
            print(r'   * Surrogate model {:d}/{:d} '.format(i, len(self.metamodels)))
            imeta_y_samples = imetamodel.sample_y(X.T, nsamples=nsamples, random_state=random_state)
            n_output_dims = imeta_y_samples.shape[1]
            if n_output_dims == 1:
                imeta_y_samples = np.squeeze(imeta_y_samples).T
                # print(rX.shape)
                # print(rimeta_y_samples.shape)
                y_samples.append(np.vstack((X, imeta_y_samples)))
                # print(ry_samples[-1].shape)
            else:
                X = X[np.newaxis,:]
                y_samples.append(np.vstack((X, imeta_y_samples.T)))
        if len(y_samples) == 1:
            y_samples = y_samples[0]
        return y_samples

    def __metamodel_setting(self, setting):
        """
        Define parameters to build the metamodels
        """
        # __________________________________________________________________________________________________________________
        #           |                           Parameters
        # metamodel |-------------------------------------------------------------------------------------------------------
        #           |           required            |            optional 
        # ------------------------------------------------------------------------------------------------------------------
        # PCE       | dist_zeta, cal_coeffs         | dist_zeta_J, dist_x, dist_x_J, basis_orders
        # __________________________________________________________________________________________________________________
        # __________________________________________________________________________________________________________________
        # For PCE model, following parameters are required:
        print(r'------------------------------------------------------------')
        print(r'>>> Initialize SurrogateModel Object...')
        print(r'------------------------------------------------------------')
        print(r' > Surrogate model properties:')
        print(r'   * {:<17s} : {:<15s}'.format('Model name', self.name))

        if 'PCE' in self.name.upper():
            ## make sure setting is a list
            self.setting = []
            if np.isscalar(setting):
                self.setting.append(int(setting))
            else:
                self.setting= list(int(x) for x in setting)
            # change setting to basis_order for code reading easily
            self.basis_orders = self.setting

            # Following parameters are required
            print(r'   * Requried parameters:')
            try:
                print(r'     - {:<15s} : {}'.format('Solve coeffs:', CAL_COEFFS_METHODS.get(self.kwparams['cal_coeffs'])))
                print(r'     - {:<15s} : {}'.format('Zeta Joint dist'    , self.kwparams['dist_zeta_J']))
                print(r'     - {:<15s} : {}'.format('Basis order'  , self.basis_orders))
            except KeyError:
                print(r'     - cal_coeffs, dist_zeta, basis_orders are required parameters for PCE model...')

            # Following parameters are optional
            self.kwparams['dist_zeta_M' ] = self.kwparams.get('dist_zeta_M'   , None)
            self.kwparams['dist_x'      ] = self.kwparams.get('dist_x'        , None)
            self.kwparams['dist_x_J'    ] = self.kwparams.get('dist_x_J'      , None)
            print(r'   * Optional parameters:')
            for key, value in self.kwparams.items():
                if key in ['cal_coeffs', 'dist_zeta','Basis order']:
                    pass
                else:
                    print(r'     - {:<15s} : {}'.format(key,value))

        elif self.name.upper() == 'GPR' :
            ## make sure setting is a list
            if isinstance(setting, list):
                self.setting = setting 
            else:
                self.setting = [setting,]
            # change setting to kernels for code reading easily
            self.kernels = self.setting

            # For GPR models, only list of kernel functions are required

            self.kwparams[ 'alpha'        ] = self.kwparams.get('alpha'       , 1e-10          )
            self.kwparams[ 'optimizer'    ] = self.kwparams.get('optimizer'   , 'fmin_l_bfgs_b')
            self.kwparams[ 'normalize_y'  ] = self.kwparams.get('normalize_y' , False          )    
            self.kwparams[ 'copy_X_train' ] = self.kwparams.get('copy_X_train', True           )
            self.kwparams[ 'random_state' ] = self.kwparams.get('random_state', None           )
            self.kwparams[ 'n_restarts_optimizer' ] = self.kwparams.get('n_restarts_optimizer', 0)

            print(r'   * Parameters:')
            print(r'   * {:<15s} : {}'.format('Kernels', self.kernels))
            print(r'   * Optional parameters:')
            for key, value in self.kwparams.items():
                print(r'     - {:<15s} : {}'.format(key,value))

        elif self.name.upper() == 'APCE':
            raise NotImplementedError
        else:
            raise NotImplementedError


    def __build_gpr_model(self, x_train, y_train):

        # For GPR model:
        # Training data: array-like, shape = (nsamples, n_features) 
        # Target values: array-like, shape = (nsamples, [n_output_dims])
       
        # Reshape input data array

        x_train = x_train.T
        if x_train.ndim == 1:
            x_train = x_train[:, np.newaxis]
        y_train = y_train.reshape(x_train.shape[0],-1) 
        # print(rx_train.shape)
        # print(ry_train.shape)
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
            self.basis_coeffs.append(kernel_params)

            # kernel_params = gp.get_params()
            print(r'   * {:<15s} : {}'.format('Kernel (Initial)', ikernel))
            print(r'   * Optimum values:')
            for key, value in kernel_params.items():
                print(r'     - {:<25} : {}'.format(key,value))

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
              (coeffs for orthogonal poly in PCE and kernel params for GPR) to self.basis_coeffs
        """
        # iorthpoly: ndim
        # x: (ndim, nsamples)
        # y: (nsamples, n_output)
        # len(f_hat): n_output

        if x.shape[-1] != y.shape[0]:
            raise ValueError("x.T and y must have same first dimension, but have shapes {} and {}".format(x.T.shape, y.shape))

        cal_coeffs = self.kwparams.get('cal_coeffs')
        print(r'   * {:<17s} : '.format('PCE model order'), end='')
        for i, iorthpoly in enumerate(self.basis):
            print(r' {:d}'.format(self.basis_orders[i]), end='')
            if cal_coeffs.upper() in ['GP','GALERKIN']:
                if w is None:
                    raise ValueError("Quadrature weights are needed for Galerkin method")
                if iorthpoly.dim != x.shape[0]:
                    raise ValueError("Polynomial base functions and variables must have same dimensions, but have Poly.ndim={} and x.ndim={}".format(iorthpoly.dim, x.shape[0]))
                # f_hat, orthpoly_coeffs = cp.fit_quadrature(iorthpoly, x, w, y, norms=self.orthpoly_norms[i], retall=True)
                f_hat, orthpoly_coeffs = cp.fit_quadrature(iorthpoly, x, w, y, retall=True)
                self.metamodels.append(f_hat)
                self.basis_coeffs.append(orthpoly_coeffs)
                self.poly_coeffs.append(f_hat.coefficients)
            elif cal_coeffs.upper() == 'OLS':
                f_hat, orthpoly_coeffs= cp.fit_regression(iorthpoly, x, y, retall=True)
                self.metamodels.append(f_hat)
                self.basis_coeffs.append(orthpoly_coeffs)
                self.poly_coeffs.append(f_hat.coefficients)
            else:
                raise ValueError('Method to calculate PCE coefficients {:s} is not defined'.format(cal_coeffs))
        print('\n')

    def __build_apce_model(self, x, y):
        print(r'\tBuilding aPCE surrogate model')
        metamodels = []
        l2_ers = []

        return (metamodels, l2_ers)

    def __build_mpce_model(self, x, y, w):
        """
        build multiple pce models for each pair of zip(x,y) 
        """
        print(r'   * {:<17s} : {}'.format('PCE model order', self.basis_orders))

        blockPrint()
        if isinstance(x, (np.ndarray, np.generic)):
            self.__build_pce_model(x, y, w)
        else:
            for ix, iy, iw in zip(x, y, w):
                self.__build_pce_model(ix, iy, iw)
        enablePrint()

    def __get_metamodel_basis(self):
        """
        Return meta model basis functions
        For PCE: orthogonal basis functions for the specified underlying distributions (dist_zeta) with specifed orders (setting)
        For GPR: list of kernels
        """
        if 'PCE' in self.name.upper():
            for p_order in self.basis_orders:
                poly, norm = cp.orth_ttr(p_order, self.kwparams['dist_zeta_J'], retall=True)
                self.basis.append(poly)
                self.orthpoly_norms.append(norm)
        elif 'GPR' in self.name.upper():
            self.kernels = self.setting 



        

