# vim:fenc=utf-8
#
# Copyright © 2017 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import chaospy as cp, numpy as np, sklearn.gaussian_process as skgp
import warnings
from museuq.utilities import metrics_collections 
from museuq.utilities.helpers import ordinal
warnings.filterwarnings(action="ignore",  message="^internal gelsd")


class SurrogateModel(object):
    """
    Abstract class for Surrogate model 
    """
    def __init__(self, random_seed = None):
        self.random_seed = random_seed
        self.name        = ''   # string, PCE, aPCE, GPR
        self.setting     = []   # list, used to generate the setting functions for surrogate models
        self.basis       = []   # list of metamodel basis functions. For PCE, list of polynomials, for GPR, list of kernels

    def fit(self,x,y,w=None):
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
        raise NotImplementedError

    def score(self,y_pred,y_true,**kwargs):
        """
        Calculate error metrics_value used to evaluate the accuracy of the approximation
            Reference: "On the accuracy of the polynomial chaos approximation R.V. Field Jr., M. Grigoriu"
        Parameters:	
            y_pred: list of predicted value set 
            y_true: array-like data from true model to be compared against.
            ps: Shape for each element in y_pred must be same as the shape of y_true

        kwargs:
            metrics: list of str: metric names to be calculated. Including:
                'explained_variance_score'
                'mean_absolute_error'
                'mean_squared_error'
                'mean_squared_log_error'
                'median_absolute_error'
                'r2_score'
                'moments'
                'upper_tail'
            prob: uppper tail probabilities to be calculated
            moment: int or array_like of ints, optional



        [values, moments, norms, upper tails, ecdf, pdf, R2] 
        -----------------------------------------------------------------------
        Returns the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares ((y_true - y_pred) ** 2).sum() and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum(). The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.

        Returns:	
        score : float
        R^2 of self.predict(X) wrt. y.
        0 | values 
        1 | moments, e = [1 - pce_moment(i)/real_moment(i)], for i in range(n)
        2 | norms : numpy.linalg.norm(x, ord=None, axis=None, keepdims=False)
        3 | upper fractile 
        4 | ecdf 
        5 | kernel density estimator pdf
        """
        self.metrics            = kwargs.get('metrics'  , ['mean_squared_error'])
        self.mquantiles_probs   = kwargs.get('prob'     , [0.9,0.99,0.999]) 
        self.moments2cal        = kwargs.get('moment'   , 1)

        print(r'   * Prediction scores ...')
        self.metrics_value = []
        #### calculate metrics from true value first
        metrics_value_true = []
        for imetrics in self.metrics:
            imetric2call = getattr(metrics_collections, imetrics.lower())

            if imetrics.lower() == 'mquantiles':
                imetrics_value = imetric2call(y_true, prob=self.mquantiles_probs)
                imetrics_value = imetrics_value.ravel().tolist()

            elif imetrics.lower() == 'moment': 
                imetrics_value = imetric2call(y_true, moment=self.moments2cal)
                imetrics_value = imetrics_value.ravel().tolist()
            else:
                imetrics_value = [0,]
            metrics_value_true= metrics_value_true+ imetrics_value
        self.metrics_value.append(metrics_value_true)

        metrics_value_true_iter = iter(metrics_value_true)
        y_pred = [y_pred, ] if isinstance(y_pred, (np.ndarray, np.generic)) else y_pred
        for iy_pred in y_pred:
            assert iy_pred.shape == y_true.shape, "Predict values and true values must have same shape, but get {} and {} instead".format(iy_pred.shape, y_true.shape)
            metrics_value_pred = []
            for imetrics in self.metrics:
                imetric2call = getattr(metrics_collections, imetrics.lower())

                if imetrics.lower() == 'mquantiles':
                    imetrics_value = imetric2call(iy_pred, prob=self.mquantiles_probs)
                    imetrics_value = imetrics_value.ravel().tolist()
                    print(r'     - {:<20s}: {:^10s} {:^10s} {:^10s}'.format(imetrics, 'True', 'Prediction', '%Error'))
                    for iprob, imquantiles in zip(self.mquantiles_probs, imetrics_value):
                        imetrics_value_true = next(metrics_value_true_iter)
                        error_perc = abs((imquantiles-imetrics_value_true)/imetrics_value_true) * 100.0 if imetrics_value_true else np.inf
                        print(r'       {:<18f}: {:^10.2f} {:^10.2f} {:^10.2f}'.format(iprob, imetrics_value_true, imquantiles, error_perc))

                elif imetrics.lower() == 'moment': 
                    imetrics_value = imetric2call(iy_pred, moment=self.moments2cal)
                    imetrics_value = imetrics_value.ravel().tolist()

                    print(r'     - {:<20s}: {:^10s} {:^10s} {:^10s}'.format(imetrics, 'True', 'Prediction', '%Error'))
                    for imoment, imoment_pred in zip(self.moments2cal,imetrics_value):
                        imetrics_value_true = next(metrics_value_true_iter)
                        error_perc = abs((imoment_pred-imetrics_value_true)/imetrics_value_true)* 100.0 if imetrics_value_true else np.inf
                        print(r'       {:<18s}: {:^10.2f} {:^10.2f} {:^10.2f}'.format(ordinal(imoment), imetrics_value_true, imoment_pred, error_perc))
                else:
                    imetrics_value = imetric2call(y_true, iy_pred)
                    imetrics_value_true = next(metrics_value_true_iter)
                    print(r'     - {:<20s}: {}'.format(imetrics, np.around(imetrics_value,2)))
                    imetrics_value = [imetrics_value,]

                metrics_value_pred = metrics_value_pred+ imetrics_value
            self.metrics_value.append(metrics_value_pred)
        return self.metrics_value
    
    def sample_y(self,X, nsamples=1, random_state=0):
        """
        Draw samples from Surrogate model and evaluate at X.

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
        raise NotImplementedError




        

