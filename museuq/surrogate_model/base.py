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

    def cal_scores(self,y_pred,y_true,**kwargs):
        """
        Calculate error metrics_value used to evaluate the accuracy of the approximation
            Reference: "On the accuracy of the polynomial chaos approximation R.V. Field Jr., M. Grigoriu"
        Parameters:	
            y_pred: array-like data from suoorgate model of shape (nsamples, noutput)
            y_true: array-like data from true model to be compared against. (nsamples, noutput)
            ps: Shape for each element in y_pred must be same as the shape of y_true

        kwargs:
            metrics: list of str: metric names to be calculated. Including:
                'explained_variance_score'
                'mean_absolute_error'
                'mean_squared_error'
                'mean_squared_log_error'
                'median_absolute_error'
                'r2_score':  1 - ESS(explained sum of squares)/TSS (total sum of squares)
                'r2_score_adj': adjusted r2
                'moment'
                'mquantilesupper_tail'
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
        metrics_value      = []

        ### default parameters for score functions
        sample_weight = kwargs.get('sample_weight'  , None)
        multioutput   = kwargs.get('multioutput'    , 'uniform_average')
        squared       = kwargs.get('squared'        , True)
        axis          = kwargs.get('axis'           , 0)
        num_predictor = kwargs.get('num_predictor'  , None)

        print(r'   * Calculating prediction metrics_value ...')
        #### calculate metrics from true value first
        metrics_value_true = []
        metrics_value_pred = []

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        assert y_pred.shape == y_true.shape
        for imetrics in self.metrics:
            imetric2call = getattr(metrics_collections, imetrics.lower())

            if imetrics.lower() == 'mquantiles':
                ### output format (nprob, noutput)
                imetrics_value_true = imetric2call(y_true, prob=self.mquantiles_probs, axis=axis, multioutput=multioutput)
                imetrics_value_pred = imetric2call(y_pred, prob=self.mquantiles_probs, axis=axis, multioutput=multioutput)
                imetrics_value = np.vstack((imetrics_value_true,imetrics_value_pred))
                metrics_value.append(imetrics_value)

            elif imetrics.lower() == 'moment': 
                ### output format (nmoment, noutput)
                moments2cal = np.sort(self.moments2cal)
                if 1 in moments2cal:
                    mean_idx = np.where(np.array(moments2cal)==1)[0]
                    mean_true = np.mean(y_true, axis=axis)
                    mean_pred = np.mean(y_pred, axis=axis)
                    if multioutput == 'uniform_average':
                        mean_pred = np.mean(mean_pred)
                        mean_true = np.mean(mean_true)
                    elif multioutput == 'raw_values':
                        mean_pred = mean_pred
                        mean_true = mean_true
                    else:
                        raise NotImplementedError

                imetrics_value_true = imetric2call(y_true, axis=axis, moment=moments2cal, multioutput=multioutput)
                imetrics_value_pred = imetric2call(y_pred, axis=axis, moment=moments2cal, multioutput=multioutput)

                imetrics_value_true[mean_idx] = mean_true
                imetrics_value_pred[mean_idx] = mean_pred
                imetrics_value = np.vstack((imetrics_value_true,imetrics_value_pred))
                metrics_value.append(imetrics_value)

            elif imetrics.lower() == 'r2_score_adj':
                ### output format (1,) or (noutput,)
                imetrics_value = imetric2call(y_true, y_pred,num_predictor=num_predictor, multioutput=multioutput)
                metrics_value.append(imetrics_value)
            else:
                ### output format (1,) or (noutput,)
                imetrics_value = imetric2call(y_true, y_pred, multioutput=multioutput)
                metrics_value.append(imetrics_value)

        ## print metric values 
        for imetrics_name, imetrics_value in zip(self.metrics, metrics_value):

            if imetrics_name.lower() == 'mquantiles':
                imetrics_value_true, imetrics_value_pred = imetrics_value
                imetrics_value_error = abs((imetrics_value_pred-imetrics_value_true)/imetrics_value_true) * 100.0 
                print(r'     - {:<30s}: {:^10s} {:^10s} {:^10s}'.format(imetrics_name, 'True', 'Prediction', '%Error'))
                for i, iprob in enumerate(self.mquantiles_probs):
                    print(r'       {:<30f}: {:^10.2f} {:^10.2f} {:^10.2f}'.format(iprob, np.around(imetrics_value_true[i],2), np.around(imetrics_value_pred[i], 2), np.around(imetrics_value_error[i],2)))

            elif imetrics_name.lower() == 'moment': 
                imetrics_value_true, imetrics_value_pred = imetrics_value
                imetrics_value_error = abs((imetrics_value_pred-imetrics_value_true)/imetrics_value_true) * 100.0 
                print(r'     - {:<30s}: {:^10s} {:^10s} {:^10s}'.format(imetrics_name, 'True', 'Prediction', '%Error'))
                for i, imoment in enumerate(self.moments2cal):
                    print(r'       {:<30s}: {:^10.2f} {:^10.2f} {:^10.2f}'.format(ordinal(imoment), np.around(imetrics_value_true[i],2), np.around(imetrics_value_pred[i], 2), np.around(imetrics_value_error[i], 2)))
            else:
                print(r'     - {:<30s}: {:^10.2f}'.format(imetrics_name, np.around(imetrics_value,2)))

        return metrics_value



        # print(r'   * Prediction metrics_value ...')
        # self.metrics_value = []
        # #### calculate metrics from true value first
        # metrics_value_true = []
        # for imetrics in self.metrics:
            # imetric2call = getattr(metrics_collections, imetrics.lower())

            # if imetrics.lower() == 'mquantiles':
                # imetrics_value = imetric2call(y_true, prob=self.mquantiles_probs, multioutput=multioutput)
                # imetrics_value = imetrics_value.ravel().tolist()

            # elif imetrics.lower() == 'moment': 
                # moments2cal = np.sort(self.moments2cal)
                # if 1 in moments2cal:
                    # mean_idx = np.where(np.array(moments2cal)==1)[0]
                    # mean = np.mean(y_true)
                # imetrics_value = imetric2call(y_true, moment=moments2cal, multioutput=multioutput)
                # imetrics_value[mean_idx] = mean
                # imetrics_value = imetrics_value.ravel().tolist()

            # else:
                # imetrics_value = [0,]
            # metrics_value_true= metrics_value_true+ imetrics_value
        # self.metrics_value.append(metrics_value_true)

        # metrics_value_true_iter = iter(metrics_value_true)
        # y_pred = [y_pred, ] if isinstance(y_pred, (np.ndarray, np.generic)) else y_pred
        # for iy_pred in y_pred:
            # assert iy_pred.shape == y_true.shape, "Predict values and true values must have same shape, but get {} and {} instead".format(iy_pred.shape, y_true.shape)
            # metrics_value_pred = []
            # for imetrics in self.metrics:
                # imetric2call = getattr(metrics_collections, imetrics.lower())

                # if imetrics.lower() == 'mquantiles':
                    # imetrics_value = imetric2call(iy_pred, prob=self.mquantiles_probs, multioutput=multioutput)
                    # imetrics_value = imetrics_value.ravel().tolist()
                    # print(r'     - {:<30s}: {:^10s} {:^10s} {:^10s}'.format(imetrics, 'True', 'Prediction', '%Error'))
                    # for iprob, imquantiles in zip(self.mquantiles_probs, imetrics_value):
                        # imetrics_value_true = next(metrics_value_true_iter)
                        # error_perc = abs((imquantiles-imetrics_value_true)/imetrics_value_true) * 100.0 if imetrics_value_true else np.inf
                        # print(r'       {:<30f}: {:^10.2f} {:^10.2f} {:^10.2f}'.format(iprob, imetrics_value_true, imquantiles, error_perc))

                # elif imetrics.lower() == 'moment': 
                    # moments2cal = np.sort(self.moments2cal)
                    # if 1 in moments2cal:
                        # mean_idx = np.where(np.array(moments2cal)==1)[0]
                        # mean = np.mean(y_true)
                    # imetrics_value = imetric2call(iy_pred, moment=moments2cal, multioutput=multioutput)
                    # imetrics_value[mean_idx] = mean
                    # imetrics_value = imetrics_value.ravel().tolist()

                    # print(r'     - {:<30s}: {:^10s} {:^10s} {:^10s}'.format(imetrics, 'True', 'Prediction', '%Error'))
                    # for imoment, imoment_pred in zip(self.moments2cal,imetrics_value):
                        # imetrics_value_true = next(metrics_value_true_iter)
                        # error_perc = abs((imoment_pred-imetrics_value_true)/imetrics_value_true)* 100.0 if imetrics_value_true else np.inf
                        # print(r'       {:<30s}: {:^10.2f} {:^10.2f} {:^10.2f}'.format(ordinal(imoment), imetrics_value_true, imoment_pred, error_perc))
                # else:
                    # imetrics_value = imetric2call(y_true, iy_pred, multioutput=multioutput)
                    # imetrics_value_true = next(metrics_value_true_iter)
                    # print(r'     - {:<30s}: {}'.format(imetrics, np.around(imetrics_value,2)))
                    # imetrics_value = [imetrics_value,]

                # metrics_value_pred = metrics_value_pred+ imetrics_value
            # self.metrics_value.append(metrics_value_pred)
        # return self.metrics_value

    
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




        

