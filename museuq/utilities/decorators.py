import numpy as np
EPSILON = np.finfo(np.float32).eps

class NotFittedError(Exception):
    pass

def random_state(func):
    def wrapper(self, *args, **kwargs):
        if self.random_seed is None:
            return func(self, *args, **kwargs)
        else:
            original_state = np.random.get_state()
            np.random.seed(self.random_seed)

            result = func(self, *args, **kwargs)
            np.random.set_state(original_state)

            return result
    return wrapper


def import_object(object_name):
    """Import an object from its Fully Qualified Name."""
    package, name = object_name.rsplit('.', 1)
    return getattr(importlib.import_module(package), name)


def get_qualified_name(_object):
    """Return the Fully Qualified Name from an instance or class."""
    module = _object.__module__
    if hasattr(_object, '__name__'):
        _class = _object.__name__

    else:
        _class = _object.__class__.__name__

    return module + '.' + _class




def vectorize(func):
    """Allow a method that only accepts scalars to accept vectors too.

    This decorator has two different behaviors depending on the dimensionality of the
    array passed as an argument:

    **1-d array**

    It will work under the assumption that the `function` argument is a callable
    with signature::

        function(self, X, *args, **kwargs)

    where X is an scalar magnitude.

    In this case the arguments of the input array will be given one at a time, and
    both the input and output of the decorated function will have shape (n,).

    **2-d array**

    It will work under the assumption that the `function` argument is a callable with signature::

        function(self, X0, ..., Xj, *args, **kwargs)

    where `Xi` are scalar magnitudes.

    It will pass the contents of each row unpacked on each call. The input is espected to have
    shape (n, j), the output a shape of (n,)

    It will return a function that is guaranteed to return a `numpy.array`.

    Args:
        function(callable): Function that only accept and return scalars.

    Returns:
        callable: Decorated function that can accept and return :attr:`numpy.array`.

    """

    def decorated(self, X, *args, **kwargs):
        if not isinstance(X, np.ndarray):
            return func(self, X, *args, **kwargs)
        
        if len(X.shape) == 1:
            X = X.reshape([-1,1])
        if len(X.shape) == 2:
            return np.fromiter(
                    (func(self, *x, *args, **kwargs) for x in X), 
                    np.dtype('float64')
                    )
        else:
            raise ValueError('Arrays of dimensionality higher than 2 are not supperted')
    decorated.__doc__ = func.__doc__
    return decorated


def scalarize(func):
    """Allow methods that only accepts 1-d vectors to work with scalars.

    Args:
        function(callable): Function that accepts and returns vectors.

    Returns:
        callable: Decorated function that accepts and returns scalars.

    """
    def decorated(self, X, *args, **kwargs):
        scalar = not isinstance(X, np.ndarray)
        if scalar:
            X = np.array([X])
        result = func(self, X, *args, **kwargs)

        if scalar:
            result = result[0]
        return result

    decorated.__doc__ = func.__doc__
    return decorated

def check_valid_values(func):
    """Raises an exception if the given values are not supported.

    Args:
        function(callable): Method whose unique argument is a numpy.array-like object.

    Returns:
        callable: Decorated function

    Raises:
        ValueError: If there are missing or invalid values or if the dataset is empty.
    """
    def decorated(self, X, *args, **kwargs):
        if isinstance(X, pd.DataFrame):
            W = X.values
        else:
            W = X

        if not len(W):
            raise ValueError("Your dataset is empty.")
        if W.dtype not in [np.dtype('float64'), np.dtype('int64')]:
            raise ValueError("There are non-numerical values in your data.")
        if np.isnan(W).any().any():
            raise ValueError("There are nan values in your dataset.")
        return func(self, X, *args, **kwargs)
    return decorated


def missing_method_scipy_wrapper(function):
    """Raises a detailed exception when a method is not available."""
    def decorated(self, *args, **kwargs):
        message = (
            'Your tried to access `{method_name}` from {class_name}, but its not available.\n '
            'There can be multiple factors causing this, please feel free to open an issue in '
            'https://github.com/DAI-Lab/Copulas/issues/new'
        )

        params = {
            'method_name': function.__name__,
            'class_name': get_qualified_name(function.__self__.__class__)
        }

        raise NotImplementedError(message.format(**params))

    return decorated
