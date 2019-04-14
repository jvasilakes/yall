from collections import namedtuple


Choice = namedtuple("Choice", ['X', 'y', "score"])


class Data(object):
    '''
    Data container object to hold features and labels.
    '''
    def __init__(self, X=None, y=None):
        self.X = X
        self.y = y
        self.__check_args(self.X, self.y)

    def __setattr__(self, name, value):
        '''
        Allows us to check that the input value satisfies
        certain constraints for x and y.
        '''
        if name == 'X' or name == 'y':
            if value is None:
                pass
            else:
                if not hasattr(value, '__iter__'):
                    raise AttributeError("Data must be iterable.")
                if not hasattr(value, 'shape'):
                    raise AttributeError("Data must have a 'shape' attribute.")
        self.__dict__[name] = value

    def __check_args(self, X, y):
        if X is not None:
            if len(X.shape) != 2:
                raise ValueError("X must be 2 dimensional.")
        if y is not None:
            if len(y.shape) != 1:
                raise ValueError("y must have shape (N,).")
        if X is not None and y is not None:
            if X.shape[0] != y.shape[0]:
                raise ValueError("Dimension 0 of X and y do not match. X: {0}, y: {1}"  # noqa
                                 .format(X.shape, y.shape))

    def __getitem__(self, key):
        '''
        Allow us to slice X and y together.
        data = Data(X: np.array(float), y: np.array(float))
        data[0] -> (X[0], y[0])
        '''
        X = self.X[key, :]
        y = self.y[key]
        return (X, y)

    def __str__(self):
        X_str = "X: {}".format(self.X)
        y_str = "y: {}".format(self.y)
        return "{0}\n{1}".format(X_str, y_str)

    def __repr__(self):
        X_str = "X: {}".format(self.X)
        y_str = "y: {}".format(self.y)
        return "{0}\n{1}\n".format(X_str, y_str)
