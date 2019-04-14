import unittest
import numpy as np

from pall.containers import Data


class DataTest(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])

    def test_create_empty(self):
        Data()

    def test_create_correct(self):
        self.y = np.array([8, 4, 5])
        Data(X=self.X, y=self.y)

    def test_create_incorrect_length(self):
        self.y = np.array([1])
        with self.assertRaises(ValueError,
                               msg="Dimension 0 of X and y do not match. X: {0}, y: {1}"  # noqa
                                    .format(self.X.shape, self.y.shape)):
            Data(X=self.X, y=self.y)

    def test_create_no_shape(self):
        self.y = 'cat'
        with self.assertRaises(AttributeError,
                               msg="Data must have a 'shape' attribute."):
            Data(X=self.X, y=self.y)

    def test_create_not_iterable(self):
        self.y = 'cat'
        with self.assertRaises(AttributeError,
                               msg="Data must be iterable."):
            Data(X=self.X, y=self.y)

    def test_index_1(self):
        self.y = np.array([8, 4, 5])
        data = Data(X=self.X, y=self.y)
        X, y = data[1]
        self.assertTrue((X == self.X[1, :]).all())
        self.assertTrue((y == self.y[1]).all())

    def test_index_2(self):
        self.y = np.array([8, 4, 5])
        data = Data(X=self.X, y=self.y)
        X, y = data[[0, 2]]
        self.assertTrue((X == self.X[[0, 2], :]).all())
        self.assertTrue((y == self.y[[0, 2]]).all())
